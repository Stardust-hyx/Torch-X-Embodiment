from functools import partial
from typing import List, Optional

import torch

from torch import nn
from .causal_former import CausalFormer
from .model import MultimodalCfg, CLIPVisionCfg, VLadapterCfg, _build_vision_tower
from .transformer import LayerNorm
from .prediction_mixin import PredictClassMixin

try:
    from transformers import BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList, \
        MaxLengthCriteria
except ImportError as e:
    pass

from transformers.generation.configuration_utils import GenerationConfig
GENERATION_CONFIG = GenerationConfig(bos_token_id=1, eos_token_id=2, pad_token_id=32000)

from .modeling_llama import RegressCausalLMOutputWithPast

class Emu(nn.Module, PredictClassMixin):
    def __init__(
        self,
        embed_dim,
        multimodal_cfg: MultimodalCfg,
        vision_cfg: CLIPVisionCfg,
        vladapter_cfg: VLadapterCfg,
        unfreeze_vit_layers: list,
        unfreeze_llm_layers: list,
        gradient_checkpointing: bool = False,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        pad_id: int = 0,
        args=None,
        apply_lemmatizer=False,
        prompt=None
    ):
        super().__init__()

        self.args = args

        multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        vladapter_cfg = VLadapterCfg(**vladapter_cfg) if isinstance(vladapter_cfg, dict) else vladapter_cfg

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            cast_dtype=cast_dtype,
        )
        if vision_cfg.freeze:
            self.visual.requires_grad_(False)

        norm_layer = partial(LayerNorm, eps=1e-6)
        
        self.ln_visual = norm_layer(vision_cfg.width)
        nn.init.constant_(self.ln_visual.bias, 0)
        nn.init.constant_(self.ln_visual.weight, 1.0)
        if vision_cfg.freeze:
            self.ln_visual.requires_grad_(False)

        from .modeling_llama import LLaMAForClsAndRegression
        from accelerate import init_empty_weights
        if args.emu_ckpt:
            with init_empty_weights():
                self.decoder = LLaMAForClsAndRegression(args=args)
        else:
            self.decoder = LLaMAForClsAndRegression(args=args)

        if multimodal_cfg.freeze:
            self.decoder.requires_grad_(False)

        if args.emu_ckpt:
            with init_empty_weights():
                self.cformer = CausalFormer(args=args,
                                        n_causal=vladapter_cfg.n_causal,
                                        vision_width=vision_cfg.width,
                                        output_dim=self.decoder.config.d_model)
        else:
            self.cformer = CausalFormer(args=args,
                                        n_causal=vladapter_cfg.n_causal,
                                        vision_width=vision_cfg.width,
                                        output_dim=self.decoder.config.d_model)
        self.cformer.requires_grad_(False)
        # self.cformer.causal_tokens.requires_grad_(True)
        # self.cformer.projection.requires_grad_(True)

        self.n_causal = vladapter_cfg.n_causal
        self.pad_id = pad_id

        self.prompt = prompt
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.image_placeholder = "[IMG]" + "<image>" * self.n_causal + "[/IMG]"
        # self.action_placeholder = "[ACT]" + "<action>" + "[/ACT]"
        self.action_placeholder = "<action>" 

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # self.visual.set_grad_checkpointing(enable)
        # self.cformer.set_grad_checkpointing()
        self.decoder.set_grad_checkpointing()
        
    def set_vit_layers_require_grad(self, flag: bool, layer_ids=[0,-1]):
        # print(self.visual)
        for i in range(len(self.visual.blocks)):
            if i % 2 == 0:
                continue
            self.visual.blocks[i].attn.qkv.requires_grad_(flag)
        # self.ln_visual.requires_grad_(flag)
        return
        
    def set_llm_layers_require_grad(self, flag: bool, layer_ids=[0,-1]):
        # print(self.decoder.lm.model)
        # for i in layer_ids:
        #     self.decoder.lm.model.layers[i].requires_grad_(flag)
        self.decoder.lm.base_model.model.model.embed_tokens.requires_grad_(False)
        self.decoder.lm.base_model.model.model.norm.requires_grad_(flag)
        self.decoder.lm.base_model.model.lm_head.requires_grad_(False)
        self.decoder.lm.base_model.model.stu_regress_head.requires_grad_(flag)
        return

    def forward(self, image, text_input, input_mask, text_output=None, output_mask=None, image_latent=None,
                image_features=None) -> RegressCausalLMOutputWithPast:
        # [B, C, H, W] --> [B, n_patch, C_vis]
        if image_latent is None or image_features is None:
            image_features = self.visual.forward_features(image)
        # ln for visual features
        image_features = self.ln_visual(image_features)
        # [B, n_patch, C_vis] --> [B, n_causal, C_llm]
        image_features = self.cformer(image_features)
        # output from hf lm model
        output = self.decoder.forward(image_features, text_input=text_input, text_output=text_output, text_mask=input_mask,
                            output_mask=output_mask)
        return output

    @torch.no_grad()
    def generate(
        self,
        samples,
        do_sample=False,
        num_beams=5,
        max_new_tokens=50,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=0.0,
        num_captions=1,
        temperature=1,
        penalty_alpha=None,  # contrastive search
        top_k=None,
        no_repeat_ngram_size=None,
        **kwargs,
    ):
        GENERATION_CONFIG.pad_token_id = self.decoder.tokenizer.pad_token_id
        GENERATION_CONFIG.bos_token_id = self.decoder.tokenizer.bos_token_id
        GENERATION_CONFIG.eos_token_id = self.decoder.tokenizer.eos_token_id
        
        image = samples["image"]
        if image is not None:
            image = image.to(dtype=torch.bfloat16)
            image_features = self.ln_visual(self.visual.forward_features(image))
            image_features = self.cformer(image_features).squeeze().to(dtype=torch.bfloat16)

        prompt = samples["prompt"] if "prompt" in samples.keys() else self.prompt

        from .modeling_llama import LLaMAForClsAndRegression
        if isinstance(self.decoder, LLaMAForClsAndRegression):
            self.decoder.tokenizer.padding_side = "left"

        input_tokens = self.decoder.tokenizer(
            prompt, 
            padding="longest", 
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.args.device)

        self.decoder.tokenizer.padding_side = "right"

        input_ids = input_tokens.input_ids
        encoder_atts = input_tokens.attention_mask

        img_token_id = self.decoder.tokenizer.convert_tokens_to_ids("<image>")  # 32003
        img_token_idx_list = input_ids.eq(img_token_id)

        with torch.amp.autocast(device_type=self.args.device.type, dtype=torch.bfloat16):
            if self.args.lora:
                inputs_embeds = self.decoder.lm.model.model.embed_tokens(input_ids)
            else:
                inputs_embeds = self.decoder.lm.model.embed_tokens(input_ids)

            if image is not None:
                image_features = image_features.reshape(-1, image_features.shape[-1])
                inputs_embeds[img_token_idx_list] = image_features

            inputs_embeds = inputs_embeds
            encoder_atts = encoder_atts

            outputs = self.decoder.lm.generate(
                generation_config=GENERATION_CONFIG,
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                penalty_alpha=penalty_alpha,
                top_k=top_k,
                no_repeat_ngram_size=no_repeat_ngram_size,
                **kwargs,
            )

            output_text = self.decoder.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            ) 

        return output_text

    @torch.no_grad()
    def generate_image(
        self,
        text: List[str],
        prompt_image_embeds: Optional[torch.Tensor] = None,
        placeholder: str = "[<IMG_PLH>]",
        act_placeholder: str = "[<ACT_PLH>]",
    ) -> torch.Tensor:
        IMAGE, BOI = self.decoder.tokenizer.convert_tokens_to_ids(["<image>", "[IMG]"])
        device = self.ln_visual.weight.device

        text = [t.replace(placeholder, self.image_placeholder) for t in text]
        text = [t.replace(act_placeholder, self.action_placeholder) for t in text]

        """ Preparing """
        inputs = self.decoder.tokenizer(text, padding="longest", return_tensors="pt")
        attention_mask = inputs.attention_mask.to(device)
        input_ids = inputs.input_ids.to(device)
        # Space Mark (' ') at the end of text will introduce an extra token, which should be removed here!
        if input_ids.shape[-1] == self.decoder.tokenizer([f"{t}[IMG]" for t in text], padding="longest", return_tensors="pt").input_ids.shape[-1]:
            attention_mask = attention_mask[:, :-1]
            input_ids = input_ids[:, :-1]
        try:
            text_embeds = self.decoder.lm.model.model.embed_tokens(input_ids)
        except:
            text_embeds = self.decoder.lm.model.embed_tokens(input_ids)

        image_idx = (input_ids == IMAGE)
        cumsum_idx = torch.flip(torch.cumsum(torch.flip(image_idx, dims=[1]), dim=1), dims=[1])
        # print(f'cumsum_idx.shape\n{cumsum_idx.shape}')
        # print(f'cumsum_idx\n{cumsum_idx}')
        if prompt_image_embeds is not None:
            prompt_idx = torch.logical_and(image_idx, cumsum_idx > 0)
            # print(f'prompt_idx\n{prompt_idx}')
            text_embeds[prompt_idx] = prompt_image_embeds

        try:
            outputs = self.decoder.lm.model.model(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        except:
            outputs = self.decoder.lm.model(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        past_key_values = outputs.past_key_values
        B, _, C = text_embeds.shape
        """ End Preparing """

        """ Begin """
        target_image_embeds = None
        for num_img_token in range(self.n_causal):
            if num_img_token == 0:
                text = [f"{t}[IMG]" for t in text]
            else:
                text = [f"{t}<image>" for t in text]
            # print(f'text\n{text}')

            inputs = self.decoder.tokenizer(text, padding="longest", return_tensors="pt")
            attention_mask = inputs.attention_mask.to(device)
            input_ids = inputs.input_ids.to(device)
            # print(f'input_ids\n{input_ids}')
            try:
                text_embeds = self.decoder.lm.model.model.embed_tokens(input_ids)
            except:
                text_embeds = self.decoder.lm.model.embed_tokens(input_ids)
            # print(f'text_embeds.shape\n{text_embeds.shape}')

            image_idx = (input_ids == IMAGE)
            cumsum_idx = torch.flip(torch.cumsum(torch.flip(image_idx, dims=[1]), dim=1), dims=[1])
            if target_image_embeds is not None:
                target_idx = torch.logical_and(image_idx, torch.logical_and(cumsum_idx > 0, cumsum_idx <= num_img_token))
                # print(f'target_idx\n{target_idx}')
                text_embeds[target_idx] = target_image_embeds

            try:
                outputs = self.decoder.lm.model.model(
                    inputs_embeds=text_embeds[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            except:
                outputs = self.decoder.lm.model(
                    inputs_embeds=text_embeds[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

            hidden_states = outputs.hidden_states[-1]
            new_image_embeds = hidden_states[:, -1]
            new_image_embeds = self.decoder.lm.stu_regress_head(new_image_embeds)
            if target_image_embeds is None:
                target_image_embeds = new_image_embeds
            else:
                new_image_embeds = new_image_embeds.view(B, 1, C)
                target_image_embeds = target_image_embeds.view(B, -1, C)
                target_image_embeds = torch.cat((target_image_embeds, new_image_embeds), dim=1)
                target_image_embeds = target_image_embeds.view(-1, C)
            # print()

            past_key_values = outputs.past_key_values

        # print(f'target_image_embeds.shape\n{target_image_embeds.shape}')
        target_image_embeds = target_image_embeds.view(B, -1, C)
        # print(f'target_image_embeds.shape\n{target_image_embeds.shape}')
        # print()

        return target_image_embeds
