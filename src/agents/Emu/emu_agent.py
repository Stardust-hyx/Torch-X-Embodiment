# -*- coding: utf-8 -*-

import json
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from typing import List, Union, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torchvision.transforms import v2
except:
    import torchvision.transforms as v2
from torch.distributions import MultivariateNormal

from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.models.vae import DiagonalGaussianDistribution
from transformers import CLIPImageProcessor, BitsAndBytesConfig
# try:
#     from transformers.utils.bitsandbytes import replace_with_bnb_linear
# except:
#     from transformers.integrations.bitsandbytes import replace_with_bnb_linear

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import time

from .modeling_emu import Emu
from utils.misc import get_augment_transform, replace_with_bnb_linear

@dataclass
class EmuAgentOutput:
    pred_dist: MultivariateNormal
    text_loss: None
    image_reg_loss: None
    image_gen_loss: None

class EmuAgent(nn.Module):

    def __init__(
        self,
        multimodal_model: str,
        feature_extractor: str,
        safety_checker: str,
        scheduler: str,
        unet: str,
        vae: str,
        args,
        eva_size=224,
        eva_mean=(0.48145466, 0.4578275, 0.40821073),
        eva_std=(0.26862954, 0.26130258, 0.27577711),
        dropout_rate: float = 0.1,
        action_dim: int = 7,
        **kwargs,
    ):
        super().__init__()
        self.args = args

        torch.cuda.set_device(args.local_rank)
        self.unet = UNet2DConditionModel.from_pretrained(
            unet,
        ).bfloat16().cuda()
        self.vae = AutoencoderKL.from_pretrained(
            vae,
        ).bfloat16().cuda()
        self.scheduler = PNDMScheduler.from_pretrained(
            scheduler,
        )
        
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        
        quantization_config = None
        if args.quantization == 'int8':
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif args.quantization in ['fp4', 'nf4']:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type=args.quantization,
                bnb_4bit_use_double_quant=args.double_quant,
            )
            
        # if quantization_config:
        #     self.unet = replace_with_bnb_linear(
        #         self.unet,
        #         ['linear_1', 'linear_2', 'to_out.0', 'ff.net.0.proj', 'ff.net.2', 'time_emb_proj'],
        #         quantization_config=quantization_config
        #     )
        if args.lora:
            print('Patching LoRA...')
            from peft import LoftQConfig, LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=8,
                lora_alpha=8,
                use_rslora=True,
                target_modules=['to_q', 'to_k', 'to_v', 'to_out.0'],
                lora_dropout=0.1,
                # modules_to_save=['input_layernorm', 'post_attention_layernorm']
            )
            self.unet = get_peft_model(self.unet, lora_config)
            self.unet.print_trainable_parameters()
        
        # if self.args.img_loss_weight > 0:
        #     self.unet.enable_gradient_checkpointing()
        #     self.vae.enable_gradient_checkpointing()

        self.tgt_img_transform = v2.Compose(
                [
                    v2.Resize(512, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
                    v2.ToDtype(torch.bfloat16, scale=True),
                    v2.Normalize([0.5], [0.5]),
                ]
            )

        # self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        #     safety_checker,
        # )
        # self.feature_extractor = CLIPImageProcessor.from_pretrained(
        #     feature_extractor,
        # )

        self.emu_encoder = self.prepare_emu(args.emu_kwargs, multimodal_model, args, quantization_config)
        self.emu_encoder.decoder.add_action_tokens()
        if len(args.emu_kwargs['unfreeze_vit_layers']) > 0:
            self.emu_encoder.set_vit_layers_require_grad(True, layer_ids=args.emu_kwargs['unfreeze_vit_layers'])
        if len(args.emu_kwargs['unfreeze_llm_layers']) > 0:
            self.emu_encoder.set_llm_layers_require_grad(True, layer_ids=args.emu_kwargs['unfreeze_llm_layers'])
        if args.gradient_checkpointing:
            self.emu_encoder.set_grad_checkpointing()

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.augment = args.augment
        self.augment_kwargs = args.augment_kwargs
        if self.augment and self.augment_kwargs:
            self.augment_kwargs['random_resized_crop']['scale']  = (0.8, 1.0) #TODO
            self.augment_transform = get_augment_transform(self.augment_kwargs)
        # self.transform = v2.Normalize(mean=eva_mean, std=eva_std)
        self.transform = v2.Compose(
                [
                    v2.ToDtype(torch.bfloat16, scale=True),
                    v2.Normalize(eva_mean, eva_std),
                ]
            )

        self.ori_img_placeholder = args.img_placeholder
        self.ori_act_placeholder = args.act_placeholder
        self.image_placeholder = self.emu_encoder.image_placeholder
        self.action_placeholder = self.emu_encoder.action_placeholder

        hidden_dim = self.emu_encoder.decoder.lm.config.hidden_size
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.action_feature_fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.Dropout(dropout_rate, inplace=True),
            nn.SiLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout(dropout_rate, inplace=True),
            nn.SiLU(inplace=True),
            # nn.Linear(256, 256),
            # nn.Dropout(dropout_rate, inplace=True),
            # nn.SiLU(inplace=True),
        ).cuda()
        self.action_mean_linear = nn.Linear(256, action_dim).cuda()

        self.register_buffer("fixed_std", torch.eye(action_dim))

    def forward(self, prompts, obs_imgs, goal_imgs, future_imgs) -> EmuAgentOutput:
        device = self.emu_encoder.ln_visual.weight.device
        dtype = self.emu_encoder.ln_visual.weight.dtype
        images, input_ids, attention_mask, goal_imgs_tgt, future_imgs_tgt = self._prepare_emu_input(prompts, obs_imgs, goal_imgs, future_imgs, device, dtype)
        # print(images.shape, input_ids.shape)
        output = self.emu_encoder.forward(images, input_ids, attention_mask)

        feature = self.action_feature_fc(output.action_feature)
        means = self.action_mean_linear(feature)
        dist = MultivariateNormal(means.float(), scale_tril=self.fixed_std.float())

        image_gen_loss = None
        if (self.training or self.args.check_loss_when_eval) and self.args.img_loss_weight > 0 and (goal_imgs_tgt is not None or future_imgs_tgt is not None):
            # image_feature = self.emu_encoder.decoder.lm.stu_regress_head(output.image_feature)
            # image_feature = image_feature.reshape((-1, self.emu_encoder.n_causal, self.hidden_dim))
            image_feature = output.image_feature
            if self.args.cfg:
                # random set 10% feature to be empty for classifier-free guidance
                image_feature[torch.rand(image_feature.size(0)) < 0.1] = 0
            image_gen_loss = self.compute_image_loss(image_feature, goal_imgs_tgt, future_imgs_tgt)
        
        # return EmuAgentOutput(
        #     pred_dist=dist,
        #     text_loss=output.llm_loss,
        #     image_loss=image_loss
        # )
        return (dist, output.llm_loss, output.img_regress_loss, image_gen_loss)
    
    def _prepare_emu_input(self, prompts, obs_imgs, goal_imgs, future_imgs, device, dtype):
        _, C, H, W = obs_imgs.shape
        if goal_imgs is not None and future_imgs is not None:
            images = torch.cat((obs_imgs, goal_imgs, future_imgs), dim=1).reshape((-1, C, H, W))
        elif goal_imgs is not None:
            images = torch.cat((obs_imgs, goal_imgs), dim=1).reshape((-1, C, H, W))
        elif future_imgs is not None:
            images = torch.cat((obs_imgs, future_imgs), dim=1).reshape((-1, C, H, W))
        else:
            images = obs_imgs

        images = images.to(device)
        if self.training and self.augment:
            images = self.augment_transform(images)

        goal_imgs_tgt, future_imgs_tgt = None, None
        if self.training or self.args.check_loss_when_eval:
            if goal_imgs is not None and future_imgs is not None:
                goal_imgs_tgt = self.tgt_img_transform(images[1::3])
                future_imgs_tgt = self.tgt_img_transform(images[2::3])
            elif goal_imgs is not None:
                goal_imgs_tgt = self.tgt_img_transform(images[1::2])
            elif future_imgs is not None:
                future_imgs_tgt = self.tgt_img_transform(images[1::2])

        # images = self.transform(images.type(dtype) / 255.0)
        images = self.transform(images)
        
        prompts = [t.replace(self.ori_img_placeholder, self.image_placeholder) for t in prompts]
        prompts = [t.replace(self.ori_act_placeholder, self.action_placeholder) for t in prompts]
        inputs = self.emu_encoder.decoder.tokenizer(prompts, padding="longest", return_tensors="pt")
        attention_mask = inputs.attention_mask.to(device)
        input_ids = inputs.input_ids.to(device)

        return images, input_ids, attention_mask, goal_imgs_tgt, future_imgs_tgt


    @torch.no_grad()
    def sample_actions(
        self, prompt, obs_img, feature_for_gen_goal_img, goal_img=None, argmax=True, do_gen_goal_img=False, guidance_scale: float = 7.5,
    ):
        if not isinstance(obs_img, torch.Tensor):
            obs_img = torch.tensor(obs_img, device=self.fixed_std.device)
        else:
            obs_img = obs_img.to(self.fixed_std.device)

        if goal_img is not None:
            if not isinstance(goal_img, torch.Tensor):
                goal_img = torch.tensor(goal_img, device=self.fixed_std.device)
            else:
                goal_img = goal_img.to(self.fixed_std.device)
            dist = self.forward([prompt], obs_img.unsqueeze_(0), goal_img.unsqueeze_(0), None)[0]
            gen_goal_img = None
        else:
            dist, feature_for_gen_goal_img, gen_goal_img = self.gen_goal_and_predict_act(
                prompt, obs_img, feature_for_gen_goal_img, do_gen_goal_img, guidance_scale=guidance_scale
            )

        if argmax:
            actions = dist.mode
        else:
            actions = dist.sample()
        return (actions, feature_for_gen_goal_img, gen_goal_img)
    
    @torch.no_grad()
    def gen_goal_and_predict_act(
        self, prompt, obs_img, feature_for_gen_goal_img=None, do_gen_goal_img=False,
        placeholder: str = "[<IMG_PLH>]",
        height: int = 512, width: int = 512, num_inference_steps: int = 50, guidance_scale: float = 7.5,
    ):
        device = self.emu_encoder.ln_visual.weight.device
        dtype = self.emu_encoder.ln_visual.weight.dtype
        do_classifier_free_guidance = guidance_scale > 1.0

        goal_img_start_idx = prompt.rfind(placeholder)
        prompt_for_gen_goal_img = prompt[:goal_img_start_idx]
        image_prompt = self.transform(obs_img.unsqueeze_(0))
        feature_for_gen_goal_img, obs_img_embed = self._prepare_and_encode_inputs(
            text_prompt=prompt_for_gen_goal_img, image_prompt=image_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            target_image_embeds=feature_for_gen_goal_img
        )

        goal_img = None
        if do_gen_goal_img:
            goal_img = self.denoise_and_output_img(
                feature_for_gen_goal_img, 1, height, width,
                num_inference_steps, guidance_scale, do_classifier_free_guidance, device, dtype
            )

        goal_img_embed = feature_for_gen_goal_img[0]
        image_features = torch.stack((obs_img_embed, goal_img_embed))

        prompts = [prompt]
        prompts = [t.replace(self.ori_img_placeholder, self.image_placeholder) for t in prompts]
        prompts = [t.replace(self.ori_act_placeholder, self.action_placeholder) for t in prompts]
        inputs = self.emu_encoder.decoder.tokenizer(prompts, padding="longest", return_tensors="pt")
        attention_mask = inputs.attention_mask.to(device)
        input_ids = inputs.input_ids.to(device)
        # output from hf lm model
        output = self.emu_encoder.decoder.forward(image_features, text_input=input_ids, text_mask=attention_mask)

        feature = self.action_feature_fc(output.action_feature)
        means = self.action_mean_linear(feature)
        dist = MultivariateNormal(means.float(), scale_tril=self.fixed_std.float())

        return dist, feature_for_gen_goal_img, goal_img
    
    @torch.no_grad()
    def generate_img(
        self,
        inputs: List[Union[Image.Image, str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> Image.Image:

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self.emu_encoder.ln_visual.weight.device
        dtype = self.emu_encoder.ln_visual.weight.dtype

        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode input prompt
        batch_size = 1
        prompt_embeds, _ = self._prepare_and_encode_inputs(
            inputs,
            device,
            dtype,
            do_classifier_free_guidance,
        )

        return self.denoise_and_output_img(
            prompt_embeds,
            batch_size, height, width,
            num_inference_steps, guidance_scale, do_classifier_free_guidance,
            device, dtype,
        )

    def _prepare_and_encode_inputs(
        self,
        inputs: List[Union[str, Image.Image]] = None,
        device: torch.device = "cpu",
        dtype: str = torch.float32,
        do_classifier_free_guidance: bool = False,
        placeholder: str = "[<IMG_PLH>]",
        text_prompt: str = None,
        image_prompt: list = None,
        target_image_embeds = None,
    ) -> torch.Tensor:
        if text_prompt is None and image_prompt is None:
            text_prompt = ""
            image_prompt = []
            for x in inputs:
                if isinstance(x, str):
                    text_prompt += x
                else:
                    text_prompt += placeholder
                    image_prompt.append(self.transform(x))
            # Nx3x224x224
            if len(image_prompt) == 0:
                image_prompt = None
            else:
                image_prompt = torch.stack(image_prompt)
                image_prompt = image_prompt.type(dtype).to(device)
        assert text_prompt is not None and image_prompt is not None

        if do_classifier_free_guidance:
            text_prompt = [text_prompt, ""]
        else:
            text_prompt = [text_prompt]

        prompt_image_embeds = self.emu_encoder.visual.forward_features(image_prompt)
        prompt_image_embeds = self.emu_encoder.ln_visual(prompt_image_embeds)
        prompt_image_embeds = self.emu_encoder.cformer(prompt_image_embeds)
        prompt_image_embeds = prompt_image_embeds.view(-1, prompt_image_embeds.shape[-1])
        if target_image_embeds is None:
            target_image_embeds = self.emu_encoder.generate_image(
                text=text_prompt,
                prompt_image_embeds=prompt_image_embeds,
                placeholder=placeholder,
            )

        return target_image_embeds, prompt_image_embeds

    def denoise_and_output_img(self,
        prompt_embeds,
        batch_size, height, width,
        num_inference_steps, guidance_scale, do_classifier_free_guidance,
        device, dtype,
    ):
        # 2. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. Prepare latent variables
        # Bx4xHxW
        shape = (
            batch_size,
            self.unet.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor
        )
        latents = torch.randn(shape, device=device, dtype=dtype)

        # 4. Denoising loop
        for t in timesteps:
            # expand the latents if we are doing classifier free guidance
            # 2B x 4 x H x W
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 10. Convert to PIL
        image = self.numpy_to_pil(image)
        return image[0]
    
    def decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def numpy_to_pil(self, images: np.ndarray) -> List[Image.Image]:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images
    
    def prepare_emu(
        self,
        model_cfg: dict,
        model_path: str,
        args: dict,
        quantization_config=None,
    ) -> Emu:
        model = Emu(**model_cfg, cast_dtype=torch.float, args=args).bfloat16()

        if quantization_config is not None:
            model = replace_with_bnb_linear(
                model,
                ["qkv", "fc1", "fc2", "attn.proj", "projection", "lm_head", "stu_regress_head", 'q', 'k', 'v', 'o', "wi", "wo"],
                quantization_config=quantization_config
            )
        # print("visual", model.visual.blocks[0].attn.qkv.weight)
        # print("decoder", model.decoder.lm.model.layers[0].self_attn.q_proj.weight)

        # Delay loading to prevent OUT OF RAM
        time.sleep(10 * self.args.local_rank)
        
        model = load_checkpoint_and_dispatch(
            model, model_path, device_map={
                "visual": "cuda", "ln_visual": "cuda", "cformer": "cuda", "decoder": "cuda"
            }
        )
        # state_dict = torch.load(model_path)
        # model.load_state_dict(state_dict, strict=True)
        # print("visual", model.visual.blocks[0].attn.qkv.weight)
        # print("decoder", model.decoder.lm.model.layers[0].self_attn.q_proj.weight)

        model = model.cuda()

        # Discard unused modules
        model.visual.head = None
        model.cformer.cformer.embed_tokens = None
        # Replace ineffective dropout with Identity function to save memory
        if model.visual.pos_drop.p == 0:
            model.visual.pos_drop = nn.Identity()
        for block in model.visual.blocks:
            if block.attn.attn_drop.p == 0:
                block.attn.attn_drop = nn.Identity()
            if block.attn.proj_drop.p == 0:
                block.attn.proj_drop = nn.Identity()
            if block.mlp.drop.p == 0:
                block.mlp.drop = nn.Identity()

        if args.lora:
            print('Patching LoRA...')
            from peft import LoftQConfig, LoraConfig, get_peft_model

            # lora_config = LoraConfig(
            #     r=8,
            #     lora_alpha=2,
            #     use_rslora=True,
            #     target_modules=['attn.proj'],
            #     lora_dropout=0.1,
            #     # modules_to_save=['patch_embed.proj']
            # )
            # model.visual = get_peft_model(model.visual, lora_config)
            # model.visual.print_trainable_parameters()
            
            lora_config = LoraConfig(
                r=8,
                lora_alpha=2,
                use_rslora=True,
                target_modules=['q', 'k', 'v', 'projection'],
                lora_dropout=0.1,
                layers_to_transform=[i for i in range(len(model.cformer.cformer.block)) if i % 2 != 0],
                layers_pattern='block'
            )
            model.cformer = get_peft_model(model.cformer, lora_config)
            model.cformer.print_trainable_parameters()
            
            model.cformer.causal_tokens.requires_grad_(True)
            
            lora_config = LoraConfig(
                r=64,
                lora_alpha=32,
                use_rslora=True,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                lora_dropout=0.15,
                # modules_to_save=['input_layernorm', 'post_attention_layernorm']
                layers_to_transform=[i for i in range(len(model.decoder.lm.model.layers)) if i % 3 == 0],
                layers_pattern='layers'
            )
            model.decoder.lm = get_peft_model(model.decoder.lm, lora_config)
            model.decoder.lm.print_trainable_parameters()
        
        # print("model.decoder.lm.base_model.model.lm_head.weight.requires_grad")
        # print(model.decoder.lm.base_model.model.lm_head.weight.requires_grad, flush=True)
        # print("model.decoder.lm.base_model.model.stu_regress_head.weight.requires_grad")
        # print(model.decoder.lm.base_model.model.stu_regress_head.weight.requires_grad, flush=True)
            
        return model

    @classmethod
    def from_pretrained(cls, path: str, args: dict, **kwargs):
        multimodal_model = kwargs.pop("multimodal_model", None)
        feature_extractor = kwargs.pop("feature_extractor", None)
        safety_checker = kwargs.pop("safety_checker", None)
        scheduler = kwargs.pop("scheduler", None)
        unet = kwargs.pop("unet", None)
        vae = kwargs.pop("vae", None)

        check_if_none = lambda x, y: y if x is None else x

        multimodal_model = check_if_none(multimodal_model, f"{path}/multimodal_encoder/pytorch_model.bin")
        feature_extractor = check_if_none(feature_extractor, f"{path}/feature_extractor")
        safety_checker = check_if_none(safety_checker, f"{path}/safety_checker")
        scheduler = check_if_none(scheduler, f"{path}/scheduler")
        unet = check_if_none(unet, f"{path}/unet")
        vae = check_if_none(vae, f"{path}/vae")

        return cls(
            multimodal_model=multimodal_model,
            feature_extractor=feature_extractor,
            safety_checker=safety_checker,
            scheduler=scheduler,
            unet=unet,
            vae=vae,
            args=args,
            **kwargs,
        )

    def renew_action_linear(self, action_dim):
        if action_dim != self.action_dim:
            hidden_dim = self.action_mean_linear.weight.data.shape[1]
            self.action_mean_linear = nn.Linear(hidden_dim, action_dim).cuda()
            self.register_buffer("fixed_std", torch.eye(action_dim))

    def set_feature_layers_require_grad(self, flag: bool):
        self.emu_encoder.set_vit_layers_require_grad(flag)
        # self.emu_encoder.cformer.requires_grad_(flag)
        self.emu_encoder.cformer.causal_tokens.requires_grad_(flag)
        for n, p in self.emu_encoder.cformer.named_parameters():
            if "lora" in n:
                assert p.requires_grad != flag, n
                p.requires_grad = flag
        for n, p in self.emu_encoder.decoder.named_parameters():
            if "lora" in n:
                assert p.requires_grad != flag, n
                p.requires_grad = flag
        self.emu_encoder.set_llm_layers_require_grad(flag)
        
        if self.args.local_rank == 0:
            for n, p in self.named_parameters():
                if p.requires_grad:
                    print(n)
        
    def set_action_fc_require_grad(self, flag: bool):
        self.action_feature_fc.requires_grad_(flag)

    def set_action_linear_require_grad(self, flag: bool):
        self.action_mean_linear.requires_grad_(flag)
        
    def compute_image_loss(self, mapping_feature, goal_imgs, future_imgs):
        if goal_imgs is not None and future_imgs is not None:
            _, C, H, W = goal_imgs.shape
            tgt_images = torch.cat((goal_imgs, future_imgs), dim=1).reshape((-1, C, H, W))
        else:
            tgt_images = goal_imgs if goal_imgs is not None else future_imgs

        latents = self.vae.encode(tgt_images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        # timesteps = timesteps.long()

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        target = noise
        # print(mapping_feature.shape)
        model_pred = self.unet(noisy_latents, timesteps, mapping_feature).sample

        # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        snr = self.compute_snr(timesteps)
        mse_loss_weights = (
            torch.stack([snr, 5 * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        )
        # We first calculate the original loss. Then we mean over the non-batch dimensions and
        # rebalance the sample-wise losses with their respective loss weights.
        # Finally, we take the mean of the rebalanced loss.
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()
        return loss
    
    def compute_snr(self,timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr