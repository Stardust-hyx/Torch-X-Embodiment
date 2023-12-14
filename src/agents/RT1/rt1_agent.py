from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from utils.misc import get_augment_transform
from .robotic_transformer_pytorch import *

@dataclass
class RT1AgentOutput:
    pred_dist: MultivariateNormal

class RT1Agent(RT1):

    def __init__(
        self,
        args,
        vit: MaxViT,
        num_actions = 7,
        depth = 6,
        heads = 8,
        dim_head = 64,
        token_learner_ff_mult = 2,
        token_learner_num_layers = 2,
        token_learner_num_output_tokens = 8,
        cond_drop_prob = 0.2,
        use_attn_conditioner = False,
        conditioner_kwargs: dict = dict(),
        dropout_rate = 0.1,
    ) -> None:
        super().__init__(vit=vit, num_actions=num_actions, conditioner_kwargs=conditioner_kwargs)

        self.to_logits = nn.Sequential(
            LayerNorm(vit.embed_dim),
            nn.Linear(vit.embed_dim, vit.embed_dim),
            nn.Dropout(dropout_rate),
            nn.SiLU(inplace=True),
            nn.Linear(vit.embed_dim, vit.embed_dim),
            nn.Dropout(dropout_rate),
            nn.SiLU(inplace=True),
            nn.Linear(vit.embed_dim, num_actions),
        )

        self.augment = args.augment
        self.augment_kwargs = args.augment_kwargs
        if self.augment and self.augment_kwargs:
            self.augment_transform = get_augment_transform(args.augment_kwargs)

        self.register_buffer("fixed_std", torch.eye(num_actions))

    @classifier_free_guidance
    def forward(self, texts, video, goal_imgs, feature_imgs, cond_drop_prob=None) -> RT1AgentOutput:
        texts = list(texts)
        video = video.to(self.fixed_std.device)
        if self.training and self.augment:
            video = self.augment_transform(video)
        
        depth = self.transformer_depth
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        frames, device = video.shape[1], self.fixed_std.device

        cond_fns = self.conditioner(
            texts,
            cond_drop_prob = cond_drop_prob,
            repeat_batch = (*((frames,) * self.num_vit_stages), *((1,) * self.transformer_depth * 2))
        )

        vit_cond_fns, transformer_cond_fns = cond_fns[:-(depth * 2)], cond_fns[-(depth * 2):]

        # video = rearrange(video, 'b c f h w -> b f c h w')
        images, packed_shape = pack_one(video, '* c h w')
        images = images.to(torch.float32) / 127.5 - 1.0

        tokens = self.vit(
            images,
            texts = texts,
            cond_fns = vit_cond_fns,
            cond_drop_prob = cond_drop_prob,
            return_embeddings = True
        )

        tokens = unpack_one(tokens, packed_shape, '* c h w')
        learned_tokens = self.token_learner(tokens)

        learned_tokens = rearrange(learned_tokens, 'b f c n -> b (f n) c')

        # causal attention mask
        attn_mask = torch.ones((frames, frames), dtype = torch.bool, device = device).triu(1)
        attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1 = self.num_learned_tokens, r2 = self.num_learned_tokens)

        # sinusoidal positional embedding
        pos_emb = posemb_sincos_1d(frames, learned_tokens.shape[-1], dtype = learned_tokens.dtype, device = learned_tokens.device)

        learned_tokens = learned_tokens + repeat(pos_emb, 'n d -> (n r) d', r = self.num_learned_tokens)

        # attention
        attended_tokens = self.transformer(learned_tokens, cond_fns = transformer_cond_fns, attn_mask = ~attn_mask)

        pooled = reduce(attended_tokens, 'b (f n) d -> b f d', 'mean', f = frames)

        logits = self.to_logits(pooled)

        dist = MultivariateNormal(logits, scale_tril=self.fixed_std)
        return RT1AgentOutput(
            pred_dist=dist
        )
    
    @torch.no_grad()
    def sample_actions(self, obs_img, goal_img, argmax=True):
        if not isinstance(obs_img, torch.Tensor):
            obs_img = torch.tensor(obs_img, device=self.fixed_std.device).unsqueeze_(0)
        else:
            obs_img = obs_img.to(self.fixed_std.device).unsqueeze_(0)
        if not isinstance(goal_img, torch.Tensor):
            goal_img = torch.tensor(goal_img, device=self.fixed_std.device).unsqueeze_(0)
        else:
            goal_img = goal_img.to(self.fixed_std.device).unsqueeze_(0)

        dist = self.forward(None, obs_img, goal_img, None).pred_dist

        if argmax:
            actions = dist.mode
        else:
            actions = dist.sample()
        return actions
