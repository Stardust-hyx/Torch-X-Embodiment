from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from utils.misc import get_augment_transform

@dataclass
class GCBCAgentOutput:
    pred_dist: MultivariateNormal

class GCBCAgent(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        args,
        encode_dim: int = 512,
        hidden_dim: int =256,
        dropout_rate: float = 0.1,
        action_dim: int = 7,
    ) -> None:
        super().__init__()

        self.encoder = encoder

        self.mlp = nn.Sequential(
            nn.Linear(encode_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.SiLU(inplace=True)
        )

        self.action_dim = action_dim
        self.action_mean_linear = nn.Linear(hidden_dim, action_dim)

        self.augment = args.augment
        self.augment_kwargs = args.augment_kwargs
        if self.augment and self.augment_kwargs:
            self.augment_transform = get_augment_transform(self.augment_kwargs)

        self.register_buffer("fixed_std", torch.eye(action_dim))

    def forward(self, prompts, obs_imgs, goal_imgs, feature_imgs) -> GCBCAgentOutput:
        obs_imgs = obs_imgs.to(self.fixed_std.device)
        goal_imgs = goal_imgs.to(self.fixed_std.device)

        if self.training and self.augment:
            obs_imgs, goal_imgs = self._augment((obs_imgs, goal_imgs))
        
        observation_and_goal = torch.concat((obs_imgs, goal_imgs), dim=-3)

        outputs = self.mlp(self.encoder(observation_and_goal))

        means = self.action_mean_linear(outputs)

        dist = MultivariateNormal(means, scale_tril=self.fixed_std)
        # return GCBCAgentOutput(
        #     pred_dist=dist
        # )
        return (dist,)
    
    def _augment(self, image_tuple):
        images = torch.cat(image_tuple)
        images = self.augment_transform(images)
        return torch.chunk(images, chunks=len(image_tuple))
    
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

        dist = self.forward(None, obs_img, goal_img, None)[0]

        if argmax:
            actions = dist.mode
        else:
            actions = dist.sample()
        return actions
    
    def renew_action_linear(self, action_dim):
        if action_dim != self.action_dim:
            hidden_dim = self.action_mean_linear.weight.data.shape[1]
            self.action_mean_linear = nn.Linear(hidden_dim, action_dim)
            self.register_buffer("fixed_std", torch.eye(action_dim))

    def set_feature_layers_require_grad(self, flag: bool):
        self.encoder.requires_grad_(flag)

    def set_action_fc_require_grad(self, flag: bool):
        self.mlp.requires_grad_(flag)

    def set_action_linear_require_grad(self, flag: bool):
        self.action_mean_linear.requires_grad_(flag)