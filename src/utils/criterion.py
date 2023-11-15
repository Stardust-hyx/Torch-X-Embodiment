import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

# class ActionCriterion(nn.Module):

#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)


def action_criterion(output_dists: MultivariateNormal, gold_actions: torch.Tensor):
    pi_actions = output_dists.mode
    log_probs = output_dists.log_prob(gold_actions)
    mse = ((pi_actions - gold_actions) ** 2).sum(-1)
    actor_loss = -(log_probs).mean()
    actor_loss_item = actor_loss.item()
    
    return actor_loss, {
        "actor_loss": actor_loss_item,
        "mse": mse.mean().item(),
        "log_probs": -actor_loss_item,
        "pi_actions": pi_actions.mean().item(),
    }
