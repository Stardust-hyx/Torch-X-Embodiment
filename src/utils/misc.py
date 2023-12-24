import os
import random
import numpy as np
import torch
try:
    from torchvision.transforms import v2
except:
    import torchvision.transforms as v2
from transformers import set_seed
from deepspeed.accelerator import get_accelerator


AUGMENT_OPS = {
    "random_resized_crop": v2.RandomResizedCrop,
    "color_jitter": v2.ColorJitter,
}


def get_augment_transform(augment_kwargs):
    transforms = []
    for op in augment_kwargs["augment_order"]:
        transform = AUGMENT_OPS[op](**augment_kwargs[op])
        transforms.append(transform)

    transforms = v2.Compose(transforms)
    # transforms = torch.nn.Sequential(*transforms)
    # transforms = torch.jit.script(transforms)
    return transforms

def set_random_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    get_accelerator().manual_seed_all(seed)
