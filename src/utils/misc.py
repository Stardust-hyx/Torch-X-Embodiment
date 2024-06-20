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

from inspect import signature
import bitsandbytes as bnb
import torch.nn as nn
from transformers.pytorch_utils import Conv1D
from accelerate import init_empty_weights


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

def clean_instruction(instruction: str):
    instruction = instruction.strip()
    instruction = instruction.lower()
    if instruction[-1] == '.':
        instruction = instruction[:-1]
    return instruction

def set_random_seed(seed, deterministic):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    get_accelerator().manual_seed_all(seed)
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        torch.use_deterministic_algorithms(True)


""" >>> Adapted from
    https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/utils/bitsandbytes.py
"""
def _replace_with_bnb_linear(
    model, modules_to_not_convert=None, current_key_name=None, quantization_config=None, has_been_replaced=False
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if (isinstance(module, nn.Linear) or isinstance(module, Conv1D)) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(".".join(current_key_name).endswith(key) for key in modules_to_not_convert):
                with init_empty_weights():
                    if isinstance(module, Conv1D):
                        in_features, out_features = module.weight.shape
                    else:
                        in_features = module.in_features
                        out_features = module.out_features

                    if quantization_config.quantization_method() == "llm_int8":
                        model._modules[name] = bnb.nn.Linear8bitLt(
                            in_features,
                            out_features,
                            module.bias is not None,
                            has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                            threshold=quantization_config.llm_int8_threshold,
                        )
                        has_been_replaced = True
                    else:
                        model._modules[name] = bnb.nn.Linear4bit(
                            in_features,
                            out_features,
                            module.bias is not None,
                            quantization_config.bnb_4bit_compute_dtype,
                            compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                            quant_type=quantization_config.bnb_4bit_quant_type,
                        )
                        has_been_replaced = True
                # Store the module class in case we need to transpose the weight later
                model._modules[name].source_cls = type(module)
                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_bnb_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_bnb_linear(model, modules_to_not_convert=None, current_key_name=None, quantization_config=None):
    """
    A helper function to replace all `torch.nn.Linear` modules by `bnb.nn.Linear8bit` modules from the `bitsandbytes`
    library. This will enable running your models using mixed int8 precision as described by the paper `LLM.int8():
    8-bit Matrix Multiplication for Transformers at Scale`. Make sure `bitsandbytes` compiled with the correct CUDA
    version of your hardware is installed before running this function. `pip install -i https://test.pypi.org/simple/
    bitsandbytes`

    The function will be run recursively and replace all `torch.nn.Linear` modules except for the `lm_head` that should
    be kept as a `torch.nn.Linear` module. The replacement is done under `init_empty_weights` context manager so no
    CPU/GPU memory is required to run this function. Int8 mixed-precision matrix decomposition works by separating a
    matrix multiplication into two streams: (1) and systematic feature outlier stream matrix multiplied in fp16
    (0.01%), (2) a regular stream of int8 matrix multiplication (99.9%). With this method, int8 inference with no
    predictive degradation is possible for very large models (>=176B parameters).

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`List[`str`]`, *optional*, defaults to `["lm_head"]`):
            Names of the modules to not convert in `Linear8bitLt`. In practice we keep the `lm_head` in full precision
            for numerical stability reasons.
        current_key_name (`List[`str`]`, *optional*):
            An array to track the current key of the recursion. This is used to check whether the current key (part of
            it) is not in the list of modules to not convert (for instances modules that are offloaded to `cpu` or
            `disk`).
    """
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    model, has_been_replaced = _replace_with_bnb_linear(
        model, modules_to_not_convert, current_key_name, quantization_config
    )

    # if not has_been_replaced:
    #     logger.warning(
    #         "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
    #         " Please double check your model architecture, or submit an issue on github if you think this is"
    #         " a bug."
    #     )

    return model
""" <<<
"""


class RepeatingLoader:

    def __init__(self, loader, init_seed):
        """Wraps an iterator to allow for infinite iteration. This is especially useful
        for DataLoader types that we wish to automatically restart upon completion.

        Args:
            loader (iterator): The data loader to repeat.
        """
        self.loader = loader
        self.seed = init_seed
        self.loader.dataset.shuffle(self.seed)
        self.data_iter = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.seed += 1
            self.loader.dataset.shuffle(self.seed)
            self.data_iter = iter(self.loader)
            batch = next(self.data_iter)
        return batch