from __future__ import annotations
from typing import NamedTuple

import torch

from comfy.model_patcher import ModelPatcher


ComfyUIConditioning = list  # Dummy type definitions for ComfyUI


class DenseDiffusionConditioning(NamedTuple):
    # List of text embeddings
    conds: list[torch.Tensor]
    # The mask to apply. Shape: [H, W]
    mask: torch.Tensor


class DenseDiffusionApplyNode:
    @staticmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "DenseDiffusion"
    DESCRIPTION = "Apply DenseDiffusion model."

    def apply(self, model: ModelPatcher) -> tuple[ModelPatcher]:
        work_model: ModelPatcher = model.clone()
        # CrossAttn
        work_model.set_model_attn2_replace()
        return work_model


class ConditioningDenseDiffusionSetMask:
    @staticmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "mask": ("MASK",),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"
    CATEGORY = "DenseDiffusion"
    DESCRIPTION = "Set mask for DenseDiffusion."

    def append(
        self,
        conditioning: ComfyUIConditioning,
        mask: torch.Tensor,
        strength: float,
    ) -> tuple[ComfyUIConditioning]:
        return (conditioning,)


NODE_CLASS_MAPPINGS = {
    "DenseDiffusionApplyNode": DenseDiffusionApplyNode,
    "ConditioningDenseDiffusionSetMask": ConditioningDenseDiffusionSetMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DenseDiffusionApplyNode": "DenseDiffusion Apply",
    "ConditioningDenseDiffusionSetMask": "Set Mask for DenseDiffusion",
}
