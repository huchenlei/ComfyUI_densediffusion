from __future__ import annotations
import itertools
from typing import NamedTuple

import torch

import comfy
from comfy.model_patcher import ModelPatcher
from comfy.ldm.modules.attention import optimized_attention

from .enums import StableDiffusionVersion


ComfyUIConditioning = list  # Dummy type definitions for ComfyUI


def get_sd_version(model: ModelPatcher) -> StableDiffusionVersion:
    """Check if the model is a Stable Diffusion XL model.
    TODO: Make this a part of comfy.model_patcher.ModelPatcher
    """
    is_sdxl = isinstance(
        model.model,
        (
            comfy.model_base.SDXL,
            comfy.model_base.SDXLRefiner,
            comfy.model_base.SDXL_instructpix2pix,
        ),
    )
    return StableDiffusionVersion.SDXL if is_sdxl else StableDiffusionVersion.SD1x


class DenseDiffusionConditioning(NamedTuple):
    # List of text embeddings
    conds: list[torch.Tensor]
    # The mask to apply. Shape: [H, W]
    mask: torch.Tensor


class DenseDiffusionCrossAttention(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options):
        return optimized_attention(q, k, v, extra_options["n_heads"])


class DenseDiffusionApplyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "DenseDiffusion"
    DESCRIPTION = "Apply DenseDiffusion model."

    def apply(self, model: ModelPatcher) -> tuple[ModelPatcher]:
        work_model: ModelPatcher = model.clone()

        # TODO: Make patching cross-attn easier/more composable with Unified ModelPatcher.
        # Current approach does not compose with IPAdapter.
        sd_version: StableDiffusionVersion = get_sd_version(work_model)
        input_ids, output_ids, middle_ids = sd_version.transformer_ids
        for transformer_id in itertools.chain(input_ids, output_ids, middle_ids):
            work_model.set_model_attn2_replace(
                DenseDiffusionCrossAttention(),
                block_name=transformer_id.block_type.value,
                number=transformer_id.block_id,
                # transformer_index param here specifies the depth index of the transformer
                transformer_index=transformer_id.block_index,
            )
        return (work_model,)


class DenseDiffusionAddCondNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "mask": ("MASK",),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "append"
    CATEGORY = "DenseDiffusion"
    DESCRIPTION = "Set a regional prompt for DenseDiffusion."

    def append(
        self,
        model: ModelPatcher,
        conditioning: ComfyUIConditioning,
        mask: torch.Tensor,
        strength: float,
    ) -> tuple[ComfyUIConditioning]:
        work_model: ModelPatcher = model.clone()
        work_model.model_options["transformer_options"]["dense_diffusion_cond"] = (
            DenseDiffusionConditioning(
                conds=[c for c, _ in conditioning], mask=mask * strength
            )
        )
        return (work_model,)


NODE_CLASS_MAPPINGS = {
    "DenseDiffusionApplyNode": DenseDiffusionApplyNode,
    "DenseDiffusionAddCondNode": DenseDiffusionAddCondNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DenseDiffusionApplyNode": "DenseDiffusion Apply",
    "DenseDiffusionAddCondNode": "DenseDiffusion Add Cond",
}
