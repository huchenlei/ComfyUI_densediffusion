from __future__ import annotations
import itertools
import math
from typing import NamedTuple

import torch

import comfy
from comfy.model_patcher import ModelPatcher

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
    # Text embeddings
    cond: torch.Tensor
    # The mask to apply. Shape: [H, W]
    mask: torch.Tensor
    pooled_output: torch.Tensor | None = None


class OmostDenseDiffusionCrossAttention(torch.nn.Module):
    @staticmethod
    def scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_bool: torch.Tensor | None = None,
        mask_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_weight = query @ key.transpose(-2, -1) * scale_factor

        if mask_scale is not None:
            attn_weight = attn_weight * mask_scale.to(attn_weight)
        if mask_bool is not None:
            attn_weight.masked_fill_(mask_bool.logical_not(), float("-inf"))

        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options: dict
    ):
        """
        y=softmax(modify(q@k))@v
        where modify() is a complicated non-linear function with many normalizations
        and tricks to change the score's distributions.

        This function implements the `modify` function used in Omost, instead
        of the original DenseDiffusion repo.

        https://github.com/lllyasviel/Omost/blob/731e74922fc6be91171688574d07624f93d3b658/lib_omost/pipeline.py#L129-L173
        """
        heads: int = extra_options["n_heads"]

        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

        dd_conds: list[DenseDiffusionConditioning] = extra_options.get("dd_conds", [])
        if dd_conds:
            B, C, H, W = extra_options["original_shape"]
            masks = []
            for dd_cond in dd_conds:
                m = (
                    torch.nn.functional.interpolate(
                        dd_cond.mask[None, None, :, :], (H, W), mode="nearest-exact"
                    )
                    .flatten()
                    .unsqueeze(1)
                    .repeat(1, dd_cond.cond.size(1))
                )
                masks.append(m)
            masks = torch.cat(masks, dim=1)

            mask_bool = masks > 0.5
            mask_scale = (H * W) / torch.sum(masks, dim=0, keepdim=True)
            mask_bool = mask_bool[None, None, :, :].repeat(q.size(0), q.size(1), 1, 1)
            mask_scale = mask_scale[None, None, :, :].repeat(q.size(0), q.size(1), 1, 1)
        else:
            mask_bool = None
            mask_scale = None

        out = self.scaled_dot_product_attention(
            q, k, v, mask_bool=mask_bool, mask_scale=mask_scale
        )
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class DenseDiffusionApplyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", "CONDITIONING")
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
                OmostDenseDiffusionCrossAttention(),
                block_name=transformer_id.block_type.value,
                number=transformer_id.block_id,
                # transformer_index param here specifies the depth index of the transformer
                transformer_index=transformer_id.block_index,
            )

        dd_conds: list[DenseDiffusionConditioning] = work_model.model_options[
            "transformer_options"
        ].get("dense_diffusion_cond", [])
        assert dd_conds, "No DenseDiffusion conditioning found!"
        cond = [
            [
                # cond
                torch.cat([dd_cond.cond for dd_cond in dd_conds], dim=1),
                # pooled_output
                {"pooled_output": dd_conds[0].pooled_output},
            ]
        ]
        return (work_model, cond)


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
        work_model.model_options["transformer_options"].setdefault(
            "dense_diffusion_cond", []
        )
        assert len(conditioning) == 1
        cond, extra_fields = conditioning[0]
        assert isinstance(extra_fields, dict)
        assert "pooled_output" in extra_fields

        work_model.model_options["transformer_options"]["dense_diffusion_cond"].append(
            DenseDiffusionConditioning(
                cond=cond,
                mask=mask * strength,
                pooled_output=extra_fields["pooled_output"],
            )
        )
        return (work_model, conditioning)


NODE_CLASS_MAPPINGS = {
    "DenseDiffusionApplyNode": DenseDiffusionApplyNode,
    "DenseDiffusionAddCondNode": DenseDiffusionAddCondNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DenseDiffusionApplyNode": "DenseDiffusion Apply",
    "DenseDiffusionAddCondNode": "DenseDiffusion Add Cond",
}
