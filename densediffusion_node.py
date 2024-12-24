from __future__ import annotations
import itertools
import math
from dataclasses import dataclass
from typing import Literal

import torch
from comfy.model_patcher import ModelPatcher
from .enums import StableDiffusionVersion

ComfyUIConditioning = list

def get_sd_version(model: ModelPatcher) -> StableDiffusionVersion:
    """Determine the Stable Diffusion version."""
    return StableDiffusionVersion.SDXL if isinstance(model.model, (
        comfy.model_base.SDXL,
        comfy.model_base.SDXLRefiner,
        comfy.model_base.SDXL_instructpix2pix,
    )) else StableDiffusionVersion.SD1x

@dataclass
class DenseDiffusionConditioning:
    cond: torch.Tensor
    mask: torch.Tensor
    pooled_output: torch.Tensor | None = None

class OmostDenseDiffusionCrossAttention(torch.nn.Module):
    @staticmethod
    def calc_mask_cond(q: torch.Tensor, extra_options: dict) -> tuple[torch.Tensor, torch.Tensor]:
        dd_conds = extra_options.get("dense_diffusion_cond", [])
        cond_or_uncond = extra_options.get("cond_or_uncond", [])
        batch_size = q.size(0)
        cond_batch_size = batch_size // len(cond_or_uncond) if 0 in cond_or_uncond else 0
        uncond_batch_size = batch_size - cond_batch_size

        if not dd_conds or cond_batch_size == 0:
            return None, None

        latent_height, latent_width = extra_options["original_shape"][-2:]
        H, W = OmostDenseDiffusionCrossAttention.calc_hidden_state_shape(q.size(2), latent_height, latent_width)

        masks = torch.cat([
            torch.nn.functional.interpolate(dd.mask[None, None], (H, W), mode="nearest-exact")
            .flatten().unsqueeze(1).repeat(1, dd.cond.size(1))
            for dd in dd_conds
        ], dim=1)

        mask_bool = masks > 0.5
        mask_scale = (H * W) / masks.sum(dim=0, keepdim=True)
        mask_bool = mask_bool.repeat(cond_batch_size, q.size(1), 1, 1)
        mask_scale = mask_scale.repeat(cond_batch_size, q.size(1), 1, 1)

        if uncond_batch_size > 0:
            uncond_mask = torch.ones_like(mask_bool)
            mask_bool = torch.cat([uncond_mask, mask_bool], dim=0) if cond_or_uncond[0] == 1 else torch.cat([mask_bool, uncond_mask], dim=0)
            mask_scale = torch.cat([uncond_mask, mask_scale], dim=0) if cond_or_uncond[0] == 1 else torch.cat([mask_scale, uncond_mask], dim=0)

        return mask_bool, mask_scale

    @staticmethod
    def calc_hidden_state_shape(sequence_length: int, H: int, W: int) -> tuple[int, int]:
        factor = math.sqrt(sequence_length / (H * W))
        return int(H * factor), int(W * factor)

    @staticmethod
    def scaled_dot_product_attention(query, key, value, mask_bool=None, mask_scale=None):
        attn_weight = (query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask_scale is not None:
            attn_weight *= mask_scale
        if mask_bool is not None:
            attn_weight.masked_fill_(~mask_bool, float("-inf"))
        return torch.softmax(attn_weight, dim=-1) @ value

    def forward(self, q, k, v, extra_options):
        heads = extra_options["n_heads"]
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = [t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v)]
        mask_bool, mask_scale = self.calc_mask_cond(q, extra_options)
        out = self.scaled_dot_product_attention(q, k, v, mask_bool, mask_scale)
        return out.transpose(1, 2).reshape(b, -1, heads * dim_head)

class DenseDiffusionApplyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    CATEGORY = "DenseDiffusion"

    def apply(self, model: ModelPatcher):
        work_model = model.clone()
        sd_version = get_sd_version(work_model)
        for transformer_id in itertools.chain(*sd_version.transformer_ids):
            work_model.set_model_attn2_replace(
                OmostDenseDiffusionCrossAttention(),
                block_name=transformer_id.block_type.value,
                number=transformer_id.block_id,
                transformer_index=transformer_id.block_index,
            )
        dd_conds = work_model.model_options["transformer_options"].get("dense_diffusion_cond", [])
        assert dd_conds, "No DenseDiffusion conditioning found!"
        for dd_cond in dd_conds:
            dd_cond.mask = dd_cond.mask.to(model.load_device, model.model_dtype())
        cond = [[torch.cat([dd.cond for dd in dd_conds], dim=1), {"pooled_output": dd_conds[0].pooled_output}]]
        return work_model, cond

class DenseDiffusionAddCondNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "mask": ("MASK",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "DenseDiffusion"

    def append(self, model, conditioning, mask, strength):
        work_model = model.clone()
        work_model.model_options["transformer_options"].setdefault("dense_diffusion_cond", []).append(
            DenseDiffusionConditioning(
                cond=conditioning[0][0],
                mask=mask.squeeze() * strength,
                pooled_output=conditioning[0][1]["pooled_output"],
            )
        )
        return work_model,

NODE_CLASS_MAPPINGS = {
    "DenseDiffusionApplyNode": DenseDiffusionApplyNode,
    "DenseDiffusionAddCondNode": DenseDiffusionAddCondNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DenseDiffusionApplyNode": "DenseDiffusion Apply",
    "DenseDiffusionAddCondNode": "DenseDiffusion Add Cond",
}
