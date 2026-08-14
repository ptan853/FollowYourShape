import math
import json
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from .model import Flux
from .modules.conditioner import HFEmbedder
from .math import apply_rope
from .attention_mask_utils import build_attention_gated_tdm_mask, normalize01, otsu_threshold, smooth_map

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import os
import matplotlib.pyplot as plt
import seaborn as sns


class SingleBlockAttentionProbe:
    def __init__(self, model: Flux, token_indices: list[int], txt_len: int, layer_ids: list[int]) -> None:
        self.token_indices = token_indices
        self.txt_len = txt_len
        self.layer_ids = set(layer_ids)
        self.records: list[torch.Tensor] = []
        self.handles = []

        for layer_id, block in enumerate(model.single_blocks):
            if layer_id in self.layer_ids:
                self.handles.append(block.register_forward_pre_hook(self._make_hook(), with_kwargs=True))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _make_hook(self):
        def hook(module, inputs, kwargs) -> None:
            info = kwargs.get("info")
            if not self.token_indices:
                return None
            if info is None or not info.get("record_attention", False):
                return None

            x = inputs[0] if inputs else kwargs["x"]
            vec = kwargs["vec"]
            pe = kwargs["pe"]
            with torch.no_grad():
                mod, _ = module.modulation(vec)
                x_mod = (1 + mod.scale) * module.pre_norm(x) + mod.shift
                qkv, _ = torch.split(module.linear1(x_mod), [3 * module.hidden_size, module.mlp_hidden_dim], dim=-1)
                q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=module.num_heads)
                q, k = module.norm(q, k, v)
                q, k = apply_rope(q, k, pe)

                img_q = q[:, :, self.txt_len :, :]
                scale = img_q.shape[-1] ** -0.5
                logits = torch.einsum("bhid,bhjd->bhij", img_q.float(), k.float()) * scale
                attn = torch.softmax(logits, dim=-1)
                scores = attn[:, :, :, self.token_indices].sum(dim=-1).mean(dim=1)[0].detach().cpu()
                self.records.append(scores)
            return None

        return hook


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def build_inject_list(num_inference_steps: int, inject_step: int, tail_pad: int = 0, front_pad: int = 0):
    total = num_inference_steps - 1
    available_middle = total - front_pad - tail_pad

    if inject_step > available_middle:
        raise ValueError(f"inject_step {inject_step} is too large. Only {available_middle} steps available between front_pad and tail_pad.")

    middle_false = available_middle - inject_step
    middle_list = [False] * middle_false + [True] * inject_step

    inject_list = [True] * front_pad + middle_list + [False] * tail_pad
    return inject_list




def get_controlnet_output(
    controlnet,
    control_patch,
    img,
    vec,
    txt,
    txt_ids,
    img_ids,
    t,
    guidance,
    controlnet_mode,
    controlnet_scale,
    guidance_start,
    guidance_end,
    step_idx,
    total_steps,
):
    if controlnet is None or control_patch is None:
        return None, None

    progress = step_idx / (total_steps - 1)
    if not (guidance_start <= progress <= guidance_end):
        return None, None

    t_tensor = torch.tensor([t], dtype=img.dtype, device=img.device)
    guidance_tensor = torch.tensor([guidance], dtype=img.dtype, device=img.device)

    return controlnet(
        hidden_states=img,
        controlnet_cond=control_patch,
        controlnet_mode=controlnet_mode,
        conditioning_scale=controlnet_scale,
        timestep=t_tensor,
        guidance=guidance_tensor,
        pooled_projections=vec,
        encoder_hidden_states=txt,
        txt_ids=txt_ids[0],
        img_ids=img_ids[0],
        joint_attention_kwargs=None,
        return_dict=False,
    )



def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info: dict = None,
    inject_list: list[bool] = None, 
    guidance: float = 4.0,
    controlnet=None,                  
    control_patch=None,             
    controlnet_scale: Union[float, list[float]] = 1.0,
    controlnet_mode: Union[int, list[int]] = 0,
    guidance_start: float = 0.0, 
    guidance_end: float = 1.0
):

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]

    print(inject_list)

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    if info is not None:
        info['inv_noise'] = {}
        info['map'] = {}
        info['edit_map'] = None

    desc = "Inversion" if inverse else "Denoising"

    for i, (t_curr, t_prev) in tqdm(enumerate(zip(timesteps[:-1], timesteps[1:])), desc=desc, total=len(timesteps) - 1):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        controlnet_block_samples, controlnet_single_block_samples = get_controlnet_output(
            controlnet=controlnet,
            control_patch=control_patch,
            img=img,
            vec=vec,
            txt=txt,
            txt_ids=txt_ids,
            img_ids=img_ids,
            t=t_curr,
            guidance=guidance,
            controlnet_mode=controlnet_mode,
            controlnet_scale=controlnet_scale,
            guidance_start=guidance_start,
            guidance_end=guidance_end,
            step_idx=i,
            total_steps=len(timesteps)
        )


        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples
        )


        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        info['second_order'] = True


        step_idx = i + 0.5
        t_val = t_vec_mid[0].item()
        
        controlnet_block_samples_mid, controlnet_single_block_samples_mid = get_controlnet_output(
            controlnet=controlnet,
            control_patch=control_patch,
            img=img_mid,
            vec=vec,
            txt=txt,
            txt_ids=txt_ids,
            img_ids=img_ids,
            t=t_val,
            guidance=guidance,
            controlnet_mode=controlnet_mode,
            controlnet_scale=controlnet_scale,
            guidance_start=guidance_start,
            guidance_end=guidance_end,
            step_idx=step_idx,
            total_steps=len(timesteps)
        )

        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info,
            controlnet_block_samples=controlnet_block_samples_mid,
            controlnet_single_block_samples=controlnet_single_block_samples_mid
        )

        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order

        if inverse:
            step =  f'step{ len(timesteps) - i - 2}'
            info['inv_noise'][step] = (pred + pred_mid) / 2

    return img, info


def denoise_with_TDM(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float],
    inverse,
    width,
    height,
    guidance: float = 4.0,
    info: dict=None,
    inject_list: list[bool] = None,
    tail_pad: int = 1,
    front_pad: int = 3,
    controlnet=None,                  
    control_patch=None,             
    controlnet_scale: Union[float, list[float]] = 1.0,
    controlnet_mode: Union[int, list[int]] = 0,
    guidance_start: float = 0.0, 
    guidance_end: float = 1.0,
    tdm_mask_mode: str = "original",
    attention_token_indices: list[int] | None = None,
    attention_layer_ids: list[int] | None = None,
    attention_token_mode: str = "part_edit",
    attention_part: str | None = None,
    attention_edit: str | None = None,
):

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]

    print(inject_list)

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    desc = "Inversion" if inverse else "Denoising"

    cut = len(inject_list) - info['inject_step'] - 2 - tail_pad

    print(f"Cutting at {cut} step")

    if info is not None:
        info['map'] = {}
        info['edit_map'] = None

    attention_probe = None
    attention_layer_ids = attention_layer_ids or list(range(28, 38))
    if tdm_mask_mode == "attention_gated":
        if not attention_token_indices:
            raise ValueError("attention_token_indices is required when tdm_mask_mode='attention_gated'")
        attention_probe = SingleBlockAttentionProbe(
            model,
            token_indices=attention_token_indices,
            txt_len=txt.shape[1],
            layer_ids=attention_layer_ids,
        )
    elif tdm_mask_mode != "original":
        raise ValueError(f"Unsupported tdm_mask_mode: {tdm_mask_mode}")

    for i, (t_curr, t_prev) in tqdm(enumerate(zip(timesteps[:-1], timesteps[1:])), desc=desc, total=len(timesteps) - 1):
        # if i == 10:
        #     break
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)


        step =  f'step{ i}'
        pred_src = info['inv_noise'][step]

        should_record_attention = attention_probe is not None and i >= front_pad and i <= cut
        attention_info = None
        if should_record_attention:
            attention_info = {
                "feature": {},
                "map": {},
                "edit_map": None,
                "inject": False,
                "inverse": False,
                "second_order": False,
                "record_attention": True,
                "t": t_curr,
            }

        pred_tar, _ = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=attention_info
        )
        img_mid_test = img + (t_prev - t_curr) / 2 * pred_tar
        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        if attention_info is not None:
            attention_info["second_order"] = True
            attention_info["t"] = float(t_vec_mid[0].item())
        pred_mid_test, _ = model(
            img=img_mid_test,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=attention_info
        )
        first_order = (pred_mid_test - pred_tar) / ((t_prev - t_curr) / 2)
        pred_tar = (pred_mid_test + pred_tar) / 2


        delta = (pred_src - pred_tar).pow(2).sum(dim=-1).sqrt()

        
        delta_min = delta.min()
        delta_max = delta.max()
        delta_norm = (delta - delta_min) / (delta_max - delta_min)
        H_patch = math.ceil(height / 16)
        W_patch = math.ceil(width / 16)
        delta_map = delta_norm[0].reshape(W_patch, H_patch)

        if info is not None and i >= front_pad and i <= cut:
            info['map'][f"{i}_delta_map"] = delta_map


        vis_dir = info.get("vis_path", None)
        if vis_dir:
            delta_dir = os.path.join(vis_dir, "delta")
            os.makedirs(delta_dir, exist_ok=True)
            delta_np = delta_map.to(torch.float32).cpu().numpy()
            np.save(os.path.join(delta_dir, f"delta_map_{i}.npy"), delta_np)
            plt.imsave(os.path.join(delta_dir, f"delta_map_{i}.png"), delta_np, cmap="viridis")



        if i == cut:
            recorded_delta_items = sorted(
                ((int(k.split("_", 1)[0]), v) for k, v in info['map'].items() if k.endswith("_delta_map")),
                key=lambda item: item[0],
            )
            recorded_delta_steps = [step for step, _ in recorded_delta_items]
            delta_stack = torch.stack([v for _, v in recorded_delta_items], dim=0)  # [N, H_patch, W_patch]
            
            scale = 5
            softmax_weights = F.softmax(delta_stack * scale, dim=0)  # [N, H, W]
            soft_mask = (delta_stack * softmax_weights).sum(dim=0)  # [H, W]
            soft_np = soft_mask.to(torch.float32).cpu().numpy()  # [H_patch, W_patch]
            smoothing_sigma = 0.7
            smoothed_np = gaussian_filter(soft_np, sigma=smoothing_sigma)

            from skimage.filters import threshold_otsu

            threshold = threshold_otsu(smoothed_np)
            # print("Otsu threshold:", threshold)
            # values = smoothed_np.flatten()
            # plt.figure(figsize=(6,4))
            # sns.histplot(values, bins=50, kde=True, color="lightblue", stat="count", alpha=0.6)
            # plt.axvline(threshold, color="red", linestyle="--", linewidth=1, label=f"Otsu τ = {threshold:.2f}")
            # plt.xlim(0, 1)
            # plt.xlabel("Value")
            # plt.ylabel("Density")
            # plt.legend()
            # plt.title("Smoothed values distribution")
            # plt.savefig("smoothed_dist.png", dpi=300, bbox_inches="tight")
            # import pdb; pdb.set_trace()


            smoothed_binary_np = (smoothed_np > threshold).astype(np.uint8)
            original_binary_np = smoothed_binary_np

            attention_np = None
            mask_result = build_attention_gated_tdm_mask(
                smoothed_tdm=smoothed_np,
                original_binary_tdm=original_binary_np,
                attention_map=None,
                mask_mode="original",
                smoothing_sigma=smoothing_sigma,
            )
            if attention_probe is not None:
                if not attention_probe.records:
                    raise RuntimeError("No target-token attention records were captured for attention-gated TDM.")
                attention_flat = torch.stack(attention_probe.records, dim=0).mean(dim=0).numpy()
                attention_np = normalize01(attention_flat.reshape(W_patch, H_patch))
                mask_result = build_attention_gated_tdm_mask(
                    smoothed_tdm=smoothed_np,
                    original_binary_tdm=original_binary_np,
                    attention_map=attention_np,
                    mask_mode=tdm_mask_mode,
                    smoothing_sigma=smoothing_sigma,
                )

            selected_binary_np = mask_result["binary_mask"].astype(np.uint8)
            binary_map = torch.tensor(selected_binary_np, device=delta_stack.device, dtype=torch.float32)


            # flatten and extract foreground patch indices
            edit_map_flat = binary_map.flatten()  # [N_patch]
            edit_indices = (edit_map_flat > 0).nonzero(as_tuple=False).squeeze(1)  # [N_foreground]
            info["edit_map"] = edit_indices  


            if vis_dir:
                os.makedirs(vis_dir, exist_ok=True)
                delta_stack_np = delta_stack.to(torch.float32).cpu().numpy()
                binary_np = binary_map.to(torch.float32).cpu().numpy()
                np.save(os.path.join(vis_dir, "delta_stack.npy"), delta_stack_np)
                np.save(os.path.join(vis_dir, "aggregated_soft_tdm.npy"), soft_np)
                np.save(os.path.join(vis_dir, "smoothed_soft_tdm.npy"), smoothed_np)
                np.save(os.path.join(vis_dir, "binary_tdm_mask.npy"), original_binary_np)
                np.save(os.path.join(vis_dir, "selected_binary_tdm_mask.npy"), binary_np)
                plt.imsave(os.path.join(vis_dir, "aggregated_soft_tdm.png"), soft_np, cmap="viridis")
                plt.imsave(os.path.join(vis_dir, "smoothed_soft_tdm.png"), smoothed_np, cmap="viridis")
                plt.imsave(os.path.join(vis_dir, "binary_tdm_mask.png"), original_binary_np, cmap="gray")
                plt.imsave(os.path.join(vis_dir, "selected_binary_tdm_mask.png"), binary_np, cmap="gray")
                if attention_np is not None:
                    np.save(os.path.join(vis_dir, "attention_gate_raw.npy"), attention_np.astype(np.float32))
                    attention_smoothed = smooth_map(attention_np, sigma=smoothing_sigma)
                    attention_threshold = otsu_threshold(attention_smoothed)
                    attention_binary = (attention_smoothed > attention_threshold).astype(np.uint8)
                    np.save(os.path.join(vis_dir, "attention_gate_smoothed.npy"), attention_smoothed)
                    np.save(os.path.join(vis_dir, "attention_gate_binary.npy"), attention_binary)
                    plt.imsave(os.path.join(vis_dir, "attention_gate_raw.png"), attention_np, cmap="viridis")
                    plt.imsave(os.path.join(vis_dir, "attention_gate_binary.png"), attention_binary, cmap="gray")
                if mask_result["hybrid_soft"] is not None:
                    np.save(os.path.join(vis_dir, "hybrid_soft_tdm_attention.npy"), mask_result["hybrid_soft"])
                    np.save(os.path.join(vis_dir, "hybrid_smoothed_tdm_attention.npy"), mask_result["hybrid_smoothed"])
                    np.save(os.path.join(vis_dir, "hybrid_binary_tdm_attention.npy"), binary_np)
                    plt.imsave(os.path.join(vis_dir, "hybrid_soft_tdm_attention.png"), mask_result["hybrid_soft"], cmap="viridis")
                    plt.imsave(os.path.join(vis_dir, "hybrid_binary_tdm_attention.png"), binary_np, cmap="gray")
                tdm_metadata = {
                    "cut_step": int(cut),
                    "front_pad": int(front_pad),
                    "tail_pad": int(tail_pad),
                    "inject_step": int(info["inject_step"]),
                    "tdm_mask_mode": tdm_mask_mode,
                    "selected_mask_source": mask_result["selected_mask_source"],
                    "recorded_delta_steps": recorded_delta_steps,
                    "num_recorded_delta_maps": int(delta_stack_np.shape[0]),
                    "aggregation": "softmax_weighted_sum",
                    "softmax_scale": scale,
                    "smoothing": "gaussian_filter",
                    "smoothing_sigma": smoothing_sigma,
                    "threshold_method": "otsu",
                    "threshold": float(threshold),
                    "selected_threshold": float(mask_result["threshold"]) if mask_result["threshold"] is not None else float(threshold),
                    "tdm_shape": list(soft_np.shape),
                    "binary_mask_area_ratio_patch_grid": float(binary_np.mean()),
                    "num_edit_tokens": int(edit_indices.numel()),
                    "attention_token_mode": attention_token_mode,
                    "attention_part": attention_part,
                    "attention_edit": attention_edit,
                    "attention_token_indices": attention_token_indices or [],
                    "attention_layer_ids": attention_layer_ids,
                    "num_attention_records": int(len(attention_probe.records)) if attention_probe is not None else 0,
                }
                with open(os.path.join(vis_dir, "tdm_metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(tdm_metadata, f, indent=2)

                plt.figure()
                plt.imshow(binary_np, cmap='viridis')
                plt.colorbar()
                plt.title("Edit Map")
                plt.savefig(os.path.join(vis_dir, "edit_map.png"))
                plt.close()
                print("Saved edit map visualization to edit_map.png")





        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]


        controlnet_block_samples, controlnet_single_block_samples = get_controlnet_output(
            controlnet=controlnet,
            control_patch=control_patch,
            img=img,
            vec=vec,
            txt=txt,
            txt_ids=txt_ids,
            img_ids=img_ids,
            t=t_curr,
            guidance=guidance,
            controlnet_mode=controlnet_mode,
            controlnet_scale=controlnet_scale,
            guidance_start=guidance_start,
            guidance_end=guidance_end,
            step_idx=i,
            total_steps=len(timesteps)
        )


        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples
        )

        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        info['second_order'] = True

        step_idx = i + 0.5
        t_val = t_vec_mid[0].item()
        controlnet_block_samples_mid, controlnet_single_block_samples_mid = get_controlnet_output(
            controlnet=controlnet,
            control_patch=control_patch,
            img=img_mid,
            vec=vec,
            txt=txt,
            txt_ids=txt_ids,
            img_ids=img_ids,
            t=t_val,
            guidance=guidance,
            controlnet_mode=controlnet_mode,
            controlnet_scale=controlnet_scale,
            guidance_start=guidance_start,
            guidance_end=guidance_end,
            step_idx=step_idx,
            total_steps=len(timesteps)
        )

        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info,
            controlnet_block_samples=controlnet_block_samples_mid,
            controlnet_single_block_samples=controlnet_single_block_samples_mid
        )

        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order

    if attention_probe is not None:
        attention_probe.close()

    return img, info


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
