from __future__ import annotations

from typing import Any

import numpy as np


def normalize01(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    lo = float(values[finite].min())
    hi = float(values[finite].max())
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return (values - lo) / (hi - lo)


def otsu_threshold(values: np.ndarray) -> float:
    values = values.astype(np.float32)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0
    if np.unique(finite_values).size <= 1:
        return float(finite_values.mean())

    hist, bin_edges = np.histogram(finite_values, bins=256)
    hist = hist.astype(np.float64)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    weight_bg = np.cumsum(hist)
    weight_fg = hist.sum() - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not valid.any():
        return float(finite_values.mean())

    mean_bg = np.cumsum(hist * centers) / np.maximum(weight_bg, 1e-12)
    mean_fg = (np.cumsum((hist * centers)[::-1]) / np.maximum(np.cumsum(hist[::-1]), 1e-12))[::-1]
    between_class_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between_class_var[~valid] = -1
    return float(centers[int(np.argmax(between_class_var))])


def smooth_map(values: np.ndarray, sigma: float) -> np.ndarray:
    values = values.astype(np.float32)
    if sigma <= 0:
        return values
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(values, sigma=sigma).astype(np.float32)
    except ImportError:
        return values


def build_attention_gated_tdm_mask(
    *,
    smoothed_tdm: np.ndarray,
    original_binary_tdm: np.ndarray,
    attention_map: np.ndarray | None,
    mask_mode: str,
    smoothing_sigma: float = 0.7,
) -> dict[str, Any]:
    if mask_mode == "original":
        return {
            "selected_mask_source": "original_tdm",
            "hybrid_soft": None,
            "hybrid_smoothed": None,
            "threshold": None,
            "binary_mask": original_binary_tdm.astype(np.uint8),
        }

    if mask_mode != "attention_gated":
        raise ValueError(f"Unsupported TDM mask mode: {mask_mode}")
    if attention_map is None:
        raise ValueError("attention_map is required when mask_mode='attention_gated'")
    if attention_map.shape != smoothed_tdm.shape:
        raise ValueError(
            f"attention_map shape {attention_map.shape} does not match TDM shape {smoothed_tdm.shape}"
        )

    hybrid_soft = normalize01(smoothed_tdm) * normalize01(attention_map)
    hybrid_smoothed = smooth_map(hybrid_soft, sigma=smoothing_sigma)
    threshold = otsu_threshold(hybrid_smoothed)
    binary_mask = (hybrid_smoothed > threshold).astype(np.uint8)

    return {
        "selected_mask_source": "attention_gated_tdm",
        "hybrid_soft": hybrid_soft.astype(np.float32),
        "hybrid_smoothed": hybrid_smoothed.astype(np.float32),
        "threshold": threshold,
        "binary_mask": binary_mask,
    }


def wordpiece_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text.replace("_", " "),
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors=None,
    )
    pad_id = tokenizer.pad_token_id
    return [int(x) for x in encoded["input_ids"] if int(x) != pad_id]


def find_subsequence_positions(sequence: list[int], subsequence: list[int]) -> list[int]:
    if not subsequence:
        return []
    out: list[int] = []
    n = len(subsequence)
    for i in range(0, len(sequence) - n + 1):
        if sequence[i : i + n] == subsequence:
            out.extend(range(i, i + n))
    return out


def select_target_token_indices(
    tokenizer: Any,
    target_prompt: str,
    part: str,
    edit: str,
    max_length: int,
    token_mode: str,
) -> list[int]:
    encoding = tokenizer(
        [target_prompt],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    pad_id = tokenizer.pad_token_id
    prompt_ids = [int(x) for x in encoding["input_ids"][0].tolist()]
    nonpad = [idx for idx, token_id in enumerate(prompt_ids) if token_id != pad_id]

    phrase_by_mode = {
        "part": [part],
        "edit": [edit],
        "part_edit": [part, edit, f"{edit} {part}"],
    }
    if token_mode not in phrase_by_mode:
        raise ValueError(f"Unknown token_mode: {token_mode}")

    selected: list[int] = []
    for phrase in phrase_by_mode[token_mode]:
        selected.extend(find_subsequence_positions(prompt_ids, wordpiece_ids(tokenizer, phrase)))

    if not selected:
        tokens = tokenizer.convert_ids_to_tokens(prompt_ids)
        needles = []
        if token_mode in {"part", "part_edit"}:
            needles.append(part.lower().replace("_", ""))
        if token_mode in {"edit", "part_edit"}:
            needles.append(edit.lower().replace("_", ""))
        for idx in nonpad:
            token = str(tokens[idx]).lower().replace("▁", "").replace("_", "")
            if any(needle and needle in token for needle in needles):
                selected.append(idx)

    if not selected:
        raise ValueError(
            f"No target tokens found for token_mode={token_mode}. "
            f"part={part!r}, edit={edit!r}, target_prompt={target_prompt!r}"
        )

    return sorted(set(idx for idx in selected if 0 <= idx < max_length))
