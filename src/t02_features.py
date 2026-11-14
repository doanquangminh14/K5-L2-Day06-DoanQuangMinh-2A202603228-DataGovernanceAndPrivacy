"""T02 Raw QC — image features and per-defect scores (shared by baseline and PoC).

Every per-defect score is scaled so that 1.0 is the decision threshold ("probably a defect").
The scale constants were frozen from the *dev* split only (self-labelled, see
reports/t02_dev_selflabels.csv); the eval split was never used to tune them.

Features
  lap        Laplacian variance (sharpness, higher = sharper)
  mean       mean grey level
  sat_local  max over a GxG grid of the fraction of saturated pixels (catches local plate glare)
  blockiness mean |diff| on 8-px JPEG block boundaries / mean |diff| elsewhere (both axes)

Normalisation (fixes the cam_B "soft optics" and "night is always dark" failure modes):
sharpness and brightness are compared with the median of the image's own (camera_id, light) group,
computed on the unlabelled input pool. No labels are used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common.image_ops import laplacian_variance, load_rgb, to_gray

EPS = 1e-6
DEFECTS = ("blur", "dark", "glare", "compression")

DEFAULTS = {
    "grid": 4,                  # glare is measured per cell of a grid x grid layout
    "sat_level": 245,           # grey level counted as saturated
    "blur_decades": 1.0,        # blur score 1.0 = sharpness 10x below the group median
    "dark_stops": 2.0,          # dark score 1.0 = 4x (2 stops) darker than the group median
    "bright_stops": 1.0,        # veil-glare score 1.0 = 2x brighter than the group median
    "glare_cell_frac": 0.05,    # local glare score 1.0 = 5% of one grid cell saturated
    "block_base": 1.0,          # blockiness of an image without JPEG blocking
    "block_scale": 0.8,         # compression score 1.0 = blockiness 1.8 (dev: clean <= 1.70, defects >= 1.88)
    "min_group": 4,             # smaller (camera, light) groups fall back to light-only, then global
    "action_threshold": 1.0,    # score >= this -> RECAPTURE
}


def _blockiness(g: np.ndarray) -> float:
    g = g.astype(np.float64)

    def ratio(d: np.ndarray) -> float:
        boundary = d[:, 7::8]
        inner = np.delete(d, np.s_[7::8], axis=1)
        return float(boundary.mean() / (inner.mean() + EPS))

    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0)).T
    return (ratio(dx) + ratio(dy)) / 2


def _sat_local(g: np.ndarray, grid: int, level: int) -> float:
    h, w = g.shape
    cells = [
        (g[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid] > level).mean()
        for i in range(grid) for j in range(grid)
    ]
    return float(max(cells))


def image_features(img: np.ndarray, grid: int = 4, sat_level: int = 245) -> dict:
    g = to_gray(img)
    return {
        "lap": laplacian_variance(img),
        "mean": float(g.mean()),
        "sat_local": _sat_local(g, grid, sat_level),
        "blockiness": _blockiness(g),
    }


def extract_features(images: pd.DataFrame, root: Path, params: dict) -> pd.DataFrame:
    """images: rows with image_id, path (relative to root). Missing files raise FileNotFoundError."""
    p = DEFAULTS | params
    rows = []
    for r in images.itertuples(index=False):
        f = Path(root) / r.path
        if not f.exists():
            raise FileNotFoundError(f"image listed in images.csv not found: {f}")
        rows.append({"image_id": r.image_id, **image_features(load_rgb(f), int(p["grid"]), int(p["sat_level"]))})
    return images.merge(pd.DataFrame(rows), on="image_id", how="left")


def _group_median(feats: pd.DataFrame, col: str, min_group: int) -> pd.Series:
    """Median of `col` within (camera_id, light); falls back to light, then global, for tiny groups."""
    out = pd.Series(feats[col].median(), index=feats.index)
    for keys in (["light"], ["camera_id", "light"]):
        if not all(k in feats.columns for k in keys):
            continue
        size = feats.groupby(keys)[col].transform("size")
        med = feats.groupby(keys)[col].transform("median")
        out = out.where(size < min_group, med)
    return out


def defect_scores(feats: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Adds score_<defect>, score (max), pred_defect_type, pred_action, reason."""
    p = DEFAULTS | params
    f = feats.copy()
    med_lap = _group_median(f, "lap", int(p["min_group"]))
    med_mean = _group_median(f, "mean", int(p["min_group"]))
    log_lap = np.log10(f["lap"] + EPS)
    log_mean = np.log2(f["mean"] + 1.0)

    f["score_blur"] = ((np.log10(med_lap + EPS) - log_lap) / p["blur_decades"]).clip(lower=0)
    f["score_dark"] = ((np.log2(med_mean + 1.0) - log_mean) / p["dark_stops"]).clip(lower=0)
    veil = ((log_mean - np.log2(med_mean + 1.0)) / p["bright_stops"]).clip(lower=0)
    local = f["sat_local"] / p["glare_cell_frac"]
    f["score_glare"] = np.maximum(veil, local)
    f["score_compression"] = ((f["blockiness"] - p["block_base"]) / p["block_scale"]).clip(lower=0)

    cols = [f"score_{d}" for d in DEFECTS]
    f["score"] = f[cols].max(axis=1).round(6)
    top = f[cols].idxmax(axis=1).str.replace("score_", "", regex=False)
    is_defect = f["score"] >= p["action_threshold"]
    f["pred_defect_type"] = top.where(is_defect, "none")
    f["pred_action"] = np.where(is_defect, "RECAPTURE", "KEEP")
    f["reason"] = [
        f"{t}={s:.2f} (lap={lap:.0f}, mean={m:.0f}, sat_local={sl:.3f}, block={b:.2f})"
        for t, s, lap, m, sl, b in zip(top, f["score"], f["lap"], f["mean"], f["sat_local"], f["blockiness"])
    ]
    return f
