"""T02 baseline required by docs/tracks/T02.md §4: ONE global Laplacian threshold.

    score       = -log10(laplacian_variance)   (less sharp = review first)
    pred_action = RECAPTURE if lap < baseline_lap_threshold (frozen on dev, see configs/lab.yaml)

No grouping by camera/light, no brightness/glare/compression features — this is the bar the PoC must beat
on the same pool, same budget B, same seed.

    python -m src.main --track t02 --pipeline src.baseline_t02 --output out/baseline --run-name baseline --offline --seed 42
"""
from __future__ import annotations

import numpy as np

from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.review import make_review_queue, write_review_queue
from .metrics.tracks import get_track
from .pipeline import load
from .t02_features import EPS, extract_features


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = load(ctx)
    params = ctx.config.get("params") or {}
    f = extract_features(data["images"], ctx.input_dir, params)
    thr = float(ctx.param("baseline_lap_threshold", 100.0))
    f["score"] = (-np.log10(f["lap"] + EPS)).round(6)
    f["pred_action"] = np.where(f["lap"] < thr, "RECAPTURE", "KEEP")
    f["pred_defect_type"] = np.where(f["lap"] < thr, "blur", "none")
    f["reason"] = [f"lap={v:.0f} vs global threshold {thr:.0f}" for v in f["lap"]]
    ctx.stats["n_items"] = len(f)
    q = make_review_queue(f, "image_id", "score", int(ctx.param("budget", 20)), reason_col="reason",
                          action_col="pred_action", evidence_col="path")
    write_review_queue(q, ctx.out("review_queue.csv"))
    f = f.sort_values("score", ascending=False, kind="mergesort")
    cols = ["image_id", "score", "pred_defect_type", "pred_action", "reason"]
    f[cols + [c for c in f.columns if c not in cols]].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(Decision(status="HOLD", reason_code="BASELINE_ONLY", reason="Single global Laplacian threshold baseline; not a release policy",
                            owner=ctx.config.get("owner", "baseline"), scope=ctx.track, evidence=[str(ctx.out(spec.pred_file))]),
                   ctx.out("decision.json"), ctx.out("decision_log.csv"))
