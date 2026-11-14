"""Group tests for the T02 PoC (no ground truth needed)."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src import main as runner
from src.t02_features import defect_scores, image_features

FIX = "fixtures/t02_raw_qc"


def _run(out, *extra):
    return runner.main(["--config", "configs/lab.yaml", "--track", "t02", "--output", str(out),
                        "--offline", "--seed", "42", *extra])


# ------------------------------------------------------------------ success path
def test_poc_end_to_end_and_reproducible(tmp_path):
    assert _run(tmp_path / "a") == 0
    pred = pd.read_csv(tmp_path / "a" / "ranked_review.csv")
    imgs = pd.read_csv(f"{FIX}/images.csv")
    assert set(pred.image_id) == set(imgs.image_id) and pred.image_id.is_unique
    assert np.isfinite(pred.score).all()
    assert set(pred.pred_action) <= {"KEEP", "RECAPTURE"}
    for f in ["review_queue.csv", "contact_sheet.html", "decision.json", "decision_log.csv", "audit_log.jsonl"]:
        assert (tmp_path / "a" / f).exists(), f
    assert len(pd.read_csv(tmp_path / "a" / "review_queue.csv")) == 20
    d = json.loads((tmp_path / "a" / "decision.json").read_text(encoding="utf-8"))
    assert d["status"] == "HOLD" and d["reason_code"] == "HUMAN_REVIEW_PENDING" and d["owner"].strip()
    assert _run(tmp_path / "b") == 0
    assert (tmp_path / "a" / "ranked_review.csv").read_text(encoding="utf-8") == (tmp_path / "b" / "ranked_review.csv").read_text(encoding="utf-8")


def test_dev_selflabels_beat_baseline(tmp_path):
    """On the group's own dev labels the PoC must rank more defects into top-B than the Laplacian baseline."""
    from tools.t02_dev_check import check
    assert _run(tmp_path / "poc") == 0
    assert runner.main(["--track", "t02", "--pipeline", "src.baseline_t02", "--output", str(tmp_path / "base"),
                        "--offline", "--seed", "42", "--run-name", "baseline"]) == 0
    labels = pd.read_csv("reports/t02_dev_selflabels.csv")
    poc = check(str(tmp_path / "poc" / "ranked_review.csv"), labels, 20)
    base = check(str(tmp_path / "base" / "ranked_review.csv"), labels, 20)
    assert poc["recall@B"] > base["recall@B"]


# ------------------------------------------------------------------ failure modes from docs/tracks/T02.md §5
def _group(n_clean, lap_clean, mean_clean, odd):
    rows = [{"image_id": f"c{i}", "camera_id": "cam", "light": "x", "lap": lap_clean, "mean": mean_clean,
             "sat_local": 0.005, "blockiness": 1.1} for i in range(n_clean)]
    rows.append({"image_id": "odd", "camera_id": "cam", "light": "x", "lap": lap_clean, "mean": mean_clean,
                 "sat_local": 0.005, "blockiness": 1.1} | odd)
    return pd.DataFrame(rows)


def test_soft_camera_and_dark_night_are_not_flagged():
    """cam_B soft optics (low lap everywhere) and night (low mean everywhere) are normal for their group."""
    soft = defect_scores(_group(5, lap_clean=200.0, mean_clean=115.0, odd={}), {})
    night = defect_scores(_group(5, lap_clean=1500.0, mean_clean=30.0, odd={}), {})
    assert (soft.pred_action == "KEEP").all()
    assert (night.pred_action == "KEEP").all()


def test_local_glare_on_sharp_image_is_flagged():
    s = defect_scores(_group(5, 1000.0, 120.0, odd={"sat_local": 0.10}), {}).set_index("image_id")
    assert s.loc["odd", "pred_defect_type"] == "glare" and s.loc["odd", "pred_action"] == "RECAPTURE"


def test_features_on_constant_image_do_not_crash():
    f = image_features(np.full((180, 320, 3), 0, dtype=np.uint8))
    assert f["lap"] == 0 and f["mean"] == 0 and np.isfinite(f["blockiness"])
    s = defect_scores(pd.DataFrame([{"image_id": "z", "camera_id": "c", "light": "l", **f}]), {})
    assert np.isfinite(s.score).all()


# ------------------------------------------------------------------ human-in-the-loop
def _reviewed(out, rows):
    q = pd.read_csv(out / "review_queue.csv", dtype=str).fillna("")
    for i, (reviewer, dec, val) in enumerate(rows):
        q.loc[i, ["reviewer", "reviewer_decision", "override_value", "review_note"]] = [reviewer, dec, val, "test"]
    q.to_csv(out / "review_queue_reviewed.csv", index=False)
    return q


def test_review_overrides_are_applied_and_audited(tmp_path):
    out = tmp_path / "poc"
    assert _run(out) == 0
    q = _reviewed(out, [("alice", "edit", "none"), ("bob", "approve", "")])
    assert _run(out) == 0
    pred = pd.read_csv(out / "ranked_review.csv").set_index("image_id")
    first = q.loc[0, "item_id"]
    assert pred.loc[first, "pred_action"] == "KEEP" and pred.loc[first, "reviewed_by"] == "alice"
    events = [json.loads(l) for l in (out / "audit_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["action"] == "review:edit" and e["item_id"] == first and e["actor"] == "alice" for e in events)
    assert json.loads((out / "decision.json").read_text(encoding="utf-8"))["status"] in {"GO", "REWORK"}


def test_review_without_reviewer_fails_loudly(tmp_path):
    out = tmp_path / "poc"
    assert _run(out) == 0
    _reviewed(out, [("", "reject", "")])
    assert _run(out) == 2


def test_tampered_input_fails_provenance(tmp_path):
    src = tmp_path / "t02_raw_qc"
    shutil.copytree(FIX, src)
    img = src / "images" / "img_00_0.jpg"
    img.write_bytes(img.read_bytes() + b"x")
    assert runner.main(["--config", "configs/lab.yaml", "--track", "t02", "--input", str(src),
                        "--output", str(tmp_path / "out"), "--offline"]) == 2
