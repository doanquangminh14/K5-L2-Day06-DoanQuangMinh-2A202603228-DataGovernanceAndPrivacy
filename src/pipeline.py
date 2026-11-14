"""PoC pipeline — T02 Raw QC Triage (blur / glare / dark / compression).

Lab contract = 5 automated steps, run by `python -m src.main ...`:
    01 load      images.csv + provenance check (manifest.csv sha256)
    02 process   per-image features -> per-defect scores normalised by (camera_id, light) group
    03 rank      top-B review queue + offline HTML contact sheet
    04 review    reviewer decisions -> overrides (pred_defect_type / pred_action) + audit log
    05 decide    batch GO / HOLD / REWORK with reason_code + owner; per image KEEP / RECAPTURE

Output: out/ranked_review.csv with image_id, score (+ pred_defect_type, pred_action, features).
Rules: never read fixtures/*/gt/; no network; thresholds frozen in configs/lab.yaml (params).
"""
from __future__ import annotations

import pandas as pd

from .common.contact_sheet import render_contact_sheet
from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.manifest import verify_manifest
from .common.review import load_review_decisions, make_review_queue, write_review_queue
from .metrics.tracks import get_track
from .t02_features import DEFECTS, defect_scores, extract_features

REVIEWED_FILE = "review_queue_reviewed.csv"
VALID_TYPES = set(DEFECTS) | {"none"}


def _require_t02(ctx: RunContext) -> None:
    if ctx.track != "t02":
        spec = get_track(ctx.track)
        raise NotImplementedError(f"this group's pipeline implements t02 only, not {spec.track} ({spec.title})")


# ---------------------------------------------------------------- 01 load
def load(ctx: RunContext) -> dict:
    _require_t02(ctx)
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
    images_csv = ctx.input_dir / "images.csv"
    if not images_csv.exists():
        raise FileNotFoundError(f"missing {images_csv}")
    images = pd.read_csv(images_csv)
    missing = [c for c in ("image_id", "path") if c not in images.columns]
    if missing:
        raise ValueError(f"images.csv missing columns {missing}")
    if images.image_id.duplicated().any():
        raise ValueError(f"images.csv has duplicate image_id: {images.image_id[images.image_id.duplicated()].tolist()[:5]}")
    return {"manifest": pd.read_csv(manifest) if manifest.exists() else None, "images": images}


# ---------------------------------------------------------------- 02 process
def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    params = ctx.config.get("params") or {}
    feats = extract_features(data["images"], ctx.input_dir, params)
    scored = defect_scores(feats, params)
    ctx.stats["n_recapture_raw"] = int((scored.pred_action == "RECAPTURE").sum())
    lead = ["image_id", "score", "pred_defect_type", "pred_action", "reason"]
    return scored[lead + [c for c in scored.columns if c not in lead]]


# ---------------------------------------------------------------- 03 rank
def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    budget = int(ctx.param("budget", 20))
    s = scored.assign(slice=scored.get("camera_id", "").astype(str) + "/" + scored.get("light", "").astype(str))
    q = make_review_queue(s, id_col="image_id", score_col="score", budget=budget, reason_col="reason",
                          action_col="pred_action", slice_col="slice", evidence_col="path")
    write_review_queue(q, ctx.out("review_queue.csv"))
    render_contact_sheet(q, ctx.out("contact_sheet.html"), image_root=ctx.input_dir,
                         title=f"T02 Raw QC — top {budget} for review")
    return q


# ---------------------------------------------------------------- 04 review
def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    """Reviewer semantics (out/review_queue_reviewed.csv):
        approve  -> keep the pipeline's call
        edit     -> override_value = true defect type (blur/dark/glare/compression/none)
        reject   -> pipeline call is wrong: flip KEEP <-> RECAPTURE (type -> none / unknown)
        escalate -> pred_action = ESCALATE (blocks a GO)
        skip     -> nothing
    `score` is never changed, so the ranking stays the model's own (honest evaluation).
    """
    final = scored.copy()
    final["reviewed_by"] = ""
    reviewed_path = ctx.output_dir / REVIEWED_FILE
    if not reviewed_path.exists():
        ctx.stats["n_reviewed"] = 0
        return final
    reviewed = load_review_decisions(reviewed_path)
    ctx.stats["n_reviewed"] = len(reviewed)
    ids = set(final.image_id.astype(str))
    for r in reviewed.itertuples(index=False):
        if r.item_id not in ids:
            raise ValueError(f"reviewed item {r.item_id!r} is not in this batch")
        m = final.image_id.astype(str) == r.item_id
        before = final.loc[m, ["pred_defect_type", "pred_action"]].iloc[0].to_dict()
        dec = r.reviewer_decision
        if dec == "edit":
            t = str(r.override_value).strip().lower()
            if t not in VALID_TYPES:
                raise ValueError(f"{r.item_id}: override_value {r.override_value!r} not in {sorted(VALID_TYPES)}")
            final.loc[m, "pred_defect_type"] = t
            final.loc[m, "pred_action"] = "KEEP" if t == "none" else "RECAPTURE"
        elif dec == "reject":
            if before["pred_action"] == "RECAPTURE":
                final.loc[m, ["pred_defect_type", "pred_action"]] = ["none", "KEEP"]
            else:
                final.loc[m, ["pred_defect_type", "pred_action"]] = ["unknown", "RECAPTURE"]
        elif dec == "escalate":
            final.loc[m, "pred_action"] = "ESCALATE"
        final.loc[m, "reviewed_by"] = r.reviewer
        after = final.loc[m, ["pred_defect_type", "pred_action"]].iloc[0].to_dict()
        ctx.audit.log(actor=r.reviewer, action=f"review:{dec}", item_id=r.item_id, before=before, after=after,
                      reason=r.review_note)
    return final


# ---------------------------------------------------------------- 05 decide
def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    """Batch policy (frozen in configs/lab.yaml before any evaluation):
        any ESCALATE                       -> HOLD   ESCALATED_ITEMS
        top-B queue not reviewed by human  -> HOLD   HUMAN_REVIEW_PENDING
        recapture rate > max_recapture_rate-> REWORK RECAPTURE_RATE_HIGH (camera/process problem)
        otherwise                          -> GO     QC_PASSED_WITH_RECAPTURES (RECAPTURE images excluded)
    """
    owner = ctx.config.get("owner", "")
    n = len(final)
    n_rec = int((final.pred_action == "RECAPTURE").sum())
    n_esc = int((final.pred_action == "ESCALATE").sum())
    rate = n_rec / n if n else 0.0
    max_rate = float(ctx.param("max_recapture_rate", 0.5))
    metrics = {"n_images": n, "n_recapture": n_rec, "n_escalate": n_esc, "recapture_rate": round(rate, 4),
               "n_reviewed": ctx.stats.get("n_reviewed", 0), "max_recapture_rate": max_rate}
    evidence = [str(ctx.out(get_track(ctx.track).pred_file)), str(ctx.out("review_queue.csv"))]
    if n_esc:
        status, code, why = "HOLD", "ESCALATED_ITEMS", f"{n_esc} image(s) escalated by reviewer, need domain owner"
    elif ctx.stats.get("n_reviewed", 0) == 0:
        status, code, why = "HOLD", "HUMAN_REVIEW_PENDING", f"top-{ctx.param('budget', 20)} queue not reviewed yet ({n_rec} flagged RECAPTURE)"
    elif rate > max_rate:
        status, code, why = "REWORK", "RECAPTURE_RATE_HIGH", f"{n_rec}/{n} images need recapture (> {max_rate:.0%})"
    else:
        status, code, why = "GO", "QC_PASSED_WITH_RECAPTURES", f"release {n - n_rec} images; {n_rec} sent to recapture"
    return Decision(status=status, reason_code=code, reason=why, owner=owner, scope=f"{ctx.track}:{ctx.input_dir.name}",
                    evidence=evidence, metrics=metrics)


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored)
    ctx.audit.log(actor="pipeline", action="review_queue_created", item_id="review_queue.csv",
                  after={"budget": len(queue), "items": queue.item_id.tolist()},
                  reason="top-B by score; awaiting reviewer decisions in review_queue_reviewed.csv")
    final = review(ctx, scored, queue)
    final = final.sort_values("score", ascending=False, kind="mergesort")
    final[spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]].to_csv(ctx.out(spec.pred_file), index=False)
    decision = decide(ctx, final)
    write_decision(decision, ctx.out("decision.json"), ctx.out("decision_log.csv"))
    ctx.audit.log(actor=decision.owner, action=f"decision:{decision.canonical}", item_id=decision.scope,
                  after=decision.metrics, reason=f"{decision.reason_code}: {decision.reason}")
