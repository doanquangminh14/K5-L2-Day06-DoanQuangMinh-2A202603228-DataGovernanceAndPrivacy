"""Internal dev-split check for T02 using the group's OWN labels (reports/t02_dev_selflabels.csv).

The official ground truth (fixtures/*/gt/) is instructor-only. This script never reads it; it scores a
ranked_review.csv against the self-labelled dev images so thresholds can be frozen without touching eval.

    python -m tools.t02_dev_check out/baseline/ranked_review.csv out/poc/ranked_review.csv
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

LABELS = "reports/t02_dev_selflabels.csv"


def check(pred_path: str, labels: pd.DataFrame, budget: int) -> dict:
    pred = pd.read_csv(pred_path)
    pred = pred[pred.image_id.isin(labels.image_id)].sort_values("score", ascending=False, kind="mergesort")
    pos = set(labels.loc[labels.defect_type != "none", "image_id"])
    top = list(pred.image_id.head(budget))
    hits = sum(i in pos for i in top)
    out = {"file": pred_path, "N_dev": len(labels), "defects": len(pos), "B": budget,
           "recall@B": round(hits / len(pos), 3), "precision@B": round(hits / budget, 3),
           "max_possible_recall@B": round(min(budget, len(pos)) / len(pos), 3)}
    if "pred_action" in pred.columns:
        flagged = set(pred.loc[pred.pred_action == "RECAPTURE", "image_id"])
        tp = len(flagged & pos)
        out["action_recall"] = round(tp / len(pos), 3)
        out["action_precision"] = round(tp / len(flagged), 3) if flagged else None
        out["false_recapture"] = sorted(flagged - pos)
        out["missed"] = sorted(pos - flagged)
    if "pred_defect_type" in pred.columns:
        m = labels[labels.defect_type != "none"].merge(pred[["image_id", "pred_defect_type"]], on="image_id")
        out["type_accuracy"] = round((m.defect_type == m.pred_defect_type).mean(), 3)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", nargs="+")
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--budget", type=int, default=20)
    a = ap.parse_args(argv)
    labels = pd.read_csv(a.labels)
    for p in a.pred:
        for k, v in check(p, labels, a.budget).items():
            print(f"{k:>22}: {v}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
