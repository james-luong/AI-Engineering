"""
src/evaluate.py
===============
Evaluation script — Tetuan City Power Consumption (TPC) SVM pipeline.

Loads the trained SVM model and test arrays, computes classification
metrics, saves JSON results and a self-contained HTML performance report.

Outputs
-------
  artifacts/metrics/evaluation_metrics.json
  reports/performance_report.html
"""

import json
import logging
import os
from datetime import datetime

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Directories ───────────────────────────────────────────────────────────────
for d in ["reports", "artifacts/metrics", "logs"]:
    os.makedirs(d, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
def load_artifacts():
    logger.info("Loading SVM model from artifacts/models/model.pkl")
    model = joblib.load("artifacts/models/model.pkl")

    logger.info("Loading test arrays from artifacts/data/")
    X_test = np.load("artifacts/data/X_test.npy")
    y_test = np.load("artifacts/data/y_test.npy")

    logger.info("Loading encoder from artifacts/preprocessing/encoder.pkl")
    encoder = joblib.load("artifacts/preprocessing/encoder.pkl")

    return model, X_test, y_test, encoder


# ─────────────────────────────────────────────────────────────────────────────
# 2.  COMPUTE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(model, X_test: np.ndarray, y_test: np.ndarray, encoder) -> dict:
    """Inference + full classification metric suite."""
    logger.info("Running inference on test set …")
    y_pred = model.predict(X_test)

    # Probabilities for ROC-AUC (model trained with probability=True)
    try:
        y_proba = model.predict_proba(X_test)
        roc_auc = float(
            roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        )
    except Exception:
        roc_auc = None

    accuracy  = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall    = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1        = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    cm        = confusion_matrix(y_test, y_pred).tolist()
    report    = classification_report(
        y_test, y_pred,
        target_names=[str(c) for c in encoder.classes_],
        output_dict=True,
    )

    metrics = {
        "timestamp":             datetime.utcnow().isoformat() + "Z",
        "test_samples":          int(len(y_test)),
        "num_classes":           len(encoder.classes_),
        "class_names":           [str(c) for c in encoder.classes_],
        "accuracy":              accuracy,
        "precision":             precision,
        "recall":                recall,
        "f1_score":              f1,
        "roc_auc":               roc_auc,
        "confusion_matrix":      cm,
        "classification_report": report,
    }

    logger.info(f"Accuracy  = {accuracy:.4f}")
    logger.info(f"Precision = {precision:.4f}")
    logger.info(f"Recall    = {recall:.4f}")
    logger.info(f"F1 Score  = {f1:.4f}")
    if roc_auc:
        logger.info(f"ROC-AUC   = {roc_auc:.4f}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SAVE JSON
# ─────────────────────────────────────────────────────────────────────────────
def save_metrics(metrics: dict) -> None:
    path = "artifacts/metrics/evaluation_metrics.json"
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info(f"Evaluation metrics saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  HTML REPORT
# ─────────────────────────────────────────────────────────────────────────────
def generate_html_report(metrics: dict) -> None:
    """Render a self-contained HTML performance report (no external CDN)."""

    def pct(v):
        return f"{v * 100:.2f}%" if v is not None else "N/A"

    cm      = metrics["confusion_matrix"]
    classes = metrics["class_names"]
    ts      = metrics["timestamp"]

    # Confusion matrix HTML
    cm_header = "<tr><th>Actual \\ Predicted</th>" + "".join(
        f"<th>{c}</th>" for c in classes
    ) + "</tr>"
    cm_rows = ""
    for i, row in enumerate(cm):
        cm_rows += f"<tr><th>{classes[i]}</th>"
        for j, val in enumerate(row):
            hl = ' class="hl"' if i == j else ""
            cm_rows += f"<td{hl}>{val}</td>"
        cm_rows += "</tr>"

    # Per-class rows
    report = metrics["classification_report"]
    per_class_rows = ""
    for cls in classes:
        if cls in report:
            r = report[cls]
            per_class_rows += (
                f"<tr><td>{cls}</td>"
                f"<td>{r['precision']:.4f}</td>"
                f"<td>{r['recall']:.4f}</td>"
                f"<td>{r['f1-score']:.4f}</td>"
                f"<td>{int(r['support'])}</td></tr>"
            )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>TPC Model Performance Report</title>
  <style>
    * {{ box-sizing:border-box; }}
    body  {{ font-family:'Segoe UI',Arial,sans-serif; margin:40px; background:#f5f7fa; color:#2c3e50; }}
    h1    {{ border-bottom:3px solid #2980b9; padding-bottom:8px; }}
    h2    {{ color:#2980b9; margin-top:30px; }}
    p.ts  {{ color:#95a5a6; font-size:.9em; }}
    p.sub {{ color:#555; margin-bottom:24px; }}
    .cards{{ display:flex; gap:16px; flex-wrap:wrap; margin:20px 0 32px; }}
    .card {{ background:#fff; border-radius:10px; padding:18px 28px;
              box-shadow:0 2px 8px rgba(0,0,0,.09); text-align:center; min-width:130px; }}
    .card .lbl {{ font-size:.78em; text-transform:uppercase; letter-spacing:.04em; color:#7f8c8d; }}
    .card .val {{ font-size:1.9em; font-weight:700; margin-top:6px; color:#2c3e50; }}
    table {{ border-collapse:collapse; width:100%; background:#fff; border-radius:8px;
             overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,.07); margin-bottom:24px; }}
    th,td {{ padding:10px 15px; border-bottom:1px solid #ecf0f1; }}
    thead th {{ background:#2980b9; color:#fff; }}
    tr:last-child td {{ border-bottom:none; }}
    td.hl {{ background:#d5f5e3; font-weight:700; }}
    tbody tr:hover td {{ background:#eaf4ff; }}
  </style>
</head>
<body>
  <h1>TPC Model Performance Report</h1>
  <p class="ts">Generated: {ts} &nbsp;|&nbsp; Test samples: {metrics['test_samples']}</p>
  <p class="sub">Model: SVM (RBF kernel, C=10, γ=0.1) &nbsp;|&nbsp;
     Dataset: Tetuan City Power Consumption &nbsp;|&nbsp;
     Target: Zone 1 Power Consumption (Low / Medium / High)</p>

  <h2>Summary Metrics</h2>
  <div class="cards">
    <div class="card"><div class="lbl">Accuracy</div> <div class="val">{pct(metrics['accuracy'])}</div></div>
    <div class="card"><div class="lbl">Precision</div><div class="val">{pct(metrics['precision'])}</div></div>
    <div class="card"><div class="lbl">Recall</div>   <div class="val">{pct(metrics['recall'])}</div></div>
    <div class="card"><div class="lbl">F1 Score</div> <div class="val">{pct(metrics['f1_score'])}</div></div>
    <div class="card"><div class="lbl">ROC-AUC</div>  <div class="val">{pct(metrics['roc_auc'])}</div></div>
  </div>

  <h2>Per-Class Breakdown</h2>
  <table>
    <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
    <tbody>{per_class_rows}</tbody>
  </table>

  <h2>Confusion Matrix</h2>
  <p style="color:#555;font-size:.9em">Diagonal cells (green) = correct predictions</p>
  <table>
    <thead>{cm_header}</thead>
    <tbody>{cm_rows}</tbody>
  </table>
</body>
</html>"""

    path = "reports/performance_report.html"
    with open(path, "w") as fh:
        fh.write(html)
    logger.info(f"HTML performance report saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  MLOps Pipeline — TPC Model Evaluation")
    logger.info("=" * 60)

    model, X_test, y_test, encoder = load_artifacts()
    metrics = compute_metrics(model, X_test, y_test, encoder)
    save_metrics(metrics)
    generate_html_report(metrics)

    logger.info("=" * 60)
    logger.info("  Evaluation complete")
    logger.info("=" * 60)
