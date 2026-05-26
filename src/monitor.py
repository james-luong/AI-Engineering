"""
src/monitor.py
==============
Drift detection and performance monitoring — TPC SVM pipeline.

Usage
-----
    python src/monitor.py --mode drift    # KS drift check only (pre-retrain)
    python src/monitor.py --mode full     # full suite (post-retrain)
    python src/monitor.py                 # defaults to full

Drift detection uses the Kolmogorov-Smirnov two-sample test on every
feature in the engineered feature set (Set 3: Scaled + Engineered).

Outputs
-------
  reports/drift_report.json
  reports/monitoring_dashboard.html
  artifacts/metrics/monitoring_metrics.json   (rolling baseline)
  monitoring/alerts/*.json                    (on threshold breach)
  logs/monitoring.log
  logs/drift_detection.log
"""

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
from scipy import stats

# ── Directories ───────────────────────────────────────────────────────────────
for d in ["monitoring/reports", "monitoring/logs", "monitoring/alerts",
          "reports", "logs", "artifacts/metrics"]:
    os.makedirs(d, exist_ok=True)

# ── Loggers ───────────────────────────────────────────────────────────────────
def _make_logger(name: str, logfile: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        for h in [logging.FileHandler(logfile), logging.StreamHandler()]:
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger

monitor_log = _make_logger("monitor", "logs/monitoring.log")
drift_log   = _make_logger("drift",   "logs/drift_detection.log")

# ── Thresholds ────────────────────────────────────────────────────────────────
KS_THRESHOLD      = 0.05   # p-value; flag drift when p < this
PERF_DROP_THR     = 0.05   # flag when accuracy drops > 5 pp
NULL_FRACTION_THR = 0.10   # flag when > 10 % of a feature is null

# Feature names (Set 3: Scaled + Engineered)
FEATURE_NAMES = [
    "Temperature", "Humidity", "Wind Speed",
    "general diffuse flows", "diffuse flows",
    "Hour", "DayOfWeek", "Month",
]


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATA DRIFT — Kolmogorov-Smirnov
# ═════════════════════════════════════════════════════════════════════════════
def detect_drift(X_train: np.ndarray, X_new: np.ndarray,
                 feature_names: list) -> dict:
    """
    Two-sample KS test on every feature.
    H₀: both samples come from the same distribution.
    Reject H₀ (drift detected) when p < KS_THRESHOLD.
    """
    drift_log.info("Running KS drift detection …")
    per_feature, drifted = {}, []

    for i, feat in enumerate(feature_names):
        stat, p = stats.ks_2samp(X_train[:, i], X_new[:, i])
        is_drift = bool(p < KS_THRESHOLD)
        per_feature[feat] = {
            "ks_statistic": float(stat),
            "p_value":      float(p),
            "drifted":      is_drift,
        }
        if is_drift:
            drifted.append(feat)
            drift_log.warning(
                f"DRIFT DETECTED | {feat:35s} | KS={stat:.4f} | p={p:.4f}"
            )
        else:
            drift_log.info(
                f"No drift       | {feat:35s} | KS={stat:.4f} | p={p:.4f}"
            )

    result = {
        "timestamp":         datetime.utcnow().isoformat() + "Z",
        "n_features_tested": len(feature_names),
        "n_drifted":         len(drifted),
        "drifted_features":  drifted,
        "drift_threshold":   KS_THRESHOLD,
        "overall_drift":     len(drifted) > 0,
        "per_feature":       per_feature,
    }
    drift_log.info(
        f"Drift summary: {len(drifted)}/{len(feature_names)} features drifted"
    )
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 2.  DATA QUALITY
# ═════════════════════════════════════════════════════════════════════════════
def check_data_quality(X_new: np.ndarray, feature_names: list) -> dict:
    """Null fraction, infinite values, and extreme outlier checks."""
    monitor_log.info("Running data quality checks …")
    issues, per_feat = [], {}

    for i, feat in enumerate(feature_names):
        col          = X_new[:, i]
        null_frac    = float(np.isnan(col).mean())
        inf_count    = int(np.isinf(col).sum())
        mean_v       = float(np.nanmean(col))
        std_v        = float(np.nanstd(col))
        outlier_frac = float((np.abs((col - mean_v) / (std_v + 1e-8)) > 10).mean())

        feat_issues = []
        if null_frac    > NULL_FRACTION_THR: feat_issues.append(f"High nulls: {null_frac:.1%}")
        if inf_count    > 0:                 feat_issues.append(f"Inf values: {inf_count}")
        if outlier_frac > 0.01:              feat_issues.append(f"Outliers >10σ: {outlier_frac:.1%}")

        per_feat[feat] = {
            "null_fraction":    null_frac,
            "inf_count":        inf_count,
            "outlier_fraction": outlier_frac,
            "issues":           feat_issues,
        }
        if feat_issues:
            issues.extend([f"{feat}: {x}" for x in feat_issues])
            monitor_log.warning(f"Quality issue — {feat}: {feat_issues}")

    result = {
        "timestamp":    datetime.utcnow().isoformat() + "Z",
        "n_samples":    int(X_new.shape[0]),
        "n_features":   int(X_new.shape[1]),
        "total_issues": len(issues),
        "issues":       issues,
        "per_feature":  per_feat,
        "passed":       len(issues) == 0,
    }
    monitor_log.info(
        f"Data quality: {'PASSED' if result['passed'] else 'ISSUES FOUND'} "
        f"({len(issues)} issues)"
    )
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 3.  PERFORMANCE MONITORING
# ═════════════════════════════════════════════════════════════════════════════
def monitor_performance() -> dict:
    """
    Compare current evaluation_metrics.json against the saved baseline
    in monitoring_metrics.json. Flags accuracy drop > PERF_DROP_THR.
    """
    monitor_log.info("Running performance monitoring …")
    current_path  = "artifacts/metrics/evaluation_metrics.json"
    baseline_path = "artifacts/metrics/monitoring_metrics.json"

    if not os.path.exists(current_path):
        monitor_log.warning("evaluation_metrics.json not found — skipping")
        return {"status": "skipped"}

    with open(current_path) as fh:
        current = json.load(fh)

    perf = {
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "current_accuracy": current.get("accuracy"),
        "current_f1":       current.get("f1_score"),
        "current_roc_auc":  current.get("roc_auc"),
        "alert":            False,
    }

    if os.path.exists(baseline_path):
        with open(baseline_path) as fh:
            baseline = json.load(fh)
        base_acc = baseline.get("current_accuracy") or baseline.get("accuracy")
        if base_acc and perf["current_accuracy"]:
            drop = base_acc - perf["current_accuracy"]
            perf["accuracy_drop"]     = float(drop)
            perf["baseline_accuracy"] = float(base_acc)
            if drop > PERF_DROP_THR:
                perf["alert"] = True
                monitor_log.warning(
                    f"PERFORMANCE ALERT: accuracy {base_acc:.4f} → "
                    f"{perf['current_accuracy']:.4f} (Δ={drop:+.4f})"
                )
            else:
                monitor_log.info(
                    f"Performance OK: {base_acc:.4f} → "
                    f"{perf['current_accuracy']:.4f} (Δ={drop:+.4f})"
                )
    else:
        monitor_log.info("No baseline found — saving current metrics as baseline")

    with open(baseline_path, "w") as fh:
        json.dump(perf, fh, indent=2)

    return perf


# ═════════════════════════════════════════════════════════════════════════════
# 4.  ALERTING
# ═════════════════════════════════════════════════════════════════════════════
def save_alert(alert_type: str, detail: dict) -> None:
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"monitoring/alerts/{alert_type}_{ts}.json"
    with open(path, "w") as fh:
        json.dump({"type": alert_type, "timestamp": ts, **detail}, fh, indent=2)
    monitor_log.warning(f"Alert written → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  REPORTS
# ═════════════════════════════════════════════════════════════════════════════
def generate_drift_report(drift: dict, quality: dict) -> None:
    path = "reports/drift_report.json"
    with open(path, "w") as fh:
        json.dump({"drift": drift, "data_quality": quality}, fh, indent=2)
    monitor_log.info(f"Drift report saved → {path}")


def generate_dashboard(drift, quality, perf) -> None:
    """Self-contained HTML monitoring dashboard."""

    def badge(ok, yes="OK", no="ALERT"):
        c = "#27ae60" if ok else "#e74c3c"
        l = yes if ok else no
        return (f'<span style="background:{c};color:#fff;padding:4px 12px;'
                f'border-radius:12px;font-size:.85em;font-weight:600">{l}</span>')

    db = badge(not (drift  or {}).get("overall_drift", False), "No Drift",  "Drift Detected")
    qb = badge(    (quality or {}).get("passed", True),         "Passed",   "Issues Found")
    pb = badge(not (perf   or {}).get("alert", False),          "OK",       "Performance Drop")
    ts = datetime.utcnow().isoformat() + "Z"

    # Drift table
    drift_rows = ""
    if drift and "per_feature" in drift:
        for feat, d in drift["per_feature"].items():
            bg = "#fde8e8" if d["drifted"] else "#eafaf1"
            drift_rows += (
                f'<tr style="background:{bg}">'
                f"<td>{feat}</td><td>{d['ks_statistic']:.4f}</td>"
                f"<td>{d['p_value']:.4f}</td>"
                f"<td>{'&#10004; Yes' if d['drifted'] else 'No'}</td></tr>"
            )

    # Quality table
    qual_rows = ""
    if quality and "per_feature" in quality:
        for feat, q in quality["per_feature"].items():
            iss = "; ".join(q.get("issues", [])) or "None"
            bg  = "#fde8e8" if q.get("issues") else "#eafaf1"
            qual_rows += (
                f'<tr style="background:{bg}">'
                f"<td>{feat}</td><td>{q['null_fraction']:.1%}</td>"
                f"<td>{q['inf_count']}</td><td>{q['outlier_fraction']:.1%}</td>"
                f"<td>{iss}</td></tr>"
            )

    # Performance block
    def fmt(v): return f"{v:.4f}" if isinstance(v, float) else ("N/A" if v is None else str(v))
    perf_rows = ""
    if perf and perf.get("current_accuracy") is not None:
        for label, key in [
            ("Current Accuracy",  "current_accuracy"),
            ("Baseline Accuracy", "baseline_accuracy"),
            ("Accuracy Drop (Δ)", "accuracy_drop"),
            ("F1 Score",          "current_f1"),
            ("ROC-AUC",           "current_roc_auc"),
        ]:
            perf_rows += f"<tr><td>{label}</td><td>{fmt(perf.get(key))}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>TPC MLOps Monitoring Dashboard</title>
  <style>
    * {{ box-sizing:border-box; }}
    body {{ font-family:'Segoe UI',Arial,sans-serif; margin:40px; background:#f0f4f8; color:#2c3e50; }}
    h1   {{ border-bottom:3px solid #8e44ad; padding-bottom:8px; }}
    h2   {{ color:#8e44ad; margin-top:30px; }}
    p.ts {{ color:#95a5a6; font-size:.9em; }}
    p.sub {{ color:#555; margin-bottom:24px; }}
    .summary {{ display:flex; gap:16px; flex-wrap:wrap; margin:20px 0 32px; }}
    .card    {{ background:#fff; border-radius:10px; padding:18px 28px;
                box-shadow:0 2px 8px rgba(0,0,0,.09); text-align:center; min-width:160px; }}
    .card .lbl {{ font-size:.78em; text-transform:uppercase; color:#7f8c8d; }}
    .card .val {{ margin-top:8px; }}
    table {{ border-collapse:collapse; width:100%; background:#fff; border-radius:8px;
             overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,.07); margin-bottom:24px; }}
    th,td {{ padding:10px 15px; border-bottom:1px solid #ecf0f1; text-align:left; }}
    thead th {{ background:#8e44ad; color:#fff; }}
    tr:last-child td {{ border-bottom:none; }}
    tbody tr:hover td {{ background:#f5eef8; }}
  </style>
</head>
<body>
  <h1>TPC MLOps Monitoring Dashboard</h1>
  <p class="ts">Generated: {ts}</p>
  <p class="sub">Dataset: Tetuan City Power Consumption &nbsp;|&nbsp;
     Model: SVM (RBF, C=10, γ=0.1) &nbsp;|&nbsp;
     Target: Zone 1 Power Consumption (Low / Medium / High)</p>

  <div class="summary">
    <div class="card"><div class="lbl">Data Drift</div>   <div class="val">{db}</div></div>
    <div class="card"><div class="lbl">Data Quality</div> <div class="val">{qb}</div></div>
    <div class="card"><div class="lbl">Performance</div>  <div class="val">{pb}</div></div>
  </div>

  <h2>Feature Drift (Kolmogorov-Smirnov Test)</h2>
  <p style="color:#555;font-size:.9em">
    Threshold: p &lt; {KS_THRESHOLD} → drift flagged.
    Features: Temperature, Humidity, Wind Speed, general diffuse flows, diffuse flows, Hour, DayOfWeek, Month.
  </p>
  <table>
    <thead><tr><th>Feature</th><th>KS Statistic</th><th>p-value</th><th>Drifted?</th></tr></thead>
    <tbody>{drift_rows or "<tr><td colspan='4'>No drift data available</td></tr>"}</tbody>
  </table>

  <h2>Data Quality</h2>
  <table>
    <thead><tr><th>Feature</th><th>Null %</th><th>Inf Count</th><th>Outlier %</th><th>Issues</th></tr></thead>
    <tbody>{qual_rows or "<tr><td colspan='5'>No quality data available</td></tr>"}</tbody>
  </table>

  <h2>Performance vs Baseline</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{perf_rows or "<tr><td colspan='2'>No performance data available</td></tr>"}</tbody>
  </table>
</body>
</html>"""

    path = "reports/monitoring_dashboard.html"
    with open(path, "w") as fh:
        fh.write(html)
    monitor_log.info(f"Monitoring dashboard saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 6.  ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════════════
def _load_drift_inputs():
    """Load X_train, X_new, and feature names if available."""
    X_train_path = "artifacts/data/X_train.npy"
    X_new_path   = "artifacts/data/X_new.npy"
    feat_path    = "artifacts/preprocessing/feature_columns.json"

    if not os.path.exists(X_new_path):
        monitor_log.info("X_new.npy not found — skipping drift detection")
        return None, None, None

    X_train = np.load(X_train_path)
    X_new   = np.load(X_new_path)
    with open(feat_path) as fh:
        feature_names = json.load(fh)["features"]
    return X_train, X_new, feature_names


def run_drift_mode():
    monitor_log.info("=== MODE: DRIFT CHECK ONLY ===")
    X_train, X_new, features = _load_drift_inputs()
    drift, quality = None, None
    if X_new is not None:
        drift   = detect_drift(X_train, X_new, features)
        quality = check_data_quality(X_new, features)
        if drift["overall_drift"]:
            save_alert("data_drift", drift)
        generate_drift_report(drift, quality)
    generate_dashboard(drift, quality, None)
    monitor_log.info("=== DRIFT CHECK COMPLETE ===")


def run_full_mode():
    monitor_log.info("=== MODE: FULL MONITORING ===")
    X_train, X_new, features = _load_drift_inputs()
    drift, quality = None, None
    if X_new is not None:
        drift   = detect_drift(X_train, X_new, features)
        quality = check_data_quality(X_new, features)
        if drift["overall_drift"]:
            save_alert("data_drift", drift)
        generate_drift_report(drift, quality)
    perf = monitor_performance()
    if perf.get("alert"):
        save_alert("performance_drop", perf)
    generate_dashboard(drift, quality, perf)
    monitor_log.info("=== FULL MONITORING COMPLETE ===")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPC MLOps Monitoring")
    parser.add_argument(
        "--mode", choices=["drift", "full"], default="full",
        help="'drift' = pre-retrain KS check; 'full' = complete monitoring suite",
    )
    args = parser.parse_args()
    if args.mode == "drift":
        run_drift_mode()
    else:
        run_full_mode()
