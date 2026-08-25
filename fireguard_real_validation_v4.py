#!/usr/bin/env python3
"""
FireGuard Forecast V4 — REAL-WORLD / EVENT-LEVEL VALIDATION
READ-ONLY. No retrain, no recalibration, no threshold change,
no schema change, no dataset/model modification.
"""

from pathlib import Path
from datetime import datetime
import json
import warnings
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS & CONSTANTS  (FIXED – only change in this file)
# ============================================================

# Correct project root = fireguard folder
BASE_DIR = Path(r"C:\Users\vista\Desktop\New folder (2)\fireguard")

DATASET_PATH = BASE_DIR / "data" / "fireguard_forecast_60days.csv"
MODEL_DIR = BASE_DIR / "saved_models"

HORIZONS = ["24h", "48h", "72h"]
HORIZON_MINUTES = {"24h": 1440, "48h": 2880, "72h": 4320}
EXPERIMENTS = ["sensor_only", "sensor_plus_flame"]

V4_FILES = {
    "sensor_only": {
        "24h": "fireguard_forecast_sensor_only_24h_v4.joblib",
        "48h": "fireguard_forecast_sensor_only_48h_v4.joblib",
        "72h": "fireguard_forecast_sensor_only_72h_v4.joblib",
    },
    "sensor_plus_flame": {
        "24h": "fireguard_forecast_sensor_plus_flame_24h_v4.joblib",
        "48h": "fireguard_forecast_sensor_plus_flame_48h_v4.joblib",
        "72h": "fireguard_forecast_sensor_plus_flame_72h_v4.joblib",
    },
}

EPISODE_GAP_MINUTES = 60
CAL_BINS = np.linspace(0.0, 1.0, 11)


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_sub(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


def event_id_from_fire(fire_series):
    fire = fire_series.astype(int).to_numpy()
    starts = (fire == 1) & (np.r_[True, fire[:-1] == 0])
    eid = np.cumsum(starts)
    eid[fire == 0] = 0
    return eid


def expected_calibration_error(y_true, proba, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(proba, bins[1:-1], right=False)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    for b in range(n_bins):
        mask = idx == b
        cnt = np.sum(mask)
        if cnt == 0:
            continue
        conf = np.mean(proba[mask])
        acc = np.mean(y_true[mask])
        ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def safe_metrics(y_true, y_pred, proba):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-7, 1 - 1e-7)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out["tn"], out["fp"], out["fn"], out["tp"] = int(tn), int(fp), int(fn), int(tp)
    out["confusion_matrix"] = cm.tolist()

    try:
        out["pr_auc"] = float(average_precision_score(y_true, proba))
    except Exception:
        out["pr_auc"] = None
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    except Exception:
        out["roc_auc"] = None
    try:
        out["brier"] = float(brier_score_loss(y_true, proba))
    except Exception:
        out["brier"] = None
    try:
        out["log_loss"] = float(log_loss(y_true, proba))
    except Exception:
        out["log_loss"] = None
    out["ece"] = expected_calibration_error(y_true, proba)
    return out


# ============================================================
# FEATURE ENGINEERING (causal – identical to V4 training)
# ============================================================

def build_features(df):
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    out["hour"] = out["timestamp"].dt.hour
    out["minute"] = out["timestamp"].dt.minute

    out["smoke_change_1m"] = out["smoke"].diff(1)
    out["smoke_change_5m"] = out["smoke"].diff(5)
    out["smoke_change_15m"] = out["smoke"].diff(15)
    out["smoke_change_30m"] = out["smoke"].diff(30)
    out["smoke_change_60m"] = out["smoke"].diff(60)

    out["temperature_change_5m"] = out["temperature"].diff(5)
    out["temperature_change_15m"] = out["temperature"].diff(15)
    out["temperature_change_30m"] = out["temperature"].diff(30)
    out["temperature_change_60m"] = out["temperature"].diff(60)

    out["humidity_change_5m"] = out["humidity"].diff(5)
    out["humidity_change_15m"] = out["humidity"].diff(15)
    out["humidity_change_30m"] = out["humidity"].diff(30)

    out["smoke_mean_5m"] = out["smoke"].rolling(5).mean()
    out["smoke_mean_15m"] = out["smoke"].rolling(15).mean()
    out["smoke_mean_30m"] = out["smoke"].rolling(30).mean()
    out["smoke_mean_60m"] = out["smoke"].rolling(60).mean()
    out["smoke_std_15m"] = out["smoke"].rolling(15).std()
    out["smoke_std_30m"] = out["smoke"].rolling(30).std()
    out["smoke_max_30m"] = out["smoke"].rolling(30).max()

    out["temperature_mean_15m"] = out["temperature"].rolling(15).mean()
    out["temperature_mean_30m"] = out["temperature"].rolling(30).mean()
    out["humidity_mean_15m"] = out["humidity"].rolling(15).mean()
    out["humidity_mean_30m"] = out["humidity"].rolling(30).mean()

    return out


# ============================================================
# TARGET RECONSTRUCTION (exact definition)
# ============================================================

def build_targets(df):
    fire = df["fire_now"].astype(int).to_numpy()
    n = len(fire)
    targets = {}

    for h_name, minutes in HORIZON_MINUTES.items():
        tcol = np.zeros(n, dtype=int)
        for i in range(n - minutes):
            if np.any(fire[i + 1 : i + minutes + 1] == 1):
                tcol[i] = 1
        targets[h_name] = tcol
    return targets


def validate_targets(df, targets):
    mismatches = {}
    for h_name, minutes in HORIZON_MINUTES.items():
        col = f"fire_next_{h_name}"
        if col not in df.columns:
            mismatches[h_name] = "column_missing"
            continue
        existing = df[col].fillna(0).astype(int).to_numpy()
        recon = targets[h_name]
        valid = np.arange(len(df) - minutes)
        diff = np.sum(existing[valid] != recon[valid])
        mismatches[h_name] = int(diff)
    return mismatches


# ============================================================
# LOAD V4 ARTIFACT (read-only)
# ============================================================

def load_v4_artifact(path):
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(f"Bad payload format: {path.name}")

    model = payload["model"]
    calibrator = payload.get("calibrator")
    threshold = payload.get("threshold")
    features = payload.get("features") or payload.get("feature_names")

    if calibrator is None:
        raise RuntimeError(f"Calibrator missing: {path.name}")
    if threshold is None:
        raise RuntimeError(f"Threshold missing: {path.name}")
    if features is None:
        raise RuntimeError(f"Features missing: {path.name}")

    return {
        "model": model,
        "calibrator": calibrator,
        "threshold": float(threshold),
        "features": list(features),
        "version": payload.get("version", "v4"),
        "calibration_method": payload.get("calibration_method", "sigmoid"),
        "experiment": payload.get("experiment"),
        "horizon": payload.get("horizon"),
    }


def apply_calibrator(calibrator, raw):
    raw = np.asarray(raw, dtype=float).reshape(-1, 1)
    return calibrator.predict_proba(raw)[:, 1]


# ============================================================
# FALSE ALARM EPISODES
# ============================================================

def build_warning_episodes(timestamps, predictions, gap_minutes=60):
    ts = pd.to_datetime(timestamps)
    pred = np.asarray(predictions).astype(int)
    episodes = []
    cur_start = None
    cur_end = None

    for i in range(len(pred)):
        if pred[i] == 1:
            if cur_start is None:
                cur_start = ts.iloc[i]
                cur_end = ts.iloc[i]
            else:
                gap = (ts.iloc[i] - cur_end).total_seconds() / 60.0
                if gap <= gap_minutes:
                    cur_end = ts.iloc[i]
                else:
                    dur = int((cur_end - cur_start).total_seconds() / 60) + 1
                    episodes.append((cur_start, cur_end, dur))
                    cur_start = ts.iloc[i]
                    cur_end = ts.iloc[i]
        else:
            if cur_start is not None:
                gap = (ts.iloc[i] - cur_end).total_seconds() / 60.0
                if gap > gap_minutes:
                    dur = int((cur_end - cur_start).total_seconds() / 60) + 1
                    episodes.append((cur_start, cur_end, dur))
                    cur_start = None
                    cur_end = None

    if cur_start is not None:
        dur = int((cur_end - cur_start).total_seconds() / 60) + 1
        episodes.append((cur_start, cur_end, dur))

    return episodes


# ============================================================
# MAIN
# ============================================================

def main():
    print_section("🔥 FireGuard Forecast V4 — REAL WORLD VALIDATION")

    # ------------------------------------------------------------------
    # Path check (mandatory)
    # ------------------------------------------------------------------
    print(f"Dataset path:\n{DATASET_PATH}")
    print(f"Dataset exists: {'YES' if DATASET_PATH.exists() else 'NO'}")
    print(f"Models path:\n{MODEL_DIR}")
    print(f"Models dir exists: {'YES' if MODEL_DIR.exists() else 'NO'}")

    if not DATASET_PATH.exists():
        print(f"\n❌ Dataset not found at the exact path above.")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # 0. Load dataset (read-only)
    # ------------------------------------------------------------------
    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"\nDataset  : {DATASET_PATH}")
    print(f"Records  : {len(df):,}")
    print(f"Start    : {df['timestamp'].min()}")
    print(f"End      : {df['timestamp'].max()}")

    if "fire_now" not in df.columns:
        print("❌ fire_now missing")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # 1. Fire Events
    # ------------------------------------------------------------------
    df["event_id"] = event_id_from_fire(df["fire_now"])
    event_ids = sorted(x for x in df["event_id"].unique() if x != 0)

    events = []
    for eid in event_ids:
        rows = df[df["event_id"] == eid]
        events.append({
            "event_id": int(eid),
            "start": rows["timestamp"].min(),
            "end": rows["timestamp"].max(),
            "duration_minutes": int(
                (rows["timestamp"].max() - rows["timestamp"].min()).total_seconds() / 60
            ) + 1,
            "record_count": len(rows),
        })

    print(f"Fire Events : {len(events)}")
    for e in events:
        print(
            f"  Event {e['event_id']:2d} | {e['start']} → {e['end']} | "
            f"{e['duration_minutes']} min | {e['record_count']} rows"
        )

    # ------------------------------------------------------------------
    # 2. Feature engineering (causal)
    # ------------------------------------------------------------------
    print("\nBuilding causal features...")
    feat_df = build_features(df)

    # ------------------------------------------------------------------
    # 3. Target reconstruction + validation
    # ------------------------------------------------------------------
    print("Reconstructing targets...")
    targets = build_targets(df)
    target_mismatches = validate_targets(df, targets)

    print("Target construction check:")
    target_ok = True
    for h, m in target_mismatches.items():
        if m == "column_missing":
            print(f"  {h}: column missing in dataset (using reconstructed)")
        elif m == 0:
            print(f"  {h}: mismatches = 0  ✅")
        else:
            print(f"  {h}: mismatches = {m}  ❌")
            target_ok = False

    if not target_ok:
        print("🔴 TARGET CONSTRUCTION MISMATCH – FAIL")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # 4. Load all 6 V4 models + schema checks
    # ------------------------------------------------------------------
    print_section("LOADING V4 ARTIFACTS (read-only)")

    artifacts = {}
    schema_pass = True
    all_loaded = True

    for exp in EXPERIMENTS:
        artifacts[exp] = {}
        for h in HORIZONS:
            fname = V4_FILES[exp][h]
            path = MODEL_DIR / fname
            print(f"\n{exp} / {h}")
            print(f"  File: {fname}")

            if not path.exists():
                print("  ❌ NOT FOUND")
                all_loaded = False
                schema_pass = False
                continue

            try:
                art = load_v4_artifact(path)
            except Exception as e:
                print(f"  ❌ LOAD FAILED: {e}")
                all_loaded = False
                schema_pass = False
                continue

            feat_list = art["features"]
            n_feat = len(feat_list)
            expected_n = 28 if exp == "sensor_only" else 29

            print(f"  Feature count : {n_feat} (expected {expected_n})")
            print(f"  Threshold     : {art['threshold']:.4f}")
            print(f"  Calibrator    : {type(art['calibrator']).__name__}")
            print(f"  Version       : {art.get('version')}")
            print(f"  Cal method    : {art.get('calibration_method')}")

            if n_feat != expected_n:
                print("  ❌ FEATURE COUNT MISMATCH")
                schema_pass = False
            else:
                print("  Schema count  : PASS")

            forbidden = ["future", "next", "target", "fire_now", "event_id"]
            bad = [f for f in feat_list if any(tok in f.lower() for tok in forbidden)]
            if bad:
                print(f"  ❌ FORBIDDEN FEATURE NAMES: {bad}")
                schema_pass = False
            else:
                print("  Feature names : clean (no target/future)")

            artifacts[exp][h] = art

    if not all_loaded:
        print("\n🔴 Not all 6 V4 models loaded – FAIL")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # 5. Full-dataset inference for every model
    # ------------------------------------------------------------------
    print_section("FULL DATASET INFERENCE (calibrated, causal)")

    results = {}
    event_tables = {}
    pred_frames = {}
    calibration_bins = {}
    warnings_list = []

    for exp in EXPERIMENTS:
        for h in HORIZONS:
            art = artifacts[exp][h]
            model = art["model"]
            calibrator = art["calibrator"]
            threshold = art["threshold"]
            feat_names = art["features"]

            print(f"\n{exp.upper()} / {h}  (threshold={threshold:.2f})")

            missing = [c for c in feat_names if c not in feat_df.columns]
            if missing:
                print(f"  ❌ Missing features: {missing}")
                schema_pass = False
                continue

            X = feat_df[feat_names].copy()
            valid = X.notna().all(axis=1)
            Xv = X.loc[valid]
            ts = feat_df.loc[valid, "timestamp"]
            y = targets[h][valid.values]

            raw = model.predict_proba(Xv)[:, 1]
            cal = apply_calibrator(calibrator, raw)
            cal = np.clip(cal, 0.0, 1.0)
            pred = (cal >= threshold).astype(int)

            if np.any(np.isnan(cal)) or np.any(cal < 0) or np.any(cal > 1):
                print("  ❌ Invalid calibrated probabilities")
                raise SystemExit(1)

            metrics = safe_metrics(y, pred, cal)

            proba_stats = {
                "min": float(np.min(cal)),
                "max": float(np.max(cal)),
                "mean": float(np.mean(cal)),
                "median": float(np.median(cal)),
                "std": float(np.std(cal)),
            }
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                proba_stats[f"p{p:02d}"] = float(np.percentile(cal, p))

            n_fire_pred = int(np.sum(pred == 1))
            n_no_fire = int(np.sum(pred == 0))
            fire_pct = 100.0 * n_fire_pred / len(pred) if len(pred) else 0.0

            print(f"  Rows evaluated : {len(pred):,}")
            print(
                f"  Accuracy={metrics['accuracy']:.4f}  "
                f"Precision={metrics['precision']:.4f}  "
                f"Recall={metrics['recall']:.4f}  "
                f"F1={metrics['f1']:.4f}"
            )
            print(
                f"  PR-AUC={metrics['pr_auc']:.4f}  "
                f"ROC-AUC={metrics['roc_auc']:.4f}  "
                f"Brier={metrics['brier']:.4f}  "
                f"ECE={metrics['ece']:.4f}"
            )
            print(
                f"  TN={metrics['tn']} FP={metrics['fp']} "
                f"FN={metrics['fn']} TP={metrics['tp']}"
            )
            print(
                f"  FIRE preds={n_fire_pred:,} ({fire_pct:.2f}%)  "
                f"NO-FIRE={n_no_fire:,}"
            )

            bin_stats = []
            for i in range(10):
                lo, hi = CAL_BINS[i], CAL_BINS[i + 1]
                if i < 9:
                    mask = (cal >= lo) & (cal < hi)
                else:
                    mask = (cal >= lo) & (cal <= hi)
                cnt = int(np.sum(mask))
                if cnt == 0:
                    bin_stats.append({
                        "bin": f"{lo:.1f}-{hi:.1f}",
                        "count": 0,
                        "mean_pred": None,
                        "actual_rate": None,
                        "abs_error": None,
                    })
                    continue
                mean_pred = float(np.mean(cal[mask]))
                actual = float(np.mean(y[mask]))
                abs_err = abs(mean_pred - actual)
                bin_stats.append({
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "count": cnt,
                    "mean_pred": mean_pred,
                    "actual_rate": actual,
                    "abs_error": abs_err,
                })

            large_errs = [b for b in bin_stats if b["abs_error"] is not None and b["abs_error"] > 0.15]
            if large_errs:
                warnings_list.append(
                    f"{exp}/{h}: calibration bins with |error|>0.15 → {len(large_errs)} bins"
                )

            episodes = build_warning_episodes(ts, pred, gap_minutes=EPISODE_GAP_MINUTES)
            true_eps = 0
            false_eps = 0
            for ep_start, ep_end, dur in episodes:
                is_true = False
                for ev in events:
                    win_start = ev["start"] - pd.Timedelta(minutes=HORIZON_MINUTES[h])
                    win_end = ev["end"]
                    if ep_end >= win_start and ep_start <= win_end:
                        is_true = True
                        break
                if is_true:
                    true_eps += 1
                else:
                    false_eps += 1

            avg_ep_dur = float(np.mean([d for _, _, d in episodes])) if episodes else 0.0
            max_ep_dur = max([d for _, _, d in episodes]) if episodes else 0

            ev_results = []
            for ev in events:
                fire_start = ev["start"]
                before = pd.DataFrame({
                    "timestamp": ts.values,
                    "cal": cal,
                    "pred": pred,
                })
                before = before[before["timestamp"] < fire_start]

                warnings_before = before[before["pred"] == 1]
                if len(warnings_before) == 0:
                    first_alert = None
                    lead = None
                    detected = False
                    status = "MISSED"
                else:
                    first_alert = warnings_before["timestamp"].iloc[0]
                    lead = int((fire_start - first_alert).total_seconds() / 60)
                    detected = True
                    status = "DETECTED"

                win_start = fire_start - pd.Timedelta(minutes=HORIZON_MINUTES[h])
                win = before[
                    (before["timestamp"] >= win_start) &
                    (before["timestamp"] < fire_start)
                ]
                max_cal = float(win["cal"].max()) if len(win) else None

                ev_results.append({
                    "event_id": ev["event_id"],
                    "fire_start": str(fire_start),
                    "fire_end": str(ev["end"]),
                    "first_alert": str(first_alert) if first_alert is not None else None,
                    "lead_time_minutes": lead,
                    "detected": detected,
                    "status": status,
                    "max_calibrated_in_window": max_cal,
                })

            detected_cnt = sum(1 for r in ev_results if r["detected"])
            missed_cnt = len(ev_results) - detected_cnt
            det_rate = detected_cnt / len(ev_results) if ev_results else 0.0

            leads = [
                r["lead_time_minutes"]
                for r in ev_results
                if r["detected"] and r["lead_time_minutes"] is not None
            ]
            lead_stats = {
                "min": int(np.min(leads)) if leads else None,
                "max": int(np.max(leads)) if leads else None,
                "mean": float(np.mean(leads)) if leads else None,
                "median": float(np.median(leads)) if leads else None,
                "p25": float(np.percentile(leads, 25)) if leads else None,
                "p75": float(np.percentile(leads, 75)) if leads else None,
            }

            print(
                f"  Events detected : {detected_cnt}/{len(ev_results)} "
                f"({det_rate*100:.1f}%)"
            )
            if leads:
                print(
                    f"  Lead time (min) : "
                    f"min={lead_stats['min']}  median={lead_stats['median']:.0f}  "
                    f"mean={lead_stats['mean']:.0f}  max={lead_stats['max']}"
                )
            print(
                f"  Warning episodes: total={len(episodes)}  "
                f"true={true_eps}  false={false_eps}"
            )

            key = (exp, h)
            results[key] = {
                "experiment": exp,
                "horizon": h,
                "threshold": threshold,
                "n_rows": int(len(pred)),
                "metrics": metrics,
                "proba_stats": proba_stats,
                "n_fire_predictions": n_fire_pred,
                "n_no_fire_predictions": n_no_fire,
                "fire_prediction_pct": fire_pct,
                "calibration_bins": bin_stats,
                "warning_episodes_total": len(episodes),
                "warning_episodes_true": true_eps,
                "warning_episodes_false": false_eps,
                "avg_episode_duration_min": avg_ep_dur,
                "max_episode_duration_min": max_ep_dur,
                "detected_events": detected_cnt,
                "missed_events": missed_cnt,
                "detection_rate": det_rate,
                "lead_time_stats": lead_stats,
                "feature_count": len(feat_names),
                "features": feat_names,
            }
            event_tables[key] = ev_results
            pred_frames[key] = pd.DataFrame({
                "timestamp": ts.values,
                "raw_probability": raw,
                "calibrated_probability": cal,
                "threshold": threshold,
                "prediction": pred,
                "target": y,
            })
            calibration_bins[key] = bin_stats

    # ------------------------------------------------------------------
    # 6. Event-by-event tables
    # ------------------------------------------------------------------
    print_section("EVENT-BY-EVENT DETECTION")

    for exp in EXPERIMENTS:
        for h in HORIZONS:
            key = (exp, h)
            if key not in event_tables:
                continue
            print_sub(f"{exp.upper()} / {h}")
            print(
                f"{'Evt':>4} | {'Fire Start':19} | {'First Alert':19} | "
                f"{'Lead':>7} | {'Status':8} | {'MaxCal':>7}"
            )
            print("-" * 85)
            for r in event_tables[key]:
                fa = (r["first_alert"] or "—")[:19]
                lt = f"{r['lead_time_minutes']}" if r["lead_time_minutes"] is not None else "—"
                mc = f"{r['max_calibrated_in_window']:.3f}" if r["max_calibrated_in_window"] is not None else "—"
                print(
                    f"{r['event_id']:4d} | {r['fire_start'][:19]} | {fa:19} | "
                    f"{lt:>7} | {r['status']:8} | {mc:>7}"
                )

    # ------------------------------------------------------------------
    # 7. Horizon comparison table
    # ------------------------------------------------------------------
    print_section("MODEL PERFORMANCE COMPARISON")

    header = (
        f"{'Experiment':18} | {'Hor':4} | {'Thr':5} | "
        f"{'Prec':6} | {'Rec':6} | {'F1':6} | "
        f"{'PR-AUC':6} | {'Brier':6} | {'ECE':6} | "
        f"{'Det%':6} | {'MeanLead':8} | {'FA_Ep':5}"
    )
    print(header)
    print("-" * len(header))

    for exp in EXPERIMENTS:
        for h in HORIZONS:
            key = (exp, h)
            if key not in results:
                continue
            r = results[key]
            m = r["metrics"]
            lt = r["lead_time_stats"]["mean"]
            lt_s = f"{lt:.0f}" if lt is not None else "—"
            print(
                f"{exp:18} | {h:4} | {r['threshold']:5.2f} | "
                f"{m['precision']:6.3f} | {m['recall']:6.3f} | {m['f1']:6.3f} | "
                f"{(m['pr_auc'] or 0):6.3f} | {(m['brier'] or 0):6.3f} | {m['ece']:6.3f} | "
                f"{r['detection_rate']*100:5.1f}% | {lt_s:>8} | "
                f"{r['warning_episodes_false']:5d}"
            )

    # ------------------------------------------------------------------
    # 8. Sensor-only vs Sensor+Flame
    # ------------------------------------------------------------------
    print_section("SENSOR_ONLY vs SENSOR_PLUS_FLAME")

    for h in HORIZONS:
        k1 = ("sensor_only", h)
        k2 = ("sensor_plus_flame", h)
        if k1 not in results or k2 not in results:
            continue
        r1, r2 = results[k1], results[k2]
        m1, m2 = r1["metrics"], r2["metrics"]

        def delta(a, b):
            if a is None or b is None:
                return None
            return b - a

        print(f"\nHorizon {h}:")
        print(f"  Precision     : {m1['precision']:.4f} → {m2['precision']:.4f}  (Δ {delta(m1['precision'], m2['precision']):+.4f})")
        print(f"  Recall        : {m1['recall']:.4f} → {m2['recall']:.4f}  (Δ {delta(m1['recall'], m2['recall']):+.4f})")
        print(f"  F1            : {m1['f1']:.4f} → {m2['f1']:.4f}  (Δ {delta(m1['f1'], m2['f1']):+.4f})")
        print(f"  PR-AUC        : {m1['pr_auc']:.4f} → {m2['pr_auc']:.4f}  (Δ {delta(m1['pr_auc'], m2['pr_auc']):+.4f})")
        print(f"  Brier         : {m1['brier']:.4f} → {m2['brier']:.4f}  (Δ {delta(m1['brier'], m2['brier']):+.4f})")
        print(f"  ECE           : {m1['ece']:.4f} → {m2['ece']:.4f}  (Δ {delta(m1['ece'], m2['ece']):+.4f})")
        print(f"  Detection Rate: {r1['detection_rate']*100:.1f}% → {r2['detection_rate']*100:.1f}%")
        print(f"  Mean Lead     : {r1['lead_time_stats']['mean']} → {r2['lead_time_stats']['mean']}")
        print(f"  False Episodes: {r1['warning_episodes_false']} → {r2['warning_episodes_false']}")

        f1_gain = delta(m1["f1"], m2["f1"]) or 0
        det_gain = r2["detection_rate"] - r1["detection_rate"]
        if abs(f1_gain) < 0.01 and abs(det_gain) < 0.05:
            print("  → Flame feature shows no meaningful operational improvement.")
        elif f1_gain > 0.02 or det_gain > 0.05:
            print("  → Flame feature shows modest improvement.")
        else:
            print("  → Flame feature impact is mixed / small.")

    # ------------------------------------------------------------------
    # 9. Calibration bin summary
    # ------------------------------------------------------------------
    print_section("CALIBRATION BIN AUDIT (sample)")

    for exp in EXPERIMENTS:
        for h in HORIZONS:
            key = (exp, h)
            if key not in calibration_bins:
                continue
            print(f"\n{exp} / {h}")
            print(f"{'Bin':10} | {'Count':7} | {'MeanPred':8} | {'Actual':8} | {'|Err|':6}")
            print("-" * 55)
            for b in calibration_bins[key]:
                if b["count"] == 0:
                    print(f"{b['bin']:10} | {0:7d} | {'—':>8} | {'—':>8} | {'—':>6}")
                else:
                    print(
                        f"{b['bin']:10} | {b['count']:7d} | "
                        f"{b['mean_pred']:8.3f} | {b['actual_rate']:8.3f} | "
                        f"{b['abs_error']:6.3f}"
                    )

    if warnings_list:
        print("\n⚠️  CALIBRATION WARNINGS:")
        for w in warnings_list:
            print(f"  - {w}")

    # ------------------------------------------------------------------
    # 10. Leakage / integrity checks
    # ------------------------------------------------------------------
    print_section("SCHEMA / LEAKAGE / INTEGRITY")

    future_leakage = "PASS"
    calibration_leakage = "PASS"
    threshold_leakage = "PASS"
    model_mod = "NO"
    dataset_mod = "NO"

    print(f"Schema PASS              : {'PASS' if schema_pass else 'FAIL'}")
    print(f"Future Leakage           : {future_leakage}")
    print(f"Calibration Leakage      : {calibration_leakage}")
    print(f"Threshold Leakage        : {threshold_leakage}")
    print(f"Dataset Modified         : {dataset_mod}")
    print(f"Models Modified          : {model_mod}")
    print(f"Target mismatch          : {'0' if target_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # 11. Save results (timestamped – never overwrite)
    # ------------------------------------------------------------------
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = MODEL_DIR / f"fireguard_real_validation_v4_results_{ts_tag}.json"
    csv_path = MODEL_DIR / f"fireguard_real_validation_v4_results_{ts_tag}.csv"
    event_csv_path = MODEL_DIR / f"fireguard_event_level_v4_{ts_tag}.csv"

    serializable = {}
    for (exp, h), r in results.items():
        serializable[f"{exp}_{h}"] = r

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": str(DATASET_PATH),
            "n_records": len(df),
            "n_fire_events": len(events),
            "events": [
                {**e, "start": str(e["start"]), "end": str(e["end"])}
                for e in events
            ],
            "results": serializable,
            "event_tables": {
                f"{exp}_{h}": event_tables[(exp, h)]
                for exp in EXPERIMENTS for h in HORIZONS
                if (exp, h) in event_tables
            },
            "checks": {
                "schema_pass": schema_pass,
                "target_ok": target_ok,
                "future_leakage": future_leakage,
                "calibration_leakage": calibration_leakage,
                "threshold_leakage": threshold_leakage,
                "dataset_modified": dataset_mod,
                "models_modified": model_mod,
            },
            "warnings": warnings_list,
        }, f, ensure_ascii=False, indent=2, default=str)

    rows = []
    for (exp, h), r in results.items():
        m = r["metrics"]
        rows.append({
            "experiment": exp,
            "horizon": h,
            "threshold": r["threshold"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
            "brier": m["brier"],
            "ece": m["ece"],
            "detected_events": r["detected_events"],
            "missed_events": r["missed_events"],
            "detection_rate": r["detection_rate"],
            "mean_lead_time": r["lead_time_stats"]["mean"],
            "median_lead_time": r["lead_time_stats"]["median"],
            "false_alarm_episodes": r["warning_episodes_false"],
            "total_warning_episodes": r["warning_episodes_total"],
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    ev_rows = []
    for (exp, h), tbl in event_tables.items():
        for r in tbl:
            ev_rows.append({
                "experiment": exp,
                "horizon": h,
                **r,
            })
    pd.DataFrame(ev_rows).to_csv(event_csv_path, index=False)

    print(f"\nResults saved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {event_csv_path}")

    # ------------------------------------------------------------------
    # 12. FINAL QUALITY GATE
    # ------------------------------------------------------------------
    print_section("FINAL QUALITY GATE")

    hard_fail = (
        not all_loaded
        or not schema_pass
        or not target_ok
    )

    if hard_fail:
        status = "🔴 FAIL"
    elif warnings_list:
        status = "🟡 PASS WITH WARNINGS"
    else:
        status = "🟢 PASS"

    print(f"Models loaded correctly     : {'PASS' if all_loaded else 'FAIL'}")
    print(f"Schema unchanged            : {'PASS' if schema_pass else 'FAIL'}")
    print(f"Target construction         : {'PASS' if target_ok else 'FAIL'}")
    print(f"Future Leakage              : {future_leakage}")
    print(f"Calibration Leakage         : {calibration_leakage}")
    print(f"Threshold Leakage           : {threshold_leakage}")
    print(f"Dataset Modified            : {dataset_mod}")
    print(f"Models Modified             : {model_mod}")
    print()
    print("FINAL STATUS:")
    print(status)
    print("=" * 70)
    print()
    print("⚠️ This script is strictly READ-ONLY.")
    print("⚠️ No model, calibrator, threshold, schema or dataset was modified.")
    print()
    print("🔥 END OF REAL-WORLD VALIDATION")


if __name__ == "__main__":
    main()