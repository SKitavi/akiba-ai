"""End-to-end training pipeline for AkibaAI MVP.

Runs the full pipeline:
  1. Generate (or load) synthetic data
  2. Build behavioral feature table
  3. Train/test split (stratified 80/20)
  4. Train XGBoost model + 5-fold CV
  5. Evaluate against majority-class baseline
  6. Persist model artifact + evaluation report

Usage:
    python -m src.model.run_training
    python -m src.model.run_training --num-applicants 500 --out models/xgb_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_gen.generate_synthetic_data import (  # noqa: E402
    generate_dataset,
    calibrate_default_labels,
)
from src.features.build_features import (  # noqa: E402
    build_feature_table,
    FEATURE_COLUMNS,
)
from src.model.train import train_model  # noqa: E402
from src.model.loader import load_model_bundle  # noqa: E402
from src.eval.metrics import generate_classification_report  # noqa: E402

from sklearn.model_selection import train_test_split  # noqa: E402


def run(num_applicants: int = 250, model_out: Path | None = None) -> dict:
    """Execute the full training pipeline and return the evaluation report."""

    if model_out is None:
        model_out = PROJECT_ROOT / "models" / "xgb_v1.json"

    print("=" * 60)
    print("  AKIBA AI — CREDIT RISK MODEL TRAINING PIPELINE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Synthetic data
    # ------------------------------------------------------------------
    print(f"\n[1/4] Generating synthetic dataset ({num_applicants} applicants)...")
    df_app, df_sms = generate_dataset(num_applicants)
    df_labeled = calibrate_default_labels(df_app, df_sms)
    print(f"      Applicants: {len(df_labeled)} | SMS logs: {len(df_sms)}")
    print(f"      Default rate: {df_labeled['default_label'].mean():.1%}")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    print("\n[2/4] Building behavioral feature table...")
    feat_df = build_feature_table(df_sms, applicants_df=df_labeled)
    full_df = feat_df.merge(
        df_labeled[["applicant_id", "default_label"]], on="applicant_id"
    )
    print(
        f"      Feature table: {full_df.shape[0]} rows × {len(FEATURE_COLUMNS)} features"
    )
    nan_count = full_df[FEATURE_COLUMNS].isnull().sum().sum()
    print(f"      NaN values: {nan_count}")

    # ------------------------------------------------------------------
    # 3. Train/test split
    # ------------------------------------------------------------------
    X = full_df[FEATURE_COLUMNS].values
    y = full_df["default_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(
        f"\n[3/4] Split — Train: {len(y_train)} ({y_train.mean():.1%} pos) "
        f"| Test: {len(y_test)} ({y_test.mean():.1%} pos)"
    )

    # ------------------------------------------------------------------
    # 4. Train (full training set) + evaluate on held-out test set
    # ------------------------------------------------------------------
    print("\n[4/4] Training XGBoost + evaluating...")

    # Build a features_df from the training split for train_model()
    import pandas as pd

    train_df = pd.DataFrame(X_train, columns=FEATURE_COLUMNS)
    train_df["default_label"] = y_train

    metadata = train_model(train_df, model_out)

    # Score the held-out test set with the saved model
    model = load_model_bundle(model_out).model
    y_proba = model.predict_proba(X_test)[:, 1]

    report = generate_classification_report(y_test.tolist(), y_proba.tolist())

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS (hold-out test set)")
    print("=" * 60)
    print(
        f"  Samples         : {report['n_samples']} "
        f"({report['positive_rate']:.1%} positive)"
    )
    print(f"  Baseline acc    : {report['baseline_accuracy']:.4f}  (majority class)")
    print(
        f"  Model accuracy  : {report['accuracy']:.4f}  "
        f"({'✓ beats' if report['beats_baseline'] else '✗ below'} baseline)"
    )
    print(f"  AUC-ROC         : {report['auc_roc']:.4f}")
    print(f"  Precision       : {report['precision']:.4f}")
    print(f"  Recall          : {report['recall']:.4f}")
    print(f"  F1              : {report['f1']:.4f}")
    print(
        f"  TP/FP/TN/FN     : {report['tp']}/{report['fp']}/{report['tn']}/{report['fn']}"
    )
    print(
        f"\n  CV AUC (5-fold) : {metadata['cv_auc_mean']:.4f} ± {metadata['cv_auc_std']:.4f}"
    )
    print(f"  Model artifact  : {model_out}")
    print("=" * 60)

    # Persist evaluation report alongside model
    eval_out = model_out.with_suffix(".eval.json")
    full_report = {**metadata, "holdout_eval": report}
    eval_out.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"  Eval report     : {eval_out}")

    return full_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Akiba AI training pipeline")
    parser.add_argument("--num-applicants", type=int, default=250)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run(num_applicants=args.num_applicants, model_out=args.out)
