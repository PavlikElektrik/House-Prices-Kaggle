from __future__ import annotations

import argparse
import json
from pathlib import Path

from house_prices.config import load_config
from house_prices.data import load_data
from house_prices.features import build_features, split_target
from house_prices.models import make_models, make_preprocessor
from house_prices.training import evaluate_cv, get_oof_predictions, save_submission, tune_blend_weights, weighted_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="House Prices baseline pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    artifact_dir = Path(cfg["paths"]["artifact_dir"])
    report_dir = artifact_dir / "reports"
    sub_dir = artifact_dir / "submissions"
    report_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_data(data_dir)
    train_df = build_features(train_df)
    test_ids = test_df["Id"].copy()
    test_df = build_features(test_df)

    X_train, y_train = split_target(train_df)
    X_test = test_df.copy()

    preprocessor = make_preprocessor(X_train)
    models = make_models(cfg["models"], preprocessor)

    cv_df = evaluate_cv(
        X_train,
        y_train,
        models,
        n_splits=int(cfg["experiment"]["cv_splits"]),
        seed=int(cfg["experiment"]["seed"]),
    )
    cv_df.to_csv(report_dir / "cv_scores.csv", index=False)

    oof_pred, model_order = get_oof_predictions(
        X_train,
        y_train,
        models,
        n_splits=int(cfg["experiment"]["cv_splits"]),
        seed=int(cfg["experiment"]["seed"]),
    )
    blend_weights = tune_blend_weights(oof_pred, y_train, n_trials=25)

    test_pred = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        test_pred[name] = model.predict(X_test)

    blend_map = {name: weight for name, weight in zip(model_order, blend_weights)}
    blend_pred = weighted_prediction(test_pred, blend_map)

    sub_path = save_submission(blend_pred, test_ids, sub_dir, cfg["output"]["submission_name"].replace(".csv", ""))

    summary = {
        "config_path": str(args.config),
        "model_order": model_order,
        "blend_weights": blend_map,
        "cv_table": cv_df.to_dict(orient="records"),
        "submission": str(sub_path),
    }
    with open(report_dir / "feature_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(cv_df.to_string(index=False))
    print(f"Submission saved to: {sub_path}")


if __name__ == "__main__":
    main()