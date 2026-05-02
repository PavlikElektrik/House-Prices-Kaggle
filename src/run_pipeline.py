from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.house_prices.config import load_config
from src.house_prices.data import load_data
from src.house_prices.features import build_features, split_target
from src.house_prices.models import make_models, make_preprocessor
from src.house_prices.training import (
    evaluate_cv,
    get_oof_predictions,
    save_submission,
    tune_blend_weights,
    weighted_prediction,
    compute_oof_metrics,
    save_oof_predictions,
    save_test_predictions,
    save_metrics,
)
from sklearn.metrics import mean_squared_error


def main() -> None:
    """Запустить полный эксперимент House Prices от начала до конца."""
    parser = argparse.ArgumentParser(description="House Prices baseline pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    artifact_dir = Path(cfg["paths"]["artifact_dir"])

    # Используем общие хелперы, чтобы директории артефактов и загрузка CSV
    # вели себя одинаково во всех точках входа.
    from src.common import make_artifact_dirs, load_train_test

    dirs = make_artifact_dirs(artifact_dir)
    report_dir = dirs["reports"]
    sub_dir = dirs["submissions"]
    pred_dir = dirs["predictions"]
    metrics_dir = dirs["metrics"]
    figures_dir = dirs["figures"]
    models_dir = dirs["models"]

    train_df, test_df = load_train_test(data_dir)
    # Сохраняем Id до генерации признаков, потому что шаг признаков удаляет Id.
    train_ids = train_df["Id"].copy()
    test_ids = test_df["Id"].copy()
    train_df = build_features(train_df)
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

    # Сохраняем артефакты, чтобы можно было разобрать запуск без повторного обучения.
    oof_metrics_df = compute_oof_metrics(oof_pred, y_train, model_order)
    oof_metrics_df.to_csv(metrics_dir / "oof_model_rmse.csv", index=False)
    save_oof_predictions(oof_pred, model_order, train_ids, pred_dir, prefix="oof")

    test_pred = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        test_pred[name] = model.predict(X_test)

    # Сохраняем отдельный CSV для каждой модели, чтобы бленды и будущие
    # эксперименты было легко воспроизвести.
    save_test_predictions(test_pred, test_ids, pred_dir, prefix="test")

    blend_map = {name: weight for name, weight in zip(model_order, blend_weights)}
    blend_pred = weighted_prediction(test_pred, blend_map)

    sub_path = save_submission(blend_pred, test_ids, sub_dir, cfg["output"]["submission_name"].replace(".csv", ""))

    import numpy as _np
    blend_oof = oof_pred @ _np.array(blend_weights)
    blend_rmse = float(_np.sqrt(mean_squared_error(y_train.values, blend_oof)))

    summary = {
        "config_path": str(args.config),
        "model_order": model_order,
        "blend_weights": blend_map,
        "cv_table": cv_df.to_dict(orient="records"),
        "oof_metrics": oof_metrics_df.to_dict(orient="records"),
        "blend_oof_rmse": blend_rmse,
        "submission": str(sub_path),
    }
    with open(report_dir / "feature_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    save_metrics({"blend_oof_rmse": blend_rmse}, metrics_dir, filename="blend")

    print(cv_df.to_string(index=False))
    print(f"Submission saved to: {sub_path}")


if __name__ == "__main__":
    main()