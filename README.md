# House Prices - Advanced Regression Techniques

This repository is a separate Kaggle project for **House Prices - Advanced Regression Techniques**.

The goal is to build a clean, repeatable pipeline:
- data loading and validation
- feature engineering
- leakage-safe cross-validation
- baseline models and model comparison
- hypothesis checks for feature ideas
- submission file generation

## What this repo demonstrates

This project is written to show the whole Kaggle workflow, not just a leaderboard result.
The emphasis is on:

- understanding the competition statement first
- choosing a baseline that is simple, explainable, and hard to break
- keeping the training loop leakage-safe
- saving artifacts so every run can be reviewed later
- documenting why a model and architecture were chosen

## Why this repo exists

I want a strong baseline that is **normal and reproducible**, not a copied leaderboard solution. The idea is to start from a solid public Kaggle-style workflow, then test hypotheses step by step:
- which house-quality features matter most
- whether log-transforming the target helps
- which categorical encodings are stable
- whether a small blend beats a single model

## Project layout

- `data/` - place `train.csv` and `test.csv` here
- `configs/` - YAML configs for experiments
- `src/house_prices/` - reusable package code
- `artifacts/` - reports, metrics, and submission files

## How to read the project

Start with [docs/kaggle_playbook.md](docs/kaggle_playbook.md) to see the competition workflow that applies to House Prices and can be reused for Titanic.

Then read [docs/house_prices_rationale.md](docs/house_prices_rationale.md) for the competition-specific reasoning.

After that, inspect `src/run_pipeline.py` to connect the narrative to the code.

Then inspect the core modules in this order:

- `src/house_prices/data.py` - dataset loading
- `src/house_prices/features.py` - feature engineering and target split
- `src/house_prices/models.py` - preprocessing and model zoo
- `src/house_prices/training.py` - CV, OOF, blending, and artifact saving

If you want the experiment outputs, look in `artifacts/` after a run.

## 3. House Prices: постановка задачи и бейзлайн

| Вопрос | Ответ |
|---|---|
| Что предсказывается? | `SalePrice` |
| Тип задачи | Регрессия |
| Метрика | Ошибка в лог-пространстве, поэтому важны относительные ошибки |
| Трансформация таргета | `log1p(SalePrice)` (это математическое преобразование: берём натуральный логарифм от `1 + цена`, чтобы уменьшить перекос очень дорогих домов и сделать обучение стабильнее) |
| Бейзлайн | Ridge, Lasso, ExtraTrees, CatBoost |

## 4. House Prices: метрики

| Модель | CV RMSE (log), mean | CV RMSE (log), std |
|---|---:|---:|
| CatBoost | 0.121959 | 0.017565 |
| ExtraTrees | 0.138385 | 0.014425 |
| Lasso | 0.143748 | 0.040411 |
| Ridge | 0.144292 | 0.039145 |

| Дополнительно | Значение |
|---|---:|
| OOF RMSE (log), CatBoost | 0.123218 |
| OOF RMSE (log), Blend | 0.124405 |

## Install

```bash
pip install -r requirements.txt
```

## Run baseline

```bash
python src/run_pipeline.py --config configs/default.yaml
```

## Expected outputs

- `artifacts/reports/cv_scores.csv`
- `artifacts/reports/feature_summary.json`
- `artifacts/metrics/*.json`
- `artifacts/predictions/*.csv`
- `artifacts/submissions/submission_*.csv`

## Notes

The pipeline trains on `log1p(SalePrice)` (логарифмируем целевую переменную через формулу `ln(1 + x)`, чтобы "сжать" выбросы по очень дорогим домам) and uses RMSLE-oriented evaluation. That is the standard shape for this competition, but the code is structured so the feature set and model list can be changed without rewriting the whole project.