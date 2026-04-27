# House Prices - Advanced Regression Techniques

This repository is a separate Kaggle project for **House Prices - Advanced Regression Techniques**.

The goal is to build a clean, repeatable pipeline:
- data loading and validation
- feature engineering
- leakage-safe cross-validation
- baseline models and model comparison
- hypothesis checks for feature ideas
- submission file generation

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
- `artifacts/submissions/submission_*.csv`

## Notes

The pipeline trains on `log1p(SalePrice)` and uses RMSLE-oriented evaluation. That is the standard shape for this competition, but the code is structured so the feature set and model list can be changed without rewriting the whole project.