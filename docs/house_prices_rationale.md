# House Prices Rationale

This document explains the competition-specific choices in the House Prices project.

## Problem statement

The task is to predict `SalePrice` from structured housing data.
The target is continuous, so this is a regression problem.
The official metric is based on logarithmic error, which makes large relative mistakes more important than small absolute ones.

## Why the target is logged

House prices are usually right-skewed. A log transform makes the target closer to a normal-like distribution and reduces the influence of very expensive outliers.

That is why the pipeline trains on `log1p(SalePrice)` and converts back only for submission.

## Why the baseline is Ridge, Lasso, ExtraTrees, and optional CatBoost

| Model | Why it is included |
|---|---|
| Ridge | Strong linear baseline for correlated tabular features |
| Lasso | Helps when some features are redundant or noisy |
| ExtraTrees | Captures nonlinear interactions without heavy tuning |
| CatBoost | Optional stronger tree baseline when the dependency is available |

This set is intentionally mixed.
Linear models tell me whether the feature engineering is useful.
Tree models tell me whether there are nonlinear interactions I am missing.

## Why the preprocessing looks like this

| Choice | Reason |
|---|---|
| Median imputation for numeric columns | Stable, simple, and resistant to outliers |
| Most-frequent imputation for categoricals | Preserves common category values |
| One-hot encoding | Works well for a medium-sized tabular competition |
| Scaling numeric features | Helps Ridge and Lasso behave well |
| `handle_unknown="ignore"` | Prevents failures when test data has unseen categories |

## Why the feature engineering is simple

The project keeps a small number of high-signal transformations:

| Feature group | Examples |
|---|---|
| Age features | `HouseAge`, `RemodAge` |
| Area aggregates | `TotalSF`, `PorchSF`, `TotalBath` |
| Quality flags | `HasPool`, `HasGarage`, `HasBasement`, `IsRemodeled` |

The point is not to create hundreds of engineered features.
The point is to create a compact, explainable baseline that is hard to overfit and easy to improve later.

## Why the architecture is split into modules

| Module | Responsibility |
|---|---|
| `data.py` | Read raw inputs |
| `features.py` | Derive model-ready columns |
| `models.py` | Build preprocessing and candidate estimators |
| `training.py` | Cross-validation, OOF predictions, blending, artifact saving |
| `run_pipeline.py` | Orchestrate one experiment |

This split helps in review because each design choice is visible in one place instead of being mixed into a single script.

## What changed from a dry baseline to a project worth reviewing

| Added element | Why it matters |
|---|---|
| `artifacts/reports` | Makes every run auditable |
| `artifacts/predictions` | Lets me inspect OOF and test predictions later |
| `artifacts/metrics` | Stores score history in a machine-readable format |
| `artifacts/submissions` | Keeps all Kaggle submissions in one place |
| README navigation | Helps another person understand the repo quickly |

## How to present this in a review

If I had to explain the project to a reviewer in one minute, I would say:

"I started from the competition metric and target distribution, built a leakage-safe baseline with a log-target regression setup, compared linear and tree models under the same preprocessing, then saved CV, OOF, predictions, and metrics so every decision can be traced. The architecture is deliberately modular so feature work, model work, and evaluation are separated."