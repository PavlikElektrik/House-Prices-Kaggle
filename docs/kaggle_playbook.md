# Kaggle Playbook

This is the workflow I use for both House Prices and Titanic.
The idea is to make the solution understandable for me, for a reviewer, and for future iterations.

## 1. Start with the problem statement

| Question | What to answer |
|---|---|
| What is being predicted? | Target definition and format of submission |
| What metric is used? | How model quality is measured |
| What can leak? | Which columns, splits, or transformations can accidentally use future information |
| What is the data shape? | Tabular, mixed types, class imbalance, missingness |

## 2. Keep a working notebook or markdown log

| Section | Purpose |
|---|---|
| To-do | What I need to test next |
| Notes | Facts about the data and competition |
| Questions | Open hypotheses that still need checking |
| Solution | Only the parts that improved the score |
| Annotations | Why a decision was made |
| Final ensemble | Which models were kept and how they were combined |
| Processed ideas | What was tried and rejected |

## 3. Choose the baseline by the competition type

| Competition type | Good first baseline | Why it is a good baseline |
|---|---|---|
| House Prices | Ridge/Lasso + robust preprocessing | Tabular regression, many correlated features, easy to explain, stable under CV |
| Titanic | Logistic Regression or Random Forest + basic preprocessing | Binary classification, small dataset, baseline is fast and easy to interpret |

The baseline should not be the fanciest model. It should be the model that answers the most questions with the least ambiguity.

## 4. Build the architecture around explanation, not just score

| Layer | What it does | Why it exists |
|---|---|---|
| `data.py` | Load raw CSV files | Keeps I/O separate from modeling |
| `features.py` | Build features and split target | Makes feature logic testable |
| `models.py` | Preprocessing and model zoo | Central place for model choice |
| `training.py` | CV, OOF, blend, export | Makes evaluation repeatable |
| `run_pipeline.py` | One end-to-end entry point | Gives a reviewer the shortest path through the project |

## 5. Decide when a model is good enough

I keep a model only if it satisfies all three:

| Check | Meaning |
|---|---|
| Score is better | It improves the metric |
| Behavior is stable | It does not depend on one lucky split |
| Explanation is clean | I can explain why it helps |

## 6. How to adapt this playbook to Titanic

| Step | Titanic version |
|---|---|
| Problem | Predict survival probability or class |
| Baseline | Logistic Regression with simple imputing and encoding |
| Features | Title from name, family size, cabin indicator, ticket group |
| Validation | Stratified CV |
| Interpretation | Focus on class imbalance, missing data, and leakage risk |

## 7. What a reviewer should see

The reviewer should be able to answer three questions from the repo alone:

1. What is the competition asking?
2. Why was this baseline chosen?
3. Why does the architecture make the experiment easier to trust and extend?