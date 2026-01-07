# Loan Default: When Do Metrics Disagree?

## Question
How do common evaluation metrics disagree as model complexity increases in loan default prediction?

## Approach
Models:
- Logistic regression (baseline)
- Random forest
- Gradient boosting

Tuning objective:
- Tune for ROC-AUC, then evaluate all metrics.

## Repo structure
- `notebooks/`: analysis notebooks in sequence
- `R/`: reusable functions (data prep, modeling, metrics, viz)
- `figures/`: exported figures used in README / writeup
