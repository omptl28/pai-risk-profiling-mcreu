# PAI-2 Risk Profiling (MCREU)

Interpretable machine learning on the Personality Assessment Inventory (PAI-2) for early detection of clinical risk. This notebook produced the analysis behind the poster "Machine Learning-Based Risk Profile Identification in Mental Health Disorders," presented at the 2026 Penn State Undergraduate Exhibition as part of the Multi-Campus Research Experience for Undergraduates (MCREU).

> **This is the earlier MCREU version of the analysis.** The machine-learning research questions were later revised and strengthened. The current versions of RQ5 to RQ7, with a fully leakage-safe evaluation, live in the `updated-analysis` repository. This repository is kept for provenance and to document the poster as presented.

## Research questions

The same interpretable-ML approach (tree ensembles plus SHAP) is applied to three outcomes:

- **RQ1, Suicide risk.** Flagging elevated suicide risk from the personality profile, with the ideation and self-harm scales removed from the features to avoid a near-tautological target.
- **RQ2, Treatment disengagement.** Classifying clinically significant treatment rejection, and checking whether theory-driven regression and machine learning agree on the traits behind it.
- **RQ3, Under-detection of personality pathology.** Comparing elevated PD-relevant pathology on the PAI against recorded personality-disorder diagnoses to characterize a detection gap.

## General methodology

- Profiles are first validity-screened using standard PAI cutoffs.
- Cross-validated AUC is estimated honestly, with SMOTE resampling refit inside each fold via an imbalanced-learn pipeline so the validation rows are never resampled.
- For the reported operating point, a model is trained on a resampled training split and evaluated on a held-out test split, with the decision threshold tuned toward a recall target.
- SHAP beeswarm and mean absolute SHAP summaries provide interpretability.

## Notebooks

```
notebooks/
  mcreu_pai_analysis.ipynb   # RQ1 to RQ3 on a shared interpretable-ML pipeline
data/                        # you supply the dataset here (not included)
```

## Limitations

- This analysis predates the revisions in `updated-analysis` and should be read as the poster-era version.
- The cross-validated AUC is leakage-safe, but the thresholded operating-point metrics select the decision threshold on the held-out test set and rely on a single train/test split. Those thresholded numbers are therefore optimistic and are not directly comparable to the pooled out-of-fold results in `updated-analysis`.
- Where an external criterion is not available, an outcome falls back to a within-instrument proxy (for example an elevated ideation T-score), which is a proxy rather than an independent clinical label.
- Several diagnosis targets are rare, which limits what can be learned for those outcomes.
- Results come from a single sample and a single instrument and have not been externally validated. SHAP describes association, not causation.

## Data availability

The dataset is not included in this repository. The PAI-2 is a proprietary instrument and the data is available only under the relevant data-use terms. Notebook paths point to a local `data/` folder you supply.

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, SHAP, matplotlib, seaborn.

## Credit

Undergraduate research at Penn State Harrisburg, MCREU program.
