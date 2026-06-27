# Credit Risk PD Model Validation and Stress Testing

This project simulates the workflow of a quantitative model validation analyst for a credit risk probability of default (PD) model.

The goal is not only to train a classifier. The project is structured as an independent validation exercise covering data quality, methodology review, model performance testing, benchmarking, stress testing, expected loss mapping, and validation documentation.

## Why This Project Fits Quantitative Risk / Model Risk Management

- **Credit risk relevance:** builds and validates borrower-level PD models.
- **Quantitative analytics:** uses logistic regression and XGBoost challenger modeling.
- **Model validation mindset:** tests discrimination, calibration, stability, cost-sensitive thresholding, and model limitations.
- **Stress testing:** evaluates base, adverse, and severe scenarios.
- **Basel-style risk parameters:** maps PD to expected loss through `EL = PD x LGD x EAD`.
- **MRM documentation:** produces a validation report inspired by SR 11-7 concepts such as conceptual soundness, outcomes analysis, benchmarking, and limitation review.

## Dataset

Dataset: UCI Statlog German Credit Data

The original target is:

- `1`: good credit
- `2`: bad credit

This project maps bad credit to `bad = 1`, so model probability output can be interpreted as PD.

Reference:

- UCI Statlog German Credit Data: https://archive.ics.uci.edu/dataset/144/statlog%2Bgerman%2Bcredit%2Bdata
- Federal Reserve SR 11-7 Model Risk Management Guidance: https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- Basel II Comprehensive Version, BIS: https://www.bis.org/publ/bcbs128.pdf

## Project Structure

```text
credit_risk_pd_validation/
  data/
    raw/
      german.data
  outputs/
    figures/
    credit_risk_pd_model_validation_report.pdf
    model_validation_metrics.csv
    stress_testing_results.csv
    feature_importance.csv
    metrics_summary.json
  src/
    run_credit_risk_validation.py
  README.md
  requirements.txt
```

## Methodology

Two models are compared:

- **Logistic Regression:** interpretable baseline, closer to traditional credit scoring and easier to challenge.
- **XGBoost:** nonlinear challenger model used to test whether interactions and nonlinearities improve performance.

Validation tests include:

- AUC and ROC curve
- KS statistic
- Brier Score
- Calibration curve
- Cross-validation AUC stability
- Cost-sensitive threshold selection
- Confusion matrix review
- Feature importance review
- Scenario stress testing
- Expected loss estimation

## Stress Testing Design

The project creates three scenarios:

- **Base:** observed portfolio.
- **Adverse:** partial borrower deterioration, higher credit amount, longer duration, and riskier categorical states.
- **Severe:** stronger version of the adverse scenario.

For each scenario, the project estimates:

- mean portfolio PD
- portfolio EAD
- expected loss
- PD uplift vs base
- expected loss uplift vs base

## Run

```bash
python src/run_credit_risk_validation.py
```

## Resume Bullet

```text
Credit Risk PD Model Validation and Stress Testing: Built and independently validated borrower-level PD models using UCI German Credit Data, with Logistic Regression as an interpretable baseline and XGBoost as a challenger model. Evaluated data quality, discrimination, calibration, stability, cost-sensitive thresholds, and model limitations using AUC, KS, Brier Score, calibration curves, cross-validation, and confusion matrix analysis. Designed base/adverse/severe stress scenarios and mapped PD to expected loss using Basel-style PD/LGD/EAD concepts; produced a validation report aligned with SR 11-7-style model risk management principles.
```

## Interview Talking Point

I designed this project to simulate a model validation analyst workflow rather than a pure machine learning exercise. I first developed PD models, then challenged them from data quality, methodology, performance, calibration, benchmarking, stress testing, and limitation perspectives. The final deliverable is a validation report, because model risk management requires not only code but also clear evidence, judgment, documentation, and recommendations.
