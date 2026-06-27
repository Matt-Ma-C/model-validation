from __future__ import annotations

import json
import math
import os
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


RAW_DATA = PROJECT_ROOT / "data" / "raw" / "german.data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = OUTPUT_DIR / "credit_risk_pd_model_validation_report.pdf"
METRICS_PATH = OUTPUT_DIR / "metrics_summary.json"

RANDOM_STATE = 42
LGD = 0.45


COLUMNS = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate_pct",
    "personal_status_sex",
    "other_debtors",
    "present_residence_since",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "target",
]

NUMERIC_FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate_pct",
    "present_residence_since",
    "age_years",
    "existing_credits",
    "num_dependents",
]
CATEGORICAL_FEATURES = [c for c in COLUMNS if c not in NUMERIC_FEATURES + ["target"]]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Raw data not found: {RAW_DATA}")
    df = pd.read_csv(RAW_DATA, sep=r"\s+", names=COLUMNS)
    # UCI target: 1 = good credit, 2 = bad credit. PD model target uses bad = 1.
    df["bad"] = (df["target"] == 2).astype(int)
    df = df.drop(columns=["target"])
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models() -> dict[str, Pipeline]:
    logistic = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    xgb = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                XGBClassifier(
                    n_estimators=140,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=2,
                ),
            ),
        ]
    )
    return {"Logistic Regression": logistic, "XGBoost Challenger": xgb}


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    data = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score", ascending=False)
    total_bad = data["y"].sum()
    total_good = len(data) - total_bad
    data["cum_bad"] = data["y"].cumsum() / total_bad
    data["cum_good"] = ((1 - data["y"]).cumsum()) / total_good
    return float((data["cum_bad"] - data["cum_good"]).abs().max())


def expected_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # German Credit cost matrix: false good approval is more costly than false bad rejection.
    # We map bad=1. FN means bad customer predicted good, cost=5; FP means good predicted bad, cost=1.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float((5 * fn + 1 * fp) / len(y_true))


def find_cost_minimizing_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    costs = []
    for t in thresholds:
        costs.append(expected_cost(y_true, (y_score >= t).astype(int)))
    idx = int(np.argmin(costs))
    return float(thresholds[idx]), float(costs[idx])


def evaluate_model(name: str, model: Pipeline, x_train, x_test, y_train, y_test) -> dict:
    model.fit(x_train, y_train)
    y_score = model.predict_proba(x_test)[:, 1]
    threshold, min_cost = find_cost_minimizing_threshold(y_test.to_numpy(), y_score)
    y_pred = (y_score >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(model, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=None)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "name": name,
        "model": model,
        "y_score": y_score,
        "threshold": threshold,
        "auc": float(roc_auc_score(y_test, y_score)),
        "roc_auc_curve": float(auc(fpr, tpr)),
        "ks": ks_statistic(y_test.to_numpy(), y_score),
        "brier": float(brier_score_loss(y_test, y_score)),
        "min_cost": min_cost,
        "cv_auc_mean": float(np.mean(cv_auc)),
        "cv_auc_std": float(np.std(cv_auc)),
        "confusion_matrix": cm.tolist(),
        "fpr": fpr,
        "tpr": tpr,
    }


def plot_eda(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4.2))
    df["bad"].value_counts(normalize=True).sort_index().plot(kind="bar", color=["#4C78A8", "#E45756"])
    plt.xticks([0, 1], ["Good credit", "Bad credit"], rotation=0)
    plt.ylabel("Share")
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "target_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.2))
    for label, grp in df.groupby("bad"):
        plt.hist(grp["credit_amount"], bins=25, alpha=0.55, label="Bad" if label else "Good")
    plt.title("Credit Amount Distribution by Credit Outcome")
    plt.xlabel("Credit amount")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "credit_amount_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.2))
    bad_rate = df.groupby("duration_months")["bad"].mean().rolling(3, min_periods=1).mean()
    bad_rate.plot(color="#F58518")
    plt.title("Observed Bad Rate by Loan Duration")
    plt.xlabel("Duration months")
    plt.ylabel("Observed bad rate")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bad_rate_by_duration.png", dpi=180)
    plt.close()


def plot_validation(results: dict[str, dict], y_test: pd.Series) -> None:
    plt.figure(figsize=(7, 5))
    for res in results.values():
        plt.plot(res["fpr"], res["tpr"], label=f"{res['name']} AUC={res['auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#999999")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    for res in results.values():
        prob_true, prob_pred = calibration_curve(y_test, res["y_score"], n_bins=8, strategy="quantile")
        plt.plot(prob_pred, prob_true, marker="o", label=f"{res['name']} Brier={res['brier']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#999999")
    plt.xlabel("Mean predicted PD")
    plt.ylabel("Observed bad rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "calibration_curve.png", dpi=180)
    plt.close()

    metrics_df = pd.DataFrame(
        [
            {
                "Model": res["name"],
                "AUC": res["auc"],
                "KS": res["ks"],
                "Brier": res["brier"],
                "Cost-min threshold": res["threshold"],
                "Expected cost": res["min_cost"],
            }
            for res in results.values()
        ]
    )
    plt.figure(figsize=(8, 3))
    x = np.arange(len(metrics_df))
    plt.bar(x - 0.2, metrics_df["AUC"], width=0.2, label="AUC", color="#4C78A8")
    plt.bar(x, metrics_df["KS"], width=0.2, label="KS", color="#F58518")
    plt.bar(x + 0.2, 1 - metrics_df["Brier"], width=0.2, label="1 - Brier", color="#54A24B")
    plt.xticks(x, metrics_df["Model"], rotation=0)
    plt.ylim(0, 1)
    plt.title("Validation Metric Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "metric_comparison.png", dpi=180)
    plt.close()


def feature_importance(best: Pipeline) -> pd.DataFrame:
    preprocess = best.named_steps["preprocess"]
    model = best.named_steps["model"]
    feature_names = preprocess.get_feature_names_out()
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        coef = np.abs(model.coef_[0])
        importances = coef / coef.sum()
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(15)
    plt.figure(figsize=(7, 5))
    plt.barh(fi["feature"][::-1], fi["importance"][::-1], color="#4C78A8")
    plt.title("Top Model Drivers")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=180)
    plt.close()
    return fi


def make_stress_scenarios(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    x = df[FEATURES].copy()
    bad = df["bad"]
    category_bad_rates = {
        col: df.groupby(col)["bad"].mean().sort_values(ascending=False)
        for col in ["checking_status", "savings_status", "employment_since", "credit_history", "property"]
    }

    def stressed(frac: float, amount_mult: float, duration_mult: float) -> pd.DataFrame:
        scenario = x.copy()
        risk_index = scenario["credit_amount"].rank(pct=True) + scenario["duration_months"].rank(pct=True)
        affected = risk_index.sort_values(ascending=False).head(math.ceil(len(scenario) * frac)).index
        scenario.loc[affected, "credit_amount"] = (scenario.loc[affected, "credit_amount"] * amount_mult).round()
        scenario.loc[affected, "duration_months"] = (scenario.loc[affected, "duration_months"] * duration_mult).round()
        for col, rates in category_bad_rates.items():
            scenario.loc[affected, col] = rates.index[0]
        return scenario

    return {
        "Base": x,
        "Adverse": stressed(frac=0.25, amount_mult=1.10, duration_mult=1.15),
        "Severe": stressed(frac=0.50, amount_mult=1.25, duration_mult=1.30),
    }


def run_stress_test(best_model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scenarios = make_stress_scenarios(df)
    for name, x_scenario in scenarios.items():
        pd_values = best_model.predict_proba(x_scenario)[:, 1]
        ead = x_scenario["credit_amount"].to_numpy()
        expected_loss = pd_values * LGD * ead
        rows.append(
            {
                "Scenario": name,
                "Mean PD": float(pd_values.mean()),
                "Portfolio EAD": float(ead.sum()),
                "Expected Loss": float(expected_loss.sum()),
                "PD Uplift vs Base": np.nan,
                "EL Uplift vs Base": np.nan,
            }
        )
    stress = pd.DataFrame(rows)
    base_pd = stress.loc[stress["Scenario"] == "Base", "Mean PD"].iloc[0]
    base_el = stress.loc[stress["Scenario"] == "Base", "Expected Loss"].iloc[0]
    stress["PD Uplift vs Base"] = stress["Mean PD"] / base_pd - 1
    stress["EL Uplift vs Base"] = stress["Expected Loss"] / base_el - 1

    plt.figure(figsize=(7, 4.2))
    plt.bar(stress["Scenario"], stress["Mean PD"], color=["#4C78A8", "#F58518", "#E45756"])
    plt.ylabel("Mean portfolio PD")
    plt.title("Stress Test: Portfolio PD by Scenario")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "stress_pd.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.2))
    plt.bar(stress["Scenario"], stress["Expected Loss"], color=["#4C78A8", "#F58518", "#E45756"])
    plt.ylabel("Expected loss")
    plt.title("Stress Test: Expected Loss by Scenario")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "stress_expected_loss.png", dpi=180)
    plt.close()
    return stress


def make_metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": res["name"],
                "AUC": round(res["auc"], 4),
                "KS": round(res["ks"], 4),
                "Brier": round(res["brier"], 4),
                "CV AUC mean": round(res["cv_auc_mean"], 4),
                "CV AUC std": round(res["cv_auc_std"], 4),
                "Cost threshold": round(res["threshold"], 3),
                "Expected cost": round(res["min_cost"], 4),
            }
            for res in results.values()
        ]
    )


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("&", "&amp;"), style)


def table_from_df(df: pd.DataFrame, col_widths=None) -> Table:
    data = [list(df.columns)] + [[format_cell(v) for v in row] for row in df.to_numpy()]
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0B2545")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C8D1DA")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def format_cell(value) -> str:
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:,.0f}"
        return f"{value:.4f}"
    return str(value)


def add_image(story, filename: str, width=6.1 * inch) -> None:
    path = FIG_DIR / filename
    if path.exists():
        img = Image(str(path), width=width, height=width * 0.62)
        story.extend([img, Spacer(1, 0.12 * inch)])


def build_report(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    best_model_name: str,
) -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyTight",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12,
            spaceAfter=5,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            spaceAfter=4,
        )
    )

    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Credit Risk PD Model Validation Report",
    )
    story = []
    story.append(Paragraph("Credit Risk PD Model Validation and Stress Testing", styles["Title"]))
    story.append(paragraph("Educational project simulating a model validation analyst workflow for a credit risk probability of default model.", styles["BodyTight"]))
    story.append(Spacer(1, 0.1 * inch))

    summary = [
        ["Dataset", "UCI Statlog German Credit Data"],
        ["Sample size", f"{len(df)} borrowers"],
        ["Bad rate", f"{df['bad'].mean():.1%}"],
        ["Best model selected", best_model_name],
        ["Regulatory framing", "SR 11-7-style validation; Basel-style PD/LGD/EAD expected loss mapping."],
    ]
    story.append(table_from_df(pd.DataFrame(summary, columns=["Item", "Summary"]), col_widths=[1.7 * inch, 4.8 * inch]))
    story.append(Spacer(1, 0.15 * inch))

    sections = [
        ("1. Model Purpose and Intended Use", "The model estimates borrower-level probability of default (PD) for a small retail credit portfolio. It is not a production underwriting model; it is an educational model validation project designed to demonstrate quantitative testing, benchmarking, stress testing, and documentation discipline."),
        ("2. Data Quality Review", "The UCI German Credit dataset contains 1,000 borrower records with categorical and numeric credit attributes. The validation review covers target distribution, variable reasonableness, borrower profile distributions, and observed risk patterns by loan duration and credit amount."),
    ]
    for title, body in sections:
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(paragraph(body, styles["BodyTight"]))
    add_image(story, "target_distribution.png")
    add_image(story, "credit_amount_distribution.png")

    story.append(Paragraph("3. Methodology Review", styles["Heading2"]))
    story.append(paragraph("Two models are developed for benchmarking. Logistic Regression is used as an interpretable baseline aligned with traditional credit scoring practice. XGBoost is used as a nonlinear challenger model to test whether tree-based interactions improve discrimination. Both models use identical train/test split logic and preprocessing.", styles["BodyTight"]))

    story.append(Paragraph("4. Outcomes Analysis and Quantitative Testing", styles["Heading2"]))
    story.append(paragraph("Performance is evaluated through discrimination, calibration, cross-validation stability, and cost-sensitive threshold selection. AUC and KS measure ranking power; Brier Score and calibration curves measure probability quality; the German Credit cost matrix is used to reflect the higher cost of approving bad borrowers.", styles["BodyTight"]))
    story.append(table_from_df(metrics_df, col_widths=[1.35 * inch] + [0.75 * inch] * (len(metrics_df.columns) - 1)))
    story.append(Spacer(1, 0.1 * inch))
    add_image(story, "roc_curve.png")
    add_image(story, "calibration_curve.png")
    add_image(story, "metric_comparison.png")

    story.append(PageBreak())
    story.append(Paragraph("5. Model Drivers and Explainability", styles["Heading2"]))
    story.append(paragraph("The most influential model drivers are reviewed for plausibility. In a real validation setting, this review would be supplemented with policy review, SME challenge, adverse action reason-code testing, and fairness controls.", styles["BodyTight"]))
    story.append(table_from_df(feature_df.rename(columns={"feature": "Feature", "importance": "Importance"}).head(10), col_widths=[4.8 * inch, 1.2 * inch]))
    add_image(story, "feature_importance.png")

    story.append(Paragraph("6. Stress Testing and Basel-style Expected Loss Mapping", styles["Heading2"]))
    story.append(paragraph("Stress scenarios are designed to mimic adverse borrower and portfolio conditions. The project maps PD to expected loss using EL = PD x LGD x EAD, with LGD fixed at 45% and EAD proxied by credit amount. This is a simplified Basel-style risk parameter exercise, not a regulatory capital model.", styles["BodyTight"]))
    story.append(table_from_df(stress_df.round(4), col_widths=[1.0 * inch, 0.8 * inch, 1.1 * inch, 1.1 * inch, 1.2 * inch, 1.2 * inch]))
    add_image(story, "stress_pd.png")
    add_image(story, "stress_expected_loss.png")

    story.append(Paragraph("7. Validation Findings and Limitations", styles["Heading2"]))
    findings = [
        "Finding 1: The challenger model should not be selected purely on AUC; calibration, cost-sensitive decisioning, interpretability, and stability should be considered together.",
        "Finding 2: Sample size is limited and may not represent a current banking portfolio. External validation and out-of-time testing are required before real use.",
        "Finding 3: Stress scenarios are educational and judgment-based. A production stress testing program would require macroeconomic scenario design, governance, and back-testing.",
        "Finding 4: Expected loss uses fixed LGD and credit amount as EAD. Production Basel models would separately estimate or validate PD, LGD, EAD, and maturity assumptions.",
    ]
    for item in findings:
        story.append(paragraph("• " + item, styles["BodyTight"]))

    story.append(Paragraph("8. Recommendations", styles["Heading2"]))
    recs = [
        "Use Logistic Regression as the challenger/reference model for interpretability and governance comparison.",
        "Introduce out-of-time validation and population stability monitoring if new vintages become available.",
        "Calibrate PD estimates before using them for expected loss or decision thresholds.",
        "Document intended use, data limitations, assumptions, and model owner responses in a validation tracker.",
    ]
    for item in recs:
        story.append(paragraph("• " + item, styles["BodyTight"]))

    doc.build(story)


def main() -> None:
    ensure_dirs()
    df = load_data()
    plot_eda(df)
    x = df[FEATURES]
    y = df["bad"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    models = build_models()
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(name, model, x_train, x_test, y_train, y_test)

    plot_validation(results, y_test)
    metrics_df = make_metrics_table(results)
    best_name = metrics_df.sort_values(["AUC", "Brier"], ascending=[False, True]).iloc[0]["Model"]
    best_model = results[best_name]["model"]
    feature_df = feature_importance(best_model)
    stress_df = run_stress_test(best_model, df)

    metrics_payload = {
        "dataset": "UCI Statlog German Credit Data",
        "sample_size": int(len(df)),
        "bad_rate": float(df["bad"].mean()),
        "best_model": best_name,
        "model_metrics": metrics_df.to_dict(orient="records"),
        "stress_results": stress_df.to_dict(orient="records"),
        "top_features": feature_df.head(10).to_dict(orient="records"),
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    metrics_df.to_csv(OUTPUT_DIR / "model_validation_metrics.csv", index=False)
    stress_df.to_csv(OUTPUT_DIR / "stress_testing_results.csv", index=False)
    feature_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    build_report(df, metrics_df, stress_df, feature_df, best_name)
    print(f"Report written to: {REPORT_PATH}")
    print(f"Metrics written to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
