from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "diabetes.csv"
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "diabetes_model.joblib"
TARGET = "Outcome"
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
# In this dataset, zero is used for missing measurements in these columns.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

st.set_page_config(page_title="Diabetes Model Lab", page_icon="🧪", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    """Read the teaching dataset bundled beside this app."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH.name}. Keep diabetes.csv in the app folder."
        )
    data = pd.read_csv(DATA_PATH)
    missing_columns = sorted(set(FEATURES + [TARGET]) - set(data.columns))
    if missing_columns:
        raise ValueError(f"The dataset is missing columns: {', '.join(missing_columns)}")
    return data[FEATURES + [TARGET]].copy()


def build_pipeline(model_name: str, n_estimators: int, max_depth: int | None) -> Pipeline:
    """Keep preprocessing and the classifier together in one fitted pipeline."""
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]

    # These distance-based/linear models benefit from standardized features.
    if model_name in {"Logistic Regression", "Support Vector Machine"}:
        steps.append(("scaler", StandardScaler()))

    if model_name == "Logistic Regression":
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    elif model_name == "Decision Tree":
        estimator = DecisionTreeClassifier(
            max_depth=max_depth, class_weight="balanced", random_state=42
        )
    elif model_name == "Random Forest":
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "Support Vector Machine":
        estimator = SVC(probability=True, class_weight="balanced", random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    steps.append(("model", estimator))
    return Pipeline(steps)


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data[FEATURES].copy()
    features[ZERO_AS_MISSING] = features[ZERO_AS_MISSING].replace(0, float("nan"))
    return features


def model_config(
    strategy: str, selected_model: str, n_estimators: int, max_depth: int | None
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "selected_model": selected_model,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }


def train_models(
    data: pd.DataFrame,
    strategy: str,
    selected_model: str,
    n_estimators: int,
    max_depth: int | None,
):
    X = prepare_features(data)
    y = data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidates = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine",
    ]
    cv_results: list[tuple[str, float]] = []

    if strategy == "Compare models and select the best":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for name in candidates:
            candidate = build_pipeline(name, n_estimators, max_depth)
            scores = cross_val_score(candidate, X_train, y_train, cv=cv, scoring="f1")
            cv_results.append((name, float(scores.mean())))
        chosen_model = max(cv_results, key=lambda item: item[1])[0]
    else:
        chosen_model = selected_model

    fitted_model = build_pipeline(chosen_model, n_estimators, max_depth)
    fitted_model.fit(X_train, y_train)
    predictions = fitted_model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No diabetes", "Diabetes"],
            output_dict=True,
            zero_division=0,
        ),
    }
    return chosen_model, fitted_model, metrics, cv_results


def save_model_bundle(model_name: str, model: Pipeline, metrics: dict[str, Any]) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    bundle = {
        "model": model,
        "model_name": model_name,
        "features": FEATURES,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "test_metrics": {
            key: float(metrics[key]) for key in ("accuracy", "precision", "recall", "f1")
        },
        "test_confusion_matrix": metrics["confusion_matrix"].tolist(),
        "classification_report": metrics["classification_report"],
    }
    joblib.dump(bundle, MODEL_PATH)


def show_model_results(metrics: dict[str, Any]) -> None:
    st.subheader("Holdout test results")
    st.caption("The held-out test portion was not used to fit the model.")
    cols = st.columns(4)
    for col, label, key in zip(
        cols,
        ["Accuracy", "Precision", "Recall", "F1 score"],
        ["accuracy", "precision", "recall", "f1"],
    ):
        col.metric(label, f"{metrics[key]:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            metrics["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No diabetes", "Diabetes"],
            yticklabels=["No diabetes", "Diabetes"],
            ax=ax,
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("**Classification report**")
        report = pd.DataFrame(metrics["classification_report"]).transpose()
        st.dataframe(report.round(3), use_container_width=True)


def show_prediction_form(model: Pipeline) -> None:
    st.subheader("Try a sample prediction")
    st.caption(
        "Enter sample measurements to see how the trained model responds. "
        "This classroom example is not a medical diagnosis."
    )
    with st.form("prediction_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose", 1, 250, 120)
        with c2:
            blood_pressure = st.number_input("Blood pressure", 1, 150, 70)
            skin_thickness = st.number_input("Skin thickness", 1, 100, 20)
        with c3:
            insulin = st.number_input("Insulin", 1, 900, 80)
            bmi = st.number_input("BMI", 1.0, 70.0, 32.0, step=0.1)
        with c4:
            pedigree = st.number_input(
                "Diabetes pedigree function", 0.0, 3.0, 0.47, step=0.01, format="%.2f"
            )
            age = st.number_input("Age", 21, 100, 33)
        submitted = st.form_submit_button("Predict")

    if submitted:
        sample = pd.DataFrame(
            [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]],
            columns=FEATURES,
        )
        prediction = int(model.predict(sample)[0])
        if prediction == 1:
            st.warning("The model classified this sample as Outcome = 1.")
        else:
            st.success("The model classified this sample as Outcome = 0.")
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(sample)[0][1])
            st.write(f"Model-estimated probability for Outcome = 1: **{probability:.1%}**")


st.title("Diabetes Classification: Interactive Model Lab")
st.write(
    "Choose an algorithm or compare several algorithms, train the selected model, "
    "review its evaluation, and try a sample prediction."
)
st.info(
    "Educational demonstration only. The dataset and model are not intended for "
    "clinical decisions or individual health advice."
)

try:
    df = load_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Dataset overview", expanded=False):
    st.write(f"Rows: **{len(df):,}** · Input features: **{len(FEATURES)}** · Target: **{TARGET}**")
    st.dataframe(df.head(), use_container_width=True)
    counts = df[TARGET].value_counts().sort_index().rename(
        index={0: "No diabetes", 1: "Diabetes"}
    )
    st.bar_chart(counts)
    st.caption("In this CSV, Outcome 0 means no diabetes and Outcome 1 means diabetes.")

st.sidebar.header("Training setup")
strategy = st.sidebar.radio(
    "How should the model be chosen?",
    ["Select a model", "Compare models and select the best"],
    help="Automatic selection compares mean F1 across 5 stratified folds on the training data.",
)
selected_model = "Logistic Regression"
if strategy == "Select a model":
    selected_model = st.sidebar.selectbox(
        "Choose an algorithm",
        ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine"],
    )
n_estimators = st.sidebar.slider("Trees in the Random Forest", 50, 400, 200, 50)
depth_option = st.sidebar.selectbox("Maximum tree depth", ["No limit", 3, 5, 8, 12], index=0)
max_depth = None if depth_option == "No limit" else int(depth_option)
save_after_training = st.sidebar.checkbox("Save trained model to this project", value=True)

current_config = model_config(strategy, selected_model, n_estimators, max_depth)
if st.sidebar.button("Load saved model", disabled=not MODEL_PATH.exists()):
    try:
        saved_bundle = joblib.load(MODEL_PATH)
        saved_version = saved_bundle.get("scikit_learn_version", "unknown")
        st.session_state["active_model"] = saved_bundle["model"]
        st.session_state["active_model_name"] = saved_bundle.get("model_name", "Saved model")
        saved_metrics = saved_bundle.get("test_metrics")
        if saved_metrics is not None and "test_confusion_matrix" in saved_bundle:
            saved_metrics["confusion_matrix"] = np.asarray(
                saved_bundle["test_confusion_matrix"]
            )
            saved_metrics["classification_report"] = saved_bundle["classification_report"]
        st.session_state["active_metrics"] = saved_metrics
        st.session_state["active_cv_results"] = []
        st.session_state["active_config"] = None
        if saved_version != sklearn.__version__:
            st.sidebar.warning(
                f"Saved with scikit-learn {saved_version}; current version is {sklearn.__version__}. "
                "Loading across versions may not be supported."
            )
        else:
            st.sidebar.success(f"Loaded {st.session_state['active_model_name']}.")
    except Exception as exc:
        st.sidebar.error(f"Could not load the saved model: {exc}")

if st.button("Train / update model", type="primary"):
    with st.spinner("Training and evaluating the model..."):
        chosen_name, fitted_model, metrics, cv_results = train_models(
            df, strategy, selected_model, n_estimators, max_depth
        )
        st.session_state["active_model"] = fitted_model
        st.session_state["active_model_name"] = chosen_name
        st.session_state["active_metrics"] = metrics
        st.session_state["active_cv_results"] = cv_results
        st.session_state["active_config"] = current_config
        if save_after_training:
            save_model_bundle(chosen_name, fitted_model, metrics)

if "active_model" in st.session_state:
    active_name = st.session_state["active_model_name"]
    st.success(f"Active model: **{active_name}**")
    if st.session_state.get("active_config") not in (None, current_config):
        st.warning("Training settings changed. Click **Train / update model** to apply them.")
    if st.session_state.get("active_cv_results"):
        st.subheader("Cross-validation comparison (training data only)")
        comparison = pd.DataFrame(
            st.session_state["active_cv_results"], columns=["Model", "Mean CV F1"]
        ).sort_values("Mean CV F1", ascending=False)
        st.dataframe(comparison.style.format({"Mean CV F1": "{:.3f}"}), hide_index=True)
        st.caption("The top mean cross-validation F1 score selected the model shown above.")
    if st.session_state.get("active_metrics") is not None:
        show_model_results(st.session_state["active_metrics"])
    show_prediction_form(st.session_state["active_model"])
else:
    st.caption("Choose training settings in the sidebar, then click **Train / update model**.")
