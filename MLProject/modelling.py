import os
import json
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import optuna
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)
from dotenv import load_dotenv
load_dotenv()
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = "diabetes_preprocessing"
N_TRIALS = 5
CV_FOLDS = 3

DAGSHUB_OWNER = os.getenv("DAGSHUB_OWNER")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")

if not DAGSHUB_OWNER:
    raise ValueError("DAGSHUB_OWNER belum diset")

if not DAGSHUB_REPO:
    raise ValueError("DAGSHUB_REPO belum diset")


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "diabetes_train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "diabetes_test.csv"))
    return train, test


def objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }

    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    return float(scores.mean())


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Random Forest")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_feature_importance(model, feature_names, save_path):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(importances)), importances[indices], color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(sorted_names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importances - Random Forest")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_roc_curve(model, X_test, y_test, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name="Random Forest")
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_title("ROC Curve - Random Forest")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)

    os.environ.pop("MLFLOW_RUN_ID", None)

    train_df, test_df = load_data()

    X_train = train_df.drop("Outcome", axis=1)
    y_train = train_df["Outcome"]
    X_test = test_df.drop("Outcome", axis=1)
    y_test = test_df["Outcome"]

    feature_names = list(X_train.columns)

    mlflow.set_experiment("diabetes-rf-ci")

    print(f"Menjalankan Bayesian optimization ({N_TRIALS} trials, TPE sampler, {CV_FOLDS}-fold CV)...")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="rf-tpe-search",
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_params = study.best_params
    print(f"\nBest params: {best_params}")
    print(f"Best F1 (CV): {study.best_value:.4f}")

    # Latih ulang model final menggunakan parameter terbaik
    final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "best_cv_f1": study.best_value,
    }

    active_run = mlflow.active_run()

    if active_run is None:
        run_context = mlflow.start_run(run_name="rf-ci-best")
    else:
        run_context = mlflow.start_run(
            run_name="rf-ci-best",
            nested=True,
        )

    with run_context:
        # Params
        for k, v in best_params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("optimizer", "optuna-tpe")
        mlflow.log_param("dataset", "pima-indians-diabetes")

        # Metrics
        for name, val in metrics.items():
            mlflow.log_metric(name, val)

        # Model
        mlflow.sklearn.log_model(final_model, artifact_path="model")

        with tempfile.TemporaryDirectory() as tmp:
            # Artifact 1: confusion matrix
            cm_path = os.path.join(tmp, "confusion_matrix.png")
            plot_confusion_matrix(y_test, y_pred, cm_path)
            mlflow.log_artifact(cm_path, artifact_path="plots")

            # Artifact 2: feature importance
            fi_path = os.path.join(tmp, "feature_importance.png")
            plot_feature_importance(final_model, feature_names, fi_path)
            mlflow.log_artifact(fi_path, artifact_path="plots")

            # Artifact 3: ROC curve
            roc_path = os.path.join(tmp, "roc_curve.png")
            plot_roc_curve(final_model, X_test, y_test, roc_path)
            mlflow.log_artifact(roc_path, artifact_path="plots")

            # Artifact 4: classification report text
            report = classification_report(
                y_test, y_pred, target_names=["No Diabetes", "Diabetes"]
            )
            report_path = os.path.join(tmp, "classification_report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            mlflow.log_artifact(report_path)

            # Artifact 5: ringkasan trial Optuna (hyperparameter search history)
            trials_summary = [
                {"trial": t.number, "value": t.value, "params": t.params}
                for t in study.trials
                if t.value is not None
            ]
            summary_path = os.path.join(tmp, "optuna_trials_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"best_params": best_params, "best_cv_f1": study.best_value, "trials": trials_summary},
                    f,
                    indent=2,
                    default=str,
                )
            mlflow.log_artifact(summary_path)

        run_id = mlflow.active_run().info.run_id
        print(f"\nRun ID: {run_id}")
        print("\nMetrics final:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
