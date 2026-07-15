import logging
import sys
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import split_data  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def get_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, random_state=RANDOM_STATE, verbose=-1
        ),
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics, y_pred, y_proba


def train_and_track(
    data_path: str = None,
    experiment_name: str = "credit-risk-model",
):
    project_root = Path(__file__).resolve().parent.parent

    if data_path is None:
        data_path = project_root / "data" / "processed" / "processed_data.csv"

    logger.info(f"Loading processed data from: {data_path}")
    df = pd.read_csv(data_path)

    target_col = "is_high_risk"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    # Drop ID, datetime, and constant columns that aren't features
    cols_to_drop = [
        "TransactionId",
        "BatchId",
        "AccountId",
        "SubscriptionId",
        "CustomerId",
        "CurrencyCode",
        "TransactionStartTime",
        "FirstTransaction",
        "LastTransaction",
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped non-feature columns: {cols_to_drop}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, target_col=target_col, random_state=RANDOM_STATE
    )

    feature_cols = [c for c in X_train.columns if c != target_col]
    logger.info(f"Features: {len(feature_cols)} columns")

    mlflow.set_experiment(experiment_name)

    models = get_models()
    results = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")

        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)

            train_metrics, _, _ = evaluate_model(model, X_train, y_train)
            test_metrics, y_pred, y_proba = evaluate_model(
                model, X_test, y_test
            )

            mlflow.log_params(
                model.get_params() if hasattr(model, "get_params") else {}
            )
            mlflow.log_metrics(
                {f"train_{k}": v for k, v in train_metrics.items()}
            )
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in test_metrics.items()}
            )

            if hasattr(model, "feature_importances_"):
                model_len = len(model.feature_importances_)
                importance_df = pd.DataFrame(
                    {
                        "feature": feature_cols[:model_len],
                        "importance": model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)
                mlflow.log_table(importance_df, "feature_importance.json")

            cm = confusion_matrix(y_test, y_pred)
            mlflow.log_text(str(cm), "confusion_matrix.txt")
            mlflow.log_text(
                classification_report(y_test, y_pred),
                "classification_report.txt",
            )

            if name == "XGBoost":
                mlflow.xgboost.log_model(model, f"{name}_model")
            elif name == "LightGBM":
                mlflow.lightgbm.log_model(model, f"{name}_model")
            else:
                mlflow.sklearn.log_model(model, f"{name}_model")

            test_msg = f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}"

            logger.info(f"{name} - {test_msg}")
            results[name] = {
                "model": model,
                "test_metrics": test_metrics,
                "run_id": mlflow.active_run().info.run_id,
            }

    best_name = max(
        results,
        key=lambda k: results[k]["test_metrics"]["roc_auc"],
    )
    best_result = results[best_name]
    message = f"(ROC-AUC: {best_result['test_metrics']['roc_auc']:.4f})"

    logger.info(f"Best model: {best_name} {message}")

    with mlflow.start_run(run_name=f"{best_name}_best"):
        mlflow.log_metric(
            "best_roc_auc",
            best_result["test_metrics"]["roc_auc"],
        )
        mlflow.log_metric("best_f1", best_result["test_metrics"]["f1"])
        mlflow.log_text(best_name, "best_model_name.txt")

        if best_name in ["XGBoost", "LightGBM"]:
            if best_name == "XGBoost":
                mlflow.xgboost.log_model(best_result["model"], "model")
            else:
                mlflow.lightgbm.log_model(best_result["model"], "model")
        else:
            mlflow.sklearn.log_model(best_result["model"], "model")

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        result = mlflow.register_model(model_uri, "credit-risk-best-model")

        msg = "Model registered as 'credit-risk-best-model'"

        logger.info(f"{msg} (version {result.version})")

    return results


if __name__ == "__main__":
    results = train_and_track()
    print("\n========== TRAINING COMPLETE ==========")

    print("\n========== MODEL COMPARISON ==========")
    for name, res in results.items():
        m = res["test_metrics"]
        print(
            f"{name:25s} | Acc: {m['accuracy']:.4f} | "
            f"Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | "
            f"F1: {m['f1']:.4f} | AUC: {m['roc_auc']:.4f}"
        )
