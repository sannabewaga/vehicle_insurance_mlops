import mlflow.sklearn
import pandas as pd
import os
import pickle
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.datascience.logging.logging import logger
from urllib.parse import urlparse
from src.datascience.entity.config_entity import ModelEvaluationConfig
from dotenv import load_dotenv

load_dotenv()

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_KEY")

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig) -> None:
        self.config = config
        
            

    def evaluate_model(self):
        try:
            logger.info("Loading test data for evaluation...")
            test_data = pd.read_csv(self.config.test_data_path)

            X_test = test_data.drop(columns=[self.config.target_column], errors="ignore")

            # Drop any non-feature columns like 'id' if present
            if "id" in X_test.columns:
                X_test = X_test.drop(columns=["id"])

            y_test = test_data[self.config.target_column]

            logger.info(f"Loading model from: {self.config.model_path}")
            with open(self.config.model_path, "rb") as f:
                model = pickle.load(f)

            logger.info("Generating predictions...")
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # Compute metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            cm = confusion_matrix(y_test, y_pred)



            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
            logger.info("Logging metrics to MLflow...")
            with mlflow.start_run():
                mlflow.log_param("model_name", self.config.model_path)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)
            
            logger.info(f"Evaluation complete. Accuracy: {acc:.4f}, F1: {f1:.4f}")
            logger.info(f"Confusion Matrix:\n{cm}")

        except Exception as e:
            logger.exception("Error during model evaluation.")
            raise e



    def run(self):
        self.evaluate_model()