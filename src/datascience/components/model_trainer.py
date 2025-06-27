import pandas as pd
import os
import pickle
from src.datascience.logging.logging import logger
from src.datascience.entity.config_entity import ModelTrainerConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config = config

    def train_model(self):
        try:
            logger.info("Starting model training...")

            # Load training and testing data
            train = pd.read_csv(self.config.train_data_path)
            test = pd.read_csv(self.config.test_data_path)

            # Split into features and target
            x_train = train.drop(columns=[self.config.target_column])
            y_train = train[self.config.target_column]

            x_test = test.drop(columns=[self.config.target_column])
            y_test = test[self.config.target_column]

            # Prepare hyperparameters
            my_params = {
                'n_estimators': [self.config.n_estimators],
                'min_samples_split': [self.config.min_samples_split],
                'min_samples_leaf': [self.config.min_samples_leaf],
                'max_depth': [self.config.max_depth],
                'criterion': [self.config.criterion]
            }

            logger.info('Starting Model Training')

            # Set up the model and RandomizedSearchCV
            clf = RandomForestClassifier(random_state=42)
            model = RandomizedSearchCV(
                estimator=clf,
                param_distributions=my_params,
                n_iter=1,
                cv=4,
                verbose=1,
                random_state=101,
                n_jobs=-1
            )

            # Train the model
            model.fit(x_train, y_train)
            logger.info("Model training completed successfully.")

            # Save the model using pickle
            model_path = os.path.join(self.config.root_dir, self.config.model_name)
            os.makedirs(self.config.root_dir, exist_ok=True)

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            logger.info(f"Model saved successfully at: {model_path}")

        except Exception as e:
            logger.exception("Error occurred during model training.")
            raise e
    def run(self):
        self.train_model()
        