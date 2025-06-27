import pandas as pd
import os
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from src.datascience.logging.logging import logger
from src.datascience.entity.config_entity import DataTransformationConfig
from collections import Counter

class DataTransformation:
    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config

    def transform_data(self):
        try:
            logger.info("Starting data transformation...")

            # Load the data
            data = pd.read_csv(self.config.data_path)
            logger.info(f"Loaded data from {self.config.data_path} with shape: {data.shape}")

            # Feature mapping and encoding
            data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
            data = pd.get_dummies(data, drop_first=True)

            # Rename and convert columns
            data = data.rename(columns={
                "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
                "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
            })

            for col in ['Vehicle_Age_lt_1_Year', 'Vehicle_Age_gt_2_Years', 'Vehicle_Damage_Yes']:
                if col in data.columns:
                    data[col] = data[col].astype(int)
                else:
                    logger.warning(f"Column '{col}' not found in data after get_dummies.")

            # Cast categorical features as string
            cat_feat = [
                'Gender', 'Driving_License', 'Previously_Insured',
                'Vehicle_Age_lt_1_Year', 'Vehicle_Age_gt_2_Years',
                'Vehicle_Damage_Yes', 'Region_Code', 'Policy_Sales_Channel'
            ]

            for column in cat_feat:
                if column in data.columns:
                    data[column] = data[column].astype(str)

            # Drop NA rows
            data.dropna(inplace=True)
            logger.info(f"Data shape after preprocessing: {data.shape}")

            # Apply SMOTEENN
            X = data.drop(columns=["Response"])
            y = data["Response"]

            smote_enn = SMOTEENN(random_state=42)
            X_resampled, y_resampled = smote_enn.fit_resample(X, y)

            logger.info(f"Resampled data shape: {X_resampled.shape}")

            # Reconstruct the DataFrame
            data = X_resampled.copy()
            data["Response"] = y_resampled

            # Train-Test Split
            train, test = train_test_split(data, test_size=0.2, random_state=42)
            logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")

            # Save the transformed data
            train_path = os.path.join(self.config.root_dir, "train.csv")
            test_path = os.path.join(self.config.root_dir, "test.csv")

            os.makedirs(self.config.root_dir, exist_ok=True)

            train.to_csv(train_path, index=False)
            test.to_csv(test_path, index=False)

            logger.info(f"Train data saved at: {train_path}")
            logger.info(f"Test data saved at: {test_path}")

            

        except Exception as e:
            logger.exception("Error occurred during data transformation.")
            raise e

    def run(self):
        self.transform_data()
