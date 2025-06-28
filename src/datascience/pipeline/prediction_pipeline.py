import sys
import pickle
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from src.datascience.logging.logging import logger


class VehicleData:
    def __init__(
        self,
        Gender,
        Age,
        Driving_License,
        Region_Code,
        Previously_Insured,
        Annual_Premium,
        Policy_Sales_Channel,
        Vintage,
        Vehicle_Age_lt_1_Year,
        Vehicle_Age_gt_2_Years,
        Vehicle_Damage_Yes,
    ):
        try:
            self.Gender = Gender
            self.Age = Age
            self.Driving_License = Driving_License
            self.Region_Code = Region_Code
            self.Previously_Insured = Previously_Insured
            self.Annual_Premium = Annual_Premium
            self.Policy_Sales_Channel = Policy_Sales_Channel
            self.Vintage = Vintage
            self.Vehicle_Age_lt_1_Year = Vehicle_Age_lt_1_Year
            self.Vehicle_Age_gt_2_Years = Vehicle_Age_gt_2_Years
            self.Vehicle_Damage_Yes = Vehicle_Damage_Yes

        except Exception as e:
            raise e

    def get_vehicle_input_dataframe(self) -> DataFrame:
        try:
            return DataFrame(self.get_vehicle_data_as_dict())
        except Exception as e:
            raise e

    def get_vehicle_data_as_dict(self):
        try:
            logger.info("Creating dictionary from input data")
            return {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Driving_License": [self.Driving_License],
                "Region_Code": [self.Region_Code],
                "Previously_Insured": [self.Previously_Insured],
                "Annual_Premium": [self.Annual_Premium],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel],
                "Vintage": [self.Vintage],
                "Vehicle_Age_lt_1_Year": [self.Vehicle_Age_lt_1_Year],
                "Vehicle_Age_gt_2_Years": [self.Vehicle_Age_gt_2_Years],
                "Vehicle_Damage_Yes": [self.Vehicle_Damage_Yes],
            }
        except Exception as e:
            raise e


class VehiclePredictor:
    def __init__(self, model_path: str = "artifacts/model_trainer/rf_model.pkl"):
        try:
            logger.info(f"Loading model from {model_path}")
            with open(Path(model_path), "rb") as f:
                self.model = pickle.load(f)
        except Exception as e:
            raise e

    def predict(self, df: pd.DataFrame) -> str:
        try:
            logger.info("Performing prediction")
            pred = self.model.predict(df)
            return str(pred[0])
        except Exception as e:
            raise e
