import pandas as pd
import os
from src.datascience.logging.logging import logger
from src.datascience.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            logger.info("Starting data validation process...")
            df = pd.read_csv(self.config.unzip_data_dir)
            logger.info(f"Loaded data from {self.config.unzip_data_dir} with shape: {df.shape}")

            expected_columns = self.config.all_schema.keys()
            actual_columns = df.columns

            missing_columns = [col for col in expected_columns if col not in actual_columns]
            if missing_columns:
                logger.warning(f"Missing columns in data: {missing_columns}")
                status = False
            else:
                logger.info("All expected columns are present.")
                status = True

            # Write validation status to file
            status_str = "SUCCESS" if status else "FAILED"
            os.makedirs(os.path.dirname(self.config.STATUS_FILE), exist_ok=True)
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Validation Status: {status_str}\n")
            logger.info(f"Validation status '{status_str}' written to: {self.config.STATUS_FILE}")

        except Exception as e:
            logger.exception("Error occurred during data validation.")
            raise e

    def run(self):
        self.validate_all_columns()
        