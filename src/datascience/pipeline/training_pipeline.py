from src.datascience.components.data_ingestion import DataIngestion
from src.datascience.components.data_validation import DataValidation
from src.datascience.components.data_transformation import DataTransformation
from src.datascience.components.model_trainer import ModelTrainer
from src.datascience.components.model_evaluation import ModelEvaluation
from src.datascience.config.configuration import ConfigurationManager
from src.datascience.logging.logging import logger


class TrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()

    def run_pipeline(self):
        try:
            # Step 1: Data Ingestion
            logger.info(">>>>> Starting data ingestion")
            data_ingestion_config = self.config.get_data_ingestion_config()
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion.run()
            logger.info("<<<<< Data ingestion complete")

            # Step 2: Data Validation
            logger.info(">>>>> Starting data validation")
            data_validation_config = self.config.get_data_validation_config()
            data_validation = DataValidation(data_validation_config)
            data_validation.run()
            logger.info("<<<<< Data validation complete")

            # Step 3: Data Transformation
            logger.info(">>>>> Starting data transformation")
            data_transformation_config = self.config.get_data_transformation_config()
            data_transformation = DataTransformation(data_transformation_config)
            data_transformation.run()
            logger.info("<<<<< Data transformation complete")

            # Step 4: Model Training
            logger.info(">>>>> Starting model training")
            model_trainer_config = self.config.get_model_trainer_config()
            model_trainer = ModelTrainer(model_trainer_config)
            model_trainer.run()
            logger.info("<<<<< Model training complete")

            # Step 5: Model Evaluation
            logger.info(">>>>> Starting model evaluation")
            model_evaluation_config = self.config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(model_evaluation_config)
            model_evaluation.run()
            logger.info("<<<<< Model evaluation complete")

        except Exception as e:
            logger.exception("❌ Training pipeline failed.")
            raise e


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
