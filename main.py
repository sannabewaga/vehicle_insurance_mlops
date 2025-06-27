from src.datascience.components.data_ingestion import DataIngestion

from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.data_validation import DataValidation
from src.datascience.components.model_trainer import ModelTrainer
from src.datascience.components.data_transformation import DataTransformation




config = ConfigurationManager()
# data_ingestion_config = config.get_data_ingestion_config()


# di = DataIngestion(data_ingestion_config)
# di.run()

# data_validation_config = config.get_data_validation_config()
# dv = DataValidation(data_validation_config)
# dv.run()

# data_transformation_config = config.get_data_transformation_config()

# dt = DataTransformation(data_transformation_config)

# dt.run()


model_trainer_config = config.get_model_trainer_config()

mt= ModelTrainer(model_trainer_config)
mt.run()
