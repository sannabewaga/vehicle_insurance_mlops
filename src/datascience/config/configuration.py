from src.datascience.constants import *
from src.datascience.utils.common import *
from src.datascience.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig


class ConfigurationManager:
    def __init__(self,config_file_path = CONFIG_FILE_PATH,params_file_path = PARAMS_FILE_PATH,schema_file_path = SCHEMA_FILE_PATH) -> None:

        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        self.schema = read_yaml(schema_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self):

        config = self.config.data_ingestion

        create_directories([config.root_dir])

        return DataIngestionConfig(

            root_dir= config.root_dir,
            unzip_dir=config.unzip_dir
        )
    
    def get_data_validation_config(self):

        config = self.config.data_validation
        schema = self.schema.COLUMNS
        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=config.unzip_data_dir,

            all_schema=schema
        )

    def get_data_transformation_config(self):

        config = self.config.data_transformation

        create_directories([config.root_dir])

        return DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path
        )
    
    def get_model_trainer_config(self):
        config = self.config.model_trainer
        params = self.params.RandomForest

        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        return ModelTrainerConfig(
            root_dir = config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path =  config.test_data_path,
            model_name = config.model_name,
            n_estimators = params.n_estimators,
            min_samples_split = params.min_samples_split,
            min_samples_leaf= params.min_samples_leaf,
            max_depth = params.max_depth,
            criterion = params.criterion,
            target_column = schema.name
        )

    def get_model_evaluation_config(self):
        config = self.config.model_evaluation
        params = self.params
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            
            root_dir=  config.root_dir,
            test_data_path=config.test_data_path,
            model_path = config.model_path,
            all_params= params.RandomForest,
            metric_file_name=  config.metric_file_name,
            target_column=schema.name,
            mlflow_uri="https://dagshub.com/sannabewaga/vehicle_insurance_mlops.mlflow"
            )
    
