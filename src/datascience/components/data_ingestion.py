from src.datascience.config.configuration import ConfigurationManager
from src.datascience.entity.config_entity import DataIngestionConfig
from pymongo import MongoClient
import pandas as pd
from src.datascience.logging.logging import logger
import os

from dotenv import load_dotenv
load_dotenv()


class DataIngestion:
    def __init__(self,config:DataIngestionConfig) -> None:
        self.config = config


    def save_file(self):
        try:
            logger.info("Starting data ingestion from MongoDB...")

            # Connect to MongoDB
            client = MongoClient(os.getenv('MONGODB_URL_KEY'))
            db = client[os.getenv('DB_NAME')]
            collection = db[os.getenv('COLLECTION_NAME')]
            logger.info("Connected to MongoDB and accessed collection.")

            # Load documents into DataFrame
            data = list(collection.find())
            if not data:
                logger.warning("No data found in the MongoDB collection.")
            df = pd.DataFrame(data)
            logger.info(f"Retrieved {len(df)} records from MongoDB.")

            # Drop _id if exists
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
                logger.info("Dropped '_id' column from DataFrame.")

            # Ensure target directory exists
            os.makedirs(self.config.unzip_dir, exist_ok=True)
            logger.info(f"Directory verified/created: {self.config.unzip_dir}")

            # Save DataFrame to fixed file name
            file_path = os.path.join(self.config.unzip_dir, "raw_data.csv")
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved to CSV at: {file_path}")

        except Exception as e:
            logger.exception("Error during data ingestion.")
            raise e



    def run(self):
        self.save_file()