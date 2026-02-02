import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.configurations.aws_connection import AWSClient
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw', 'raw.csv')
    train_data_path: str = os.path.join('artifacts', 'raw', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'raw', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.bucket_name = 'stroke-prediction-bucket-mlops'
        self.s3_key = 'stroke-data.csv'

    def initiate_data_ingestion(self):
        try:
            logging.info('Data Ingestion Started.')

            aws_client = AWSClient()
            client = aws_client.client

            response = client.get_object(Bucket=self.bucket_name, Key=self.s3_key)
            print("✅ Successfully connected to AWS S3")

            
            df = pd.read_csv(response['Body'])
            logging.info('Dataset loaded.')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Raw dataset saved.')

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Data ingestion completed.')
            
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == '__main__':
    data_ingestor = DataIngestion()
    data_ingestor.initiate_data_ingestion()