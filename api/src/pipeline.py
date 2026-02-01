from components.data_ingestion import DataIngestion
from components.data_preprocessing import DataProcessor
from components.model_trainer import ModelTrainer
from components.model_evaluation import ModelEvaluation
import pandas as pd
import sys

from src.exception import CustomException

def run_pipeline():
    try:
        print('='*50)
        print('Starting ML Pipeline')
        print('='*50)
        print('Data Ingestion')

        # Data ingestion
        data_ingestor = DataIngestion()
        data_ingestor.initiate_data_ingestion()

        print('='*50)
        print('Data Processing')

        # Data processing
        train_path = 'artifacts/raw/train.csv'
        test_path = 'artifacts/raw/test.csv'

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        data_processor = DataProcessor()
        data_processor.preprocess_data(train_df, test_df)

        print('='*50)
        print('Model Trainer')

        # Model trainer
        processed_data_path = 'artifacts/processed/train.csv'
        train_df = pd.read_csv(processed_data_path)
        X_train = train_df.drop('stroke', axis=1)
        y_train = train_df['stroke']

        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(X_train, y_train)

        print('='*50)
        print('Model Evaluation')

        # Model evaluation
        processed_data_path = 'artifacts/processed/test.csv'
        test_df = pd.read_csv(processed_data_path)
        X_test = test_df.drop('stroke', axis=1)
        y_test = test_df['stroke']
        model_path = 'artifacts/model.pkl'

        model_evaluation = ModelEvaluation()
        model = model_evaluation.load_model(model_path)
        model_evaluation.evaluate_model(X_test, y_test, model)

        print('='*50)
        print('Pipeline Execution Completed')
        print('='*50)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == '__main__':
    run_pipeline()