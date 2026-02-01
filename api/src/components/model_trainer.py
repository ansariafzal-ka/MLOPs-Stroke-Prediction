import pandas as pd
import numpy as np
import os
import sys
import pickle
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, y_train):
        try:
            logging.info('Model training started.')

            mlflow.set_experiment('stroke_prediction')

            with mlflow.start_run(run_name='logistic regressor'):

                params = {
                    'class_weight': 'balanced',
                    'C': 1,
                    'max_iter': 3000,
                    'solver': 'lbfgs',
                    'random_state': 42
                }

                mlflow.log_params(params)

                model = LogisticRegression(**params)
                model.fit(X_train, y_train)

                logging.info('Model training completed.')

                mlflow.sklearn.log_model(sk_model=model, name="model")

                os.makedirs(os.path.dirname(self.model_config.trained_model_path), exist_ok=True)
                with open(self.model_config.trained_model_path, 'wb') as f:
                    pickle.dump(model, f)

                logging.info('Model saved successfully.')

                return model
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    processed_data_path = 'artifacts/processed/train.csv'
    train_df = pd.read_csv(processed_data_path)
    X_train = train_df.drop('stroke', axis=1)
    y_train = train_df['stroke']

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(X_train, y_train)
