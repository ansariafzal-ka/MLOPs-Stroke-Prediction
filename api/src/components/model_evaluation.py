import pandas as pd
import os
import sys
import pickle

from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import recall_score, precision_score
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class ModelEvaluation:

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            return model

    def evaluate_model(self, X_test, y_test, model):
        try:
            logging.info('Model evaluation started.')
            y_pred = model.predict(X_test)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)

            mlflow.set_experiment('stroke_prediction')

            with mlflow.start_run(run_name='model_evaluation'):
                mlflow.log_metric('recall', recall)
                mlflow.log_metric('precision', precision)

            logging.info(f'Recall: {recall}, Precision: {precision}')
            logging.info('Model evaluation completed.')
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    processed_data_path = 'artifacts/processed/test.csv'
    test_df = pd.read_csv(processed_data_path)
    X_test = test_df.drop('stroke', axis=1)
    y_test = test_df['stroke']
    model_path = 'artifacts/model.pkl'

    model_evaluation = ModelEvaluation()
    model = model_evaluation.load_model(model_path)
    model_evaluation.evaluate_model(X_test, y_test, model)
    