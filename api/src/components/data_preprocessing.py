import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

from src.logger import logging
from src.exception import CustomException
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DataPreprocessorConfig:
    train_processed_data_path: str = os.path.join('artifacts', 'processed', 'train.csv')
    test_processed_data_path: str = os.path.join('artifacts', 'processed', 'test.csv')


class DataProcessor:
    def __init__(self):
        self.processing_config = DataPreprocessorConfig()

    def preprocess_data(self, train_df, test_df):
        try:

            logging.info('Data processing started.')
            # dropping the missing values in bmi feature
            train_df['bmi'] = train_df['bmi'].fillna(train_df['bmi'].median())
            test_df['bmi'] = test_df['bmi'].fillna(test_df['bmi'].median())

            # applying log transformation to bmi and avg_glucose_level features
            train_df['bmi'] = np.log1p(train_df['bmi'])
            train_df['avg_glucose_level'] = np.log1p(train_df['avg_glucose_level'])
            test_df['bmi'] = np.log1p(test_df['bmi'])
            test_df['avg_glucose_level'] = np.log1p(test_df['avg_glucose_level'])

            # remove the Other category from gender feature
            train_df = train_df[train_df['gender'] != 'Other']
            test_df = test_df[test_df['gender'] != 'Other']

            # grouping rare work types
            train_df['work_type'] = train_df['work_type'].replace({
                'Never_worked': 'Other',
                'children': 'Other'
            })
            test_df['work_type'] = test_df['work_type'].replace({
                'Never_worked': 'Other',
                'children': 'Other'
            })

            # creating age threshold feature
            train_df['over_40'] = (train_df['age'] > 40).astype(int)
            test_df['over_40'] = (test_df['age'] > 40).astype(int)

            # creating medical risk score
            train_df['medical_risk_score'] = train_df['hypertension'] + train_df['heart_disease']
            test_df['medical_risk_score'] = test_df['hypertension'] + test_df['heart_disease']

            # one-hot encoding gender and Residence_type features
            train_df = pd.get_dummies(train_df, columns=['gender', 'Residence_type'], drop_first=True)
            test_df = pd.get_dummies(test_df, columns=['gender', 'Residence_type'], drop_first=True)

            # ordinal encoding ever_married, work_type, smoking_status features
            train_df['ever_married'] = train_df['ever_married'].map({'Yes': 1, 'No': 0})
            train_df['work_type'] = train_df['work_type'].map({'Self-employed': 2, 'Private': 1, 'Govt_job': 1, 'Other': 0})
            train_df['smoking_status'] = train_df['smoking_status'].map({'formerly smoked': 2, 'smokes': 1, 'never smoked': 0, 'Unknown': 0})
            test_df['ever_married'] = test_df['ever_married'].map({'Yes': 1, 'No': 0})
            test_df['work_type'] = test_df['work_type'].map({'Self-employed': 2, 'Private': 1, 'Govt_job': 1, 'Other': 0})
            test_df['smoking_status'] = test_df['smoking_status'].map({'formerly smoked': 2, 'smokes': 1, 'never smoked': 0, 'Unknown': 0})

            # scaling the features
            continuous_features = ['age', 'bmi', 'avg_glucose_level']
            scaler = StandardScaler()
            train_df[continuous_features] = scaler.fit_transform(train_df[continuous_features])
            test_df[continuous_features] = scaler.transform(test_df[continuous_features])

            logging.info('Data processing completed.')

            os.makedirs(os.path.dirname(self.processing_config.train_processed_data_path), exist_ok=True)
            train_df.to_csv(self.processing_config.train_processed_data_path, index=False, header=True)
            test_df.to_csv(self.processing_config.test_processed_data_path, index=False, header=True)

            logging.info('Processed data saved.')

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':

    # path for the raw set
    train_path = 'artifacts/raw/train.csv'
    test_path = 'artifacts/raw/test.csv'

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    data_processor = DataProcessor()
    data_processor.preprocess_data(train_df, test_df)