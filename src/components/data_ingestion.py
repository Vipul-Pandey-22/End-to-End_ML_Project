import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformations import DataTransformation, DataTransformerConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', "train.csv")
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reading dataset from csv file
        """
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv('C:/Users/vip/Desktop/Projects/End-to-End-ML_Project/notebook/data/StudentsPerformance.csv')
            logging.info("Imported the dataset successfuly")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train Test Split Happen")
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion data has done......")

            return (
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_ingestion_with_db():
        pass
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr=train_arr, test_arr=test_arr, preprocessor_path=None)
    # model_trainer.initiate_hyperparameter_tuning(train_arr=train_arr, test_arr=test_arr)

