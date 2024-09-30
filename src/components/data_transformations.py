import os
import sys
import pandas as pd 
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformerConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_feature = ["writing score", "reading score"]
            categorical_feature = ["gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder(sparse_output=True)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"performing StandardScaling on {numerical_feature}")
            logging.info(f"performing One Hot Encoding on {categorical_feature}")

            preprocessor = ColumnTransformer(
                [
                ("Numerical Pipeline", numerical_pipeline, numerical_feature),
                ("Categorical Pipeline", categorical_pipeline, categorical_feature)
                ]
            )

            return preprocessor
        except Exception as e:
            logging.info("Failed during data transformer")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Importing train and test data for transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("data successfully read")
            logging.info("Obtaining preprocessing object...")
            preprocessor = self.get_data_transformer_object()

            target_column = "math score"
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saving preprocessing object...")

            save_object (
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor
            )

            logging.info("Saved preprocessing object...")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        

