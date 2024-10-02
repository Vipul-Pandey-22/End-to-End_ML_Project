import os
import sys
import dill
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model 

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr, preprocessor_path):
        try:
            logging.info("Split the training and testing data....")

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1], train_arr[:,-1],
                test_arr[:,:-1], test_arr[:,-1]
            ) 

            logging.info(f"train test split has been done succeefully...{X_train} \n {X_test}")

            models = {
                "Linear Regression": LinearRegression(),
                "K-Nearest Regressor": KNeighborsRegressor(),
                "Decison Tree Regressor": DecisionTreeRegressor(),
                "XgBoost Regressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "CatBoosting Regressor": CatBoostRegressor()
            }
            params={
                "Decison Tree Regressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XgBoost Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "K-Nearest Regressor": {}
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train, X_test=X_test, y_test=y_test,models=models, params=params)

            # to get the best model's score
            best_model_score = max(sorted(model_report.values()))
            # to get the best model's mame
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            # if best_model_score < 60: 
            #     raise CustomException("No best model found")
            
            logging.info(f"best model found on training and testing data for {best_model_name}")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # See the result
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            logging.info(f"best score for testing: {best_model_score}")
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
""" 
    def initiate_hyperparameter_tuning(self, train_arr, test_arr):
        try:
            logging.info("Split the training and testing data....")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1], train_arr[:,-1],
                test_arr[:,:-1], test_arr[:,-1]
            ) 
            models = {
                # "Ridge": RidgeCV(),
                # "Lasso": LassoCV(),
                # "ElastinNet":ElasticNetCV(),
                "K-Nearest Regressor": KNeighborsRegressor(),
                "Decison Tree Regressor": DecisionTreeRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }
            param_grid = {
                "K-Nearest Regressor": {
                    'n_neighbors': [3, 5, 7, 10],
                    'weights': ['uniform', 'distance']
                },
                "Decison Tree Regressor": {
                    "splitter":["best","random"],
                    "max_depth" : [1,3,5,7,9,11,12],
                    "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
                    "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                    "max_features":["auto","log2","sqrt",None],
                    "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] 
                },
                "XGBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1]
                },
                "Random Forest Regressor": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
            score_report:dict = hyperparameter_tuning(X_train, y_train, X_test, y_test, models, param_grid)
            print(score_report)
        except Exception as e:
            raise CustomException(e, sys)

"""