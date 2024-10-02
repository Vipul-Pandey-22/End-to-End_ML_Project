import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
        logging.info("File saved successfully...")
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train ,y_train, X_test, y_test, models, params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_key = list(models.keys())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param_grid=para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            logging.info(
                f"""Model training has been done.... \n
                see the best parametes: {gs.best_params_} for {model_key} \n
                train_model_score: {train_model_score} \n
                test model score: {test_model_score}"""
            )
        
        print(report)
        return report   

    except Exception as e:
        raise CustomException(e, sys)
    
"""def hyperparameter_tuning(X_train, y_train, X_test, x_test, models, param_grid):
    try:
        logging.info("Initiate Hypyer Parameter Tuning")
        report = {}
        for i in range(len(list(models.values()))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            grid_search = GridSearchCV(model, param_grid=param_grid[model_name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            grid_search_score_on_training_data = grid_search.predict(X_train)
            grid_search_score_on_testing_data = grid_search.predict(X_test)
            report[model_name] = {
                'best_params_': grid_search.best_params_,
                'best_score_': grid_search.best_score_,
                'training_score': grid_search_score_on_training_data,
                'testing_score': grid_search_score_on_testing_data
            }

            logging.info(f"{model_name} completed with {grid_search.best_score_}")
        
        logging.info(f"Hyperparameter Tuning Completed...... f{report}")
        return report

    except Exception as e:
        raise CustomException(e, sys)"""