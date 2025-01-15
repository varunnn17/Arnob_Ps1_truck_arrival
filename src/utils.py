import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException



# src/utils.py

from sklearn.utils.validation import _deprecate_positional_args



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def test_on_unseen_data(final_best_model, model_2, model_3, unseen):
    final_best_probs = final_best_model.predict_proba(unseen)[:, 1]  # Probabilities for class 1
    model_2_probs = model_2.predict_proba(unseen)[:, 1]  # Probabilities for class 1
    model_3_probs = model_3.predict_proba(unseen)[:, 1]
    final_best_weight = 2
    model_2_weight = 1  # Higher weight for XGBoost
    model_3_weight = 1

    # Combine the probabilities using weighted average
    weighted_probs = (final_best_weight * final_best_probs + model_2_weight * model_2_probs + model_3_weight*model_3_probs) / (final_best_weight + model_2_weight + model_3_weight)

    # Convert probabilities to final predictions (threshold at 0.5)
    y_pred = (weighted_probs >= 0.49).astype(int)
    return y_pred, weighted_probs 