import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from src.exception import CustomException
import logging
from src.utils import save_object
from src.utils import evaluate_models
from dataclasses import dataclass 
import sys
import os
import pandas as pd

@dataclass
class ModelTrainerConfig: 
    #This ModelTrainerConfig class holds a configuration for the model trainer.
    #after creating model ,will want to save my pickle file
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    #Step 1 -Main Class uses the configuration defined in ModelTrainerConfig to train and save
    #the models

    #Step 2 - #the __init__ method initialises the model_trainer_config attribute
        #with an instance of ModelTrainerConfig
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() 


            #Step 3 - Divides the train and test array into predictor and target variables
    def initiate_model_trainer(self,train_df_final,test_df_final):
        try:
            logging.info("Split training and test input data")
            

            target_column_name = "TargetVariable"
            X_train = train_df_final.drop(columns = [target_column_name],axis = 1)
            y_train = train_df_final[target_column_name]

            X_test = test_df_final.drop(columns = [target_column_name],axis = 1)
            y_test = test_df_final[target_column_name]

            rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

            # Hyperparameter grids for tuning
            rf_param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }

            #Print shape
            print("Shape of dataframes")
            print(X_train.shape)
            print(y_train.shape)


         
            
            

            
            # Set up cross-validation strategy
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            

            #Hyperparameter tuning
            rf_grid_recall = GridSearchCV(rf_model, rf_param_grid, cv=cv_strategy, scoring='recall', n_jobs=-1)

            # Fit models using cross-validation and find the best hyperparameters
            rf_grid_recall.fit(X_train, y_train)

            # Get best models for XGB boost
            best_rf_model_on_recall = rf_grid_recall.best_estimator_

            #Model Scores on Training Holdout set
            #Calculate Metrics - Recall,Precision,Accuracy#
            rf_cv_score_recall = cross_val_score(best_rf_model_on_recall, X_train, y_train, cv=cv_strategy, scoring='recall').mean()
            rf_cv_score_precision = cross_val_score(best_rf_model_on_recall, X_train, y_train, cv=cv_strategy, scoring='precision').mean()
            rf_cv_score_accuracy = cross_val_score(best_rf_model_on_recall, X_train, y_train, cv=cv_strategy, scoring='accuracy').mean()

            # Select the best model based on cross-validation score
            model_scores = {
                'Random Forest Accuracy': rf_cv_score_accuracy,
                'Random Forest Precision': rf_cv_score_precision,
                'Random Forest Recall': rf_cv_score_recall,
                
            }
            #Model Scores on Training Holdout set
            #print(model_scores)
            print("Scores on Validation(which is taken from the TrainingDataFrame)",model_scores)

            #####################ON UNSEEN DATA#####################
            #Model Scores on TEST set
            #Calculate Metrics - Recall,Precision,Accuracy#
            rf_cv_score_recall_test = cross_val_score(best_rf_model_on_recall, X_test, y_test, cv=cv_strategy, scoring='recall').mean()
            rf_cv_score_precision_test = cross_val_score(best_rf_model_on_recall, X_test, y_test, cv=cv_strategy, scoring='precision').mean()
            rf_cv_score_accuracy_test = cross_val_score(best_rf_model_on_recall, X_test, y_test, cv=cv_strategy, scoring='accuracy').mean()

            # Select the best model based on cross-validation score
            model_scores_test = {
                'Random Forest Accuracy': rf_cv_score_accuracy_test,
                'Random Forest Precision': rf_cv_score_precision_test,
                'Random Forest Recall': rf_cv_score_recall_test,
                
            }
            #Model Scores on Test  set
            print("Scores on Holdout or Test Set",model_scores_test)
#Step 10 - Save the model part - basically dumping the best model

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_rf_model_on_recall
            )
            return model_scores,model_scores_test
        except Exception as e:
            raise CustomException(e, sys)

            