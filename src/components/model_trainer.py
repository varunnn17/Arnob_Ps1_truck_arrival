import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from src.exception import CustomException
import logging
from src.utils import save_object, load_object, test_on_unseen_data
from src.utils import evaluate_models
from dataclasses import dataclass 
import sys
import os
import pandas as pd


@dataclass
class ModelTrainerConfig: 
    #This ModelTrainerConfig class holds a configuration for the model trainer.
    #after creating model ,will want to save my pickle file
    trained_final_best_model_file_path = os.path.join("artifacts","final_best_model.pkl")
    trained_model_2_file_path = os.path.join("artifacts","model_2.pkl")
    trained_model_3_file_path = os.path.join("artifacts","model_3.pkl")


class ModelTrainer:
    #Step 1 -Main Class uses the configuration defined in ModelTrainerConfig to train and save
    #the models

    #Step 2 - #the __init__ method initialises the model_trainer_config attribute
        #with an instance of ModelTrainerConfig
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    

            #Step 3 - Divides the train and test array into predictor and target variables
    def initiate_model_trainer(self,train_df_final):
        try:
            logging.info("Split training and test input data")
            

            target_column_name = "TargetVariable"
            print("shape of train_df_final before splitting to X_train and y_train", train_df_final.shape)
            X_train = train_df_final.drop(columns = [target_column_name],axis = 1)
            y_train = train_df_final[target_column_name]
            print("Checking for null values in the training data")
            print(X_train.isna().sum())
            X_train.to_csv("artifacts/X_train.csv", index=False)

            # X_test = test_df_final.drop(columns = [target_column_name],axis = 1)
            # y_test = test_df_final[target_column_name]

            rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
            xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42, scale_pos_weight=3)

            #xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42, scale_pos_weight=3)
            gb_model = GradientBoostingClassifier(random_state=42)


            # Hyperparameter grids for tuning
            rf_param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }

            xgb_param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }

            gb_param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.05],
                'max_depth': [3, 5]
            }

            #Print shape
            print("Shape of dataframes")
            print(X_train.shape)
            print(y_train.shape)

            # Set up cross-validation strategy
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            #Hyperparameter tuning on Random Forest, XG boost, Gradient Boost
            rf_grid_recall = GridSearchCV(rf_model, rf_param_grid, cv=cv_strategy, scoring='recall', n_jobs=-1)
            xgb_grid_recall = GridSearchCV(xgb_model, xgb_param_grid, cv=cv_strategy, scoring='recall', n_jobs=-1)
            gb_grid_recall = GridSearchCV(gb_model, gb_param_grid, cv=cv_strategy, scoring='recall', n_jobs=-1)

            # Fit abobe 3 models using cross-validation and find the best hyperparameters
            rf_grid_recall.fit(X_train, y_train)
            xgb_grid_recall.fit(X_train, y_train)
            gb_grid_recall.fit(X_train, y_train)

            # Get best models for XGB boost
            best_rf_model_on_recall = rf_grid_recall.best_estimator_
            best_xgb_model_on_recall = xgb_grid_recall.best_estimator_
            best_gb_model_on_recall = gb_grid_recall.best_estimator_

            #Model Scores on Training Holdout set
            #### Metric Evaluation of the 3 models selected on Recall####
            #Calculate Metrics - Recall,Precision,Accuracy#
            rf_cv_score_recall = cross_val_score(best_rf_model_on_recall, X_train, y_train, cv=cv_strategy, scoring='recall').mean()
            xgb_cv_score_recall = cross_val_score(best_xgb_model_on_recall, X_train, y_train, cv=cv_strategy, scoring='recall').mean()
            gb_cv_score_recall = cross_val_score(best_gb_model_on_recall, X_train, y_train, cv=cv_strategy, scoring='recall').mean()


            #### Metric Evaluation of the 3 models selected on Precision####
            rf_cv_score_precision = cross_val_score(best_rf_model_on_recall, X_train, y_train, cv=cv_strategy, scoring='precision').mean()
            xgb_cv_score_precision = cross_val_score(best_xgb_model_on_recall, X_train, y_train, cv=cv_strategy, scoring='precision').mean()
            gb_cv_score_precision = cross_val_score(best_gb_model_on_recall, X_train, y_train, cv=cv_strategy, scoring='precision').mean()


            #### Metric Evaluation of the 3 models selected on Accuracy####
            rf_cv_score_accuracy = cross_val_score(best_rf_model_on_recall, X_train, y_train, cv=cv_strategy, scoring='accuracy').mean()
            xgb_cv_score_accuracy = cross_val_score(best_xgb_model_on_recall, X_train, y_train, cv=cv_strategy, scoring='accuracy').mean()
            gb_cv_score_accuracy = cross_val_score(best_gb_model_on_recall, X_train, y_train, cv=cv_strategy, scoring='accuracy').mean()

            # Select the best model based on cross-validation score
            model_scores = {
                'Random Forest Accuracy': rf_cv_score_accuracy,
                'Random Forest Precision': rf_cv_score_precision,
                'Random Forest Recall': rf_cv_score_recall,
                'XGBoost Accuracy': xgb_cv_score_accuracy,
                'XGBoost Precision': xgb_cv_score_precision,
                'XGBoost Recall': xgb_cv_score_recall,
                'Gradient Boosting Accuracy': gb_cv_score_accuracy,
                'Gradient Boosting Precision': gb_cv_score_precision,
                'Gradient Boosting Recall': gb_cv_score_recall
            }
                        
            return model_scores, best_rf_model_on_recall, best_xgb_model_on_recall, best_gb_model_on_recall #,model_scores_test#
        except Exception as e:
            raise CustomException(e, sys)
    
    def find_best_model(self, model_scores):
        try:
            logging.info("Selecting the best model to give the highest weightage")
            # model_scores, _, _, _ = self.initiate_model_trainer(train_df_final)
            # Weights for Accuracy, Precision, and Recall
            weights = {'Accuracy': 1, 'Precision': 2, 'Recall': 2}
            weighted_scores = {}
            for model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
                #extract individual scores for the model
                accuracy = model_scores.get(f'{model_name} Accuracy', 0)
                precision = model_scores.get(f'{model_name} Precision', 0)
                recall = model_scores.get(f'{model_name} Recall', 0)

                weighted_score = (
                      weights['Accuracy'] * accuracy +
                      weights['Precision'] * precision +
                      weights['Recall'] * recall)
    
                # Store the weighted score for the model
                weighted_scores[model_name] = weighted_score
            best_model_name = max(weighted_scores, key=weighted_scores.get)
            best_score = weighted_scores[best_model_name]
            return best_model_name, best_score

        except Exception as e:
            raise CustomException(e, sys)
    
    def all_models_used(self, train_df_final):
        try:
            model_scores, best_rf_model_on_recall, best_xgb_model_on_recall, best_gb_model_on_recall = self.initiate_model_trainer(train_df_final)
            best_model_name, _ = self.find_best_model(model_scores)

            if best_model_name == "XGBoost":
                final_best_model = best_xgb_model_on_recall
                model_2 = best_rf_model_on_recall
                model_3 = best_gb_model_on_recall
            elif best_model_name == "Random Forest":
                final_best_model = best_rf_model_on_recall
                model_2 = best_xgb_model_on_recall
                model_3 = best_gb_model_on_recall
            elif best_model_name == "Gradient Boosting":
                final_best_model = best_gb_model_on_recall
                model_2 = best_rf_model_on_recall
                model_3 = best_xgb_model_on_recall
            save_object(
                file_path = self.model_trainer_config.trained_final_best_model_file_path,
                obj = final_best_model
            )
            save_object(
                file_path = self.model_trainer_config.trained_model_2_file_path,
                obj = model_2
            )
            save_object(
                file_path = self.model_trainer_config.trained_model_3_file_path,
                obj = model_3
            )
        except Exception as e:
            raise CustomException(e, sys)
        
    def predict_for_validation_set(self, test_df_final):
        target_column_name = "TargetVariable"
        X_test = test_df_final.drop(columns = [target_column_name],axis = 1)
        y_test = test_df_final[target_column_name]
        final_best_model_path = 'artifacts/final_best_model.pkl'
        model_2_path = 'artifacts/model_2.pkl'
        model_3_path = 'artifacts/model_3.pkl'
        final_best_model = load_object(file_path = final_best_model_path) #should be created in utils
        model_2 = load_object(file_path = model_2_path)
        model_3 = load_object(file_path = model_3_path)
        y_pred, weighted_probs = test_on_unseen_data(final_best_model, model_2, model_3, X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        # Print the results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")



        # Print the probabilities of the first 5 predictions
        print("\nPredicted Probabilities for the first 5 test samples:")
        print(weighted_probs[:5])
        
        
    
    


            

            