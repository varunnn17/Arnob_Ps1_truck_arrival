#Read the data and split the data into train and test
import os 
import sys # as we will use customer exception
from src.exception import CustomException
#from src.logger import logging
import logging # Changed by Arnob on 01-01-2025
import pandas as pd 

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#Will uncomment later - Arnob - 01-01-2025


from src.components.data_trans_3 import DataTransformation3  
from src.components.data_trans_3 import DataTransformationConfig3 
#dataclass is a decorator, if you are only defining variables then you can use data class, but if you have other 
#function, need init

#Will uncomment later - Arnob - 01-01-2025
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts","train.csv") #all data will be saved in artifacts path, filename train.csv
    #os.path dynamically adjust / or \ based on os
    #output => artifacts\train.csv
    test_data_path: str=os.path.join("artifacts","test.csv") #all data will be saved in artifacts path
    raw_data_path: str=os.path.join("artifacts","PS_1_TruckArrival_Class_Dataset_withActualColumns.csv") #all data will be saved in artifacts path

    #we can directly define the class variable without using init

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() 
        # When we call the class DataIngestionConfig the above 3 path will be saved in the path variables, they will
        #basically have the sub objects

    #Function 1
    def initiate_data_ingestion(self):
        #mongo db client can be present in utils
        logging.info("Entered the data ingestion method or component")
        try:
            #Step - 1 = Reading the dataset
            df = pd.read_csv(r"notebook\data\PS_1_TruckArrival_Class_Dataset_withActualColumns.csv") #Here we can read from anywhere
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True) #Getting the directory name and not deleting
            # if its existing

            #Step -2 = Converted the Raw data path into csv file
            df.to_csv(self.ingestion_config.raw_data_path,index = False,header = True)

            logging.info("Train test split initiated")

            #Step -3 = Splitting the Train and Test data
            train_set,test_set = train_test_split(df,test_size = 0.2,random_state = 42)

            #Step -4 = Saving the train and test data
            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header = True)

            logging.info("Ingestion of the data is completed")

            return(
                #Step -5 = We pass the train data and test data path to the next step i.e Data Transformation
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

#Data-Ingestion 1: THis is the starting point as it has the main method, it then calls 
#initiate_data_ingestion() - To divide into train and test
if __name__ == "__main__":
    print("########1- Starting Journery from data_ingestion.py")
    obj=DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    print("Splitting the Data into Test and Train, and sending it for Data Transformation")

#Data-Ingestion 2: THis creates an object of the DataTransformation Class present in data_trans_3
#and passes the train_data_path and test_data_path

#Data Transformation => Returns back the Train and Test files after transformation
    data_transformation3 = DataTransformation3() #It will call this -> self.data_transformation_config
    print("########2- Starting with Data Transformation")
    
    #Initating data transformation phase 1
    train_df_final,test_df_final,train_data_final_path,test_data_final_path,_= data_transformation3.initiate_data_transformation2(train_data_path,test_data_path)

######################################################################

   #Data-Ingestion 3: Finally object of the ModelTrainer() class is created
   #and passes the train_df_final and train_df_final , these are the final dataframes which has the
   #i/p features with Transformed columns
   #initiate_model_trainer() -> this function creates the model and stores it 
   #Final Scores are returned which we can print
    modeltrainer = ModelTrainer()
    print("########3- Sending the final dataframes for model building after transformation")
    print("Sample of train_dataframe")
    print(train_df_final.head(2))

    print("########4- Printing the scores in CV fold and Holdout Set")
    model_scores,model_scores_test = modeltrainer.initiate_model_trainer(train_df_final,test_df_final)


 
