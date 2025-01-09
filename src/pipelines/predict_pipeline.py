import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object # to load our pickle file
from datetime import datetime,date,time

#First Class -> has the init function without nothing, 
class PredictPipeline:
    def __init__(self):
        pass

    #will simply do prediction
    # two pckle files we have currently, preprocessor and  model
    #Prediction - 1: Gives prediction of 1 input
    def predict(self,features):
        try:
            print("########Step-4- Extension - Inside predict_pipeline.py")
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            print("########Step-4- Data Transformation Triggered through preprocessor.pkl")
            #load_obect we will craete, will load the pickle file
            model = load_object(file_path = model_path) #should be created in utils
            preprocessor = load_object(file_path = preprocessor_path)
            #print("Exploring the preprocessor object")
            #print(type(preprocessor))
            #print(preprocessor.transformers)
            data_final_to_pred = preprocessor.transform(features)
            print("########Step-4 - End of Data Transformation#########################")
            print("########Step-5 - Printing the Model Object")
            print(type(model))
            print("########Step-6 - Printing Final DataFrame for Prediction")
            print(data_final_to_pred.head(2))
            print("########Step-7 - Printing DataFrame Info")
            print(data_final_to_pred.info())
            preds = model.predict(data_final_to_pred) #is typically a 1D array containing labels or predicted values.
            pred_proba = model.predict_proba(data_final_to_pred) #s a 2D array containing probabilities, with rows corresponding to samples and columns to class probabilities
            return preds,pred_proba

        except Exception as e:
            raise CustomException(e,sys)
        
           

#Second Class -> Responsible for matching all the input we are passing in the html to the backend
class CustomData:
    def __init__( self,           
            date,
            transport_company,
            relation_name,
            relation_code,
            trip_nr,
            order_number,
            external_reference,
            order_type,
            customer,
            planned_date,
            planned_time
            #arrival_date,
            #arrival_time
            ):

            #Creating variable using self, the values are coming from web app agianst the respective variable
            self.date = date
            self.transport_company = transport_company
            self.relation_name = relation_name
            self.relation_code = relation_code
            self.trip_nr = trip_nr
            self.order_number = order_number
            self.external_reference = external_reference
            self.order_type = order_type
            self.customer = customer
            self.planned_date = planned_date
            self.planned_time = planned_time
            #self.arrival_date = arrival_date
            #self.arrival_time = arrival_time

    #It will basically return all our input in the form of a dataframe
    #From my web appplication , will get mapped to a datafram
    #could have been done in app.py but due to modularisation it is showed here 
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Date": [self.date],
                "Transport Company": [self.transport_company],
                "RelationName":[self.relation_name],
                "RelationCode":[self.relation_code],
                "Trip Nr":[self.trip_nr],
                "Order Number":[self.order_number],
                "External reference":[self.external_reference],
                "Order type":[self.order_type],
                "Customer":[self.customer],
                "Planned Date":[self.planned_date],
                "Planned Time":[self.planned_time]
                #"Arrival Date":[self.arrival_date],
                #"Arrival Time":[self.arrival_time],
                }
            final_df = pd.DataFrame(custom_data_input_dict)
            #print("In predict_pipeline",final_df)
            print("#Step 1: Extension - In predict_pipeline - Please find dataframe info")
            print(final_df.info())
            return final_df

        except Exception as e:
            raise CustomException(e,sys)