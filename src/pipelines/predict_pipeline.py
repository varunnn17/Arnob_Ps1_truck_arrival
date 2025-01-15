import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object, test_on_unseen_data # to load our pickle file
from datetime import datetime,date,time
from src.components.model_trainer import ModelTrainer

#First Class -> has the init function without nothing, 
class PredictPipeline:
    def __init__(self):
        pass

    #will simply do prediction
    # two pckle files we have currently, preprocessor and  model
    #Prediction - 1: Gives prediction of 1 input
    def predict(self, file_path, processed_file_path):
        try:
            print("########Step-4- Extension - Inside predict_pipeline.py")
            final_best_model_path = 'artifacts/final_best_model.pkl'
            model_2_path = 'artifacts/model_2.pkl'
            model_3_path = 'artifacts/model_3.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            print("########Step-4- Data Transformation Triggered through preprocessor.pkl")
            #load_obect we will craete, will load the pickle file
            final_best_model = load_object(file_path = final_best_model_path) #should be created in utils
            model_2 = load_object(file_path = model_2_path)
            model_3 = load_object(file_path = model_3_path)
            preprocessor = load_object(file_path = preprocessor_path)
            input_data = pd.read_csv(file_path)
            # Save the original data for reference
            original_data = input_data.copy()

            data_final_to_pred = preprocessor.transform(input_data)
            print("########Step-4 - End of Data Transformation#########################")
            print("########Step-5 - Printing the Model Object")
            print("########Step-6 - Printing Final DataFrame for Prediction")
            print(data_final_to_pred.head(2))
            print("########Step-7 - Printing DataFrame Info")
            print(data_final_to_pred.info())
            _, weighted_probs = test_on_unseen_data(final_best_model, model_2, model_3, data_final_to_pred)
            input_data['Delay_Percentage'] = weighted_probs * 100
            input_data['Status'] = (input_data['Delay_Percentage']>=50).map({True: 'Delayed', False: 'On Time'})
            columns_to_append = input_data[['Delay_Percentage', 'Status']]
            final_df = pd.concat([original_data, columns_to_append], axis=1)
            final_df.to_csv(processed_file_path, index=False)
            
            

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