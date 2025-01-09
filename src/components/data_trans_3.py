from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object # jsut used for saving the pickle fil
import sys # provides access to system specific parameter and funcs, used to interact with Python runtime env
from dataclasses import dataclass
import os

#print("version in data_trans_3")
#print("numpy",np.__version__)
#print("pandas",pd.__version__)
#print("########Step-4-Inside Data Transformation")


class CompleteTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to derive the target variable.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        # Create a copy to avoid modifying the original data
        df = df.copy()
        try:
################################################
#3.0 - Transforming the date and time columns to form PlannedDateTime & ArrivedDateTime
            print("Shape Before Transformation - 3.0 - Forming PlannedDateTime & ArrivedDateTime:",df.shape)
            df["PlannedDateTime"] = pd.to_datetime(
                df["Planned Date"] + " " + df["Planned Time"],
                format="%d/%m/%Y %I:%M:%S %p",
            )
            #df["ArrivedDateTime"] = pd.to_datetime(
                #df["Arrival Date"] + " " + df["Arrival Time"],
                #format="%d/%m/%Y %I:%M:%S %p",
            #)
            print("Shape After Transformation - 3.0 - Forming PlannedDateTime & ArrivedDateTime:",df.shape)
################################################
#3.1 - Label Encoding CarrierID,RelationID,CustomerID            
            # Initialize LabelEncoder
            label_encoder = LabelEncoder()
            print("Shape Before Transformation - 3.1 - Label Encoding CarrierID,RelationID,CustomerID:",df.shape)
            # Transformation 2:Perform label encoding on Transport Company,RelationCode,Customer
            logging.info("Encoding categorical columns with LabelEncoder.")
            df["CarrierID"] = label_encoder.fit_transform(df["Transport Company"])
            df["RelationID"] = label_encoder.fit_transform(df["RelationCode"])
            df["CustomerID"] = label_encoder.fit_transform(df["Customer"])
            print("Shape After Transformation - 3.1 - Label Encoding CarrierID,RelationID,CustomerID:",df.shape)
################################################
#3.2 - Adding derived column,NumberOfOrders against each TripId 
            # Transformation 3:Add derived column,NumberOfOrders against each TripId
            print("Shape Before Transformation - 3.2 - Adding derived column,NumberOfOrders against each TripId:",df.shape)
            logging.info("Adding the 'NumberOfOrders' column.")
            df["NumberOfOrders"] = 1
            print("Shape After Transformation - 3.2 - Adding derived column,NumberOfOrders against each TripId:",df.shape)
################################################
#3.3 - Selecting only Required Columns
            print("Shape Before Transformation - 3.3 - Selecting only Required Columns:",df.shape)
            df = df[
                [
                    "Date",
                    "CarrierID",
                    "RelationID",
                    "Trip Nr",
                    "Order type",
                    "CustomerID",
                    "PlannedDateTime",
                    #"ArrivedDateTime",
                    "NumberOfOrders",
                ]
            ]
            print("Shape After Transformation - 3.3 - Selecting only Required Columns:",df.shape)

                     
################################################
#3.5 - Removing Records with PlannedDateTime as NULL
    
            print("Shape Before Transformation - 3.5 - Remove Null Planned Time:",df.shape)
            df = df.dropna(subset=["PlannedDateTime"])
            logging.info(f"Records with PlannedDateTime Removed")
            print("Shape After Transformation - 3.5 - Remove Null Planned Time:",df.shape)
           
################################################
#3.6 - Removing Records with Duplicate Trip Nos
            print("Shape Before Transformation - 3.6 - Remove Duplicate Trip Nos:",df.shape)
            trip_count_multiple = df.groupby("Trip Nr")["Trip Nr"].count()
            trips_with_mult_counts = trip_count_multiple[trip_count_multiple > 1].index
            df_filtered = df[df["Trip Nr"].isin(trips_with_mult_counts)]
            df = df.drop(df_filtered.index)  # Removing duplicates
            logging.info("Duplicate trips removed successfully.")
            print("Shape After Transformation - 3.6 - Remove Duplicate Trip Nos:",df.shape)
            
################################################
#3.7 - Deriving TimeBasedFeatures from PlannedDateTime
            print("Shape Before Transformation - 3.7 - Deriving TimeBasedFeatures from PlannedDateTime:",df.shape)
            df['Planned_Hour'] = df['PlannedDateTime'].dt.hour
            df['Planned_Day'] = df['PlannedDateTime'].dt.day
            df['Planned_Weekday'] = df['PlannedDateTime'].dt.weekday  # Monday=0, Sunday=6
            df['Planned_Month'] = df['PlannedDateTime'].dt.month
            df['Planned_Year'] = df['PlannedDateTime'].dt.year
            df['Planned_Week'] = df['PlannedDateTime'].dt.isocalendar().week  # Weeks 1-52 in a year
        # Add IsWeekend column: 1 if weekend (Saturday=5, Sunday=6), 0 otherwise
            df['IsWeekend'] = df['Planned_Weekday'].apply(lambda x: 1 if x >= 5 else 0)
            logging.info("3.7 TimeBasedFeatures created successfully.")
            print("Shape After Transformation - 3.7 - Deriving TimeBasedFeatures from PlannedDateTime:",df.shape)
            
################################################
#3.8 - Deriving Cyclical & Frequency Features from columns created in 3.7
            print("Shape Before Transformation - 3.8 - Deriving Cyclical & Frequency Features from columns created in 3.7:",df.shape)
            ##########
            #1PlannedHour - Cyclic and Frequency Encoding
            ##########
            # Hourly Cyclic Encoding (0-23)
            df['Planned_Hour_sin'] = np.sin(2 * np.pi * df['Planned_Hour'] / 24)
            df['Planned_Hour_cos'] = np.cos(2 * np.pi * df['Planned_Hour'] / 24)

            def time_of_day(hour):
                if 6 <= hour < 12:
                    return 'Morning'
                elif 12 <= hour < 18:
                    return 'Afternoon'
                elif 18 <= hour < 24:
                    return 'Evening'
                else:
                    return 'Night'

            df['Planned_TimeOfDay'] = df['Planned_Hour'].apply(time_of_day)

            # Step 2: Apply one-hot encoding to Planned_TimeOfDay
            time_of_day_encoded = pd.get_dummies(df['Planned_TimeOfDay'], prefix='Planned_TimeOfDay', drop_first=True).astype(int)  # Convert to integer type explicitly      
            # Step 3: Concatenate the original dataframe with the one-hot encoded columns
            df = pd.concat([df, time_of_day_encoded], axis=1)
            columns_to_drop_1 = ["Planned_TimeOfDay"]
            df = df.drop(columns=columns_to_drop_1)
            

            # Frequency Encoding #Based on the No Of Trips
            freq_map_plannedhour = df['Planned_Hour'].value_counts(normalize=True).to_dict()
            df['Planned_Hour_freq'] = df['Planned_Hour'].map(freq_map_plannedhour)

            ##########
            #2PlannedDay - Cyclic and Frequency Encoding
            ##########
            #Cyclic
            df['Planned_Day_sin'] = np.sin(2 * np.pi * df['Planned_Day'] / 29)
            df['Planned_Day_cos'] = np.cos(2 * np.pi * df['Planned_Day'] / 29)
            # Frequency Encoding #Based on the No Of Trips
            freq_map_plannedday = df['Planned_Day'].value_counts(normalize=True).to_dict()
            df['Planned_Day_freq'] = df['Planned_Day'].map(freq_map_plannedday)

            ##########
            #3PlannedWeekday - Cyclic and Frequency Encoding
            ##########
            #Cyclical Encoding
            df['Planned_Weekday_sin'] = np.sin(2 * np.pi * df['Planned_Weekday'] / 7)
            df['Planned_Weekday_cos'] = np.cos(2 * np.pi * df['Planned_Weekday'] / 7)


            # Frequency Encoding #Based on the No Of Trips
            freq_map = df['Planned_Weekday'].value_counts(normalize=True).to_dict()
            df['Planned_Weekday_freq'] = df['Planned_Weekday'].map(freq_map)

            ##########
            #4PlannedMonth - Cyclic and Frequency Encoding
            ##########
            #Cyclical Encoding
            df['Planned_Month_sin'] = np.sin(2 * np.pi * df['Planned_Month'] / 12)
            df['Planned_Month_cos'] = np.cos(2 * np.pi * df['Planned_Month'] / 12)


            # Frequency Encoding #Based on the No Of Trips
            freq_map_plannedmonth = df['Planned_Month'].value_counts(normalize=True).to_dict()
            df['Planned_Month_freq'] = df['Planned_Month'].map(freq_map_plannedmonth)

            ##########
            #5PlannedYear - Cyclic and Frequency Encoding
            ##########
            df['Planned_Year_sin'] = np.sin(2 * np.pi * df['Planned_Year'] / 12)
            df['Planned_Year_cos'] = np.cos(2 * np.pi * df['Planned_Year'] / 12)


            # Frequency Encoding #Based on the No Of Trips
            freq_map_plannedyear = df['Planned_Year'].value_counts(normalize=True).to_dict()
            df['Planned_Year_freq'] = df['Planned_Year'].map(freq_map_plannedyear)

            ##########
            #6PlannedWeek - Cyclic and Frequency Encoding
            ##########
            #Cyclical Encoding
            df['Planned_Week_sin'] = np.sin(2 * np.pi * df['Planned_Week'] / 12)
            df['Planned_Week_cos'] = np.cos(2 * np.pi * df['Planned_Week'] / 12)


            # Frequency Encoding #Based on the No Of Trips
            freq_map_plannedweek = df['Planned_Week'].value_counts(normalize=True).to_dict()
            df['Planned_Week_freq'] = df['Planned_Week'].map(freq_map_plannedweek)

            ##########
            #7IsWeekend - Frequency Encoding
            ##########
            freq_map_isweekend = df['IsWeekend'].value_counts(normalize=True).to_dict()
            df['IsWeekend_freq'] = df['IsWeekend'].map(freq_map_isweekend)
            logging.info("3.8 Cyclical & Frequency Based Features for DateTime Based columns created")
            print("Shape After Transformation - 3.8 - Deriving Cyclical & Frequency Features from columns created in 3.7:",df.shape)
        
################################################
#3.9 - Frequency Encoding for CustomerID,RelationID,CarrierID,OrderType
            print("Shape before Transformation - 3.9 - Frequency Encoding for Ids:",df.shape)
            # Step - 1 Frequency Encoding For CustomerID,
            freq_map_custid = df['CustomerID'].value_counts(normalize=True).to_dict()
            df['CustomerID_freq'] = df['CustomerID'].map(freq_map_custid)


            # Step - 2 Frequency Encoding For RelationID,
            freq_map_relationid = df['RelationID'].value_counts(normalize=True).to_dict()
            df['RelationID_freq'] = df['RelationID'].map(freq_map_relationid)


            # Step - 3 Frequency Encoding For CarrierID,
            freq_map_carrierid = df['CarrierID'].value_counts(normalize=True).to_dict()
            df['CarrierID_freq'] = df['CarrierID'].map(freq_map_carrierid)

            # Step - 4 Frequency Encoding For Order type,
            freq_map_ordertype = df['Order type'].value_counts(normalize=True).to_dict()
            df['OrderType_freq'] = df['Order type'].map(freq_map_ordertype)
            logging.info("3.9 - Frequency Encoding for CustomerID,RelationID,CarrierID,OrderType created successfully.")
            print("Shape After Transformation - 3.9 - Frequency Encoding for Ids:",df.shape)
            
################################################
#3.10 - Interaction Based Features
            print("Shape Before Transformation - 3.10 - Interaction Based Features:",df.shape)
            df['Carrier_Relation_Interaction'] = df['CarrierID_freq'] * df['RelationID_freq']
            df['Carrier_Relation_Customer_Interaction'] = df['CarrierID_freq'] * df['RelationID_freq'] * df['CustomerID_freq']
            df['Customer_Relation_Interaction'] = df['CustomerID_freq'] * df['RelationID_freq']
            df['Customer_Carrier_Interaction'] = df['CustomerID_freq'] * df['CarrierID_freq']
            logging.info("3.10 - Interaction Based Features created successfully.")
            print("Shape After Transformation - 3.10 - Interaction Based Features:",df.shape)
            
        
################################################
#3.11 - Feature Based on CarrierId & RelationId Aggregation of Number of Orders
            print("Shape Before Transformation - 3.11 - Feature Based on CarrierId & RelationId Aggregation of Number of Orders:",df.shape)
            # Calculate the frequency of each CarrierID and RelationID combination
            order_frequency = df.groupby(['CarrierID', 'RelationID'])['NumberOfOrders'].sum() / len(df)

            # Convert the result to a DataFrame
            order_frequency = order_frequency.rename('Carrier_Relation_Order_Frequency').reset_index()

            # Merge the frequency encoding back to the original dataframe
            df = df.merge(order_frequency, on=['CarrierID', 'RelationID'], how='left')
            logging.info("3.11 - Feature Based on CarrierId & RelationId Aggregation of Number of Orders created successfully.")
            print("Shape After Transformation - 3.11 - Feature Based on CarrierId & RelationId Aggregation of Number of Orders:",df.shape)
        
################################################
#3.12 - Ranking Carriers based on Trip Freq
            print("Shape Before Transformation - 3.12 - Ranking Carriers based on Trip Freq:",df.shape)
            # Define the thresholds for ranking based on the Carried_Ord_Freq
            high_threshold = df['CarrierID_freq'].quantile(0.67)  # Top 33% (High)
            medium_threshold = df['CarrierID_freq'].quantile(0.33)  # Middle 33% (Medium)

            # Create a function to rank the carriers based on Carried_Ord_Freq
            def rank_carriers(freq):
                if freq >= high_threshold:
                    return 3  # High
                elif freq >= medium_threshold:
                    return 2  # Medium
                else:
                    return 1  # Low

            # Apply the rank function to the Carried_Ord_Freq column
            df['CarrierRank'] = df['CarrierID_freq'].apply(rank_carriers)
            logging.info("3.12 - Feature Based on ranking of carriers created successfully.")
            print("Shape After Transformation - 3.12 - Ranking Carriers based on Trip Freq:",df.shape)

################################################
#3.13 - Feature based on OrderDensity Per Hour
            print("Shape Before Transformation - 3.13 - Feature based on OrderDensity Per Hour:",df.shape)
            # Calculate Order Density by Hour
            order_density_hour = df['Planned_Hour'].value_counts(normalize=True).rename('Order_Density_Per_Hour')


            # Merge back into the original dataframe if needed
            df = df.merge(order_density_hour, left_on='Planned_Hour', right_index=True, how='left')
            logging.info("3.13 - Feature based on OrderDensity Per Hour created successfully.")
            print("Shape After Transformation - 3.13 - Feature based on OrderDensity Per Hour:",df.shape)
            
        
################################################
#3.14 - Dropping all ID based columns
            print("Shape before Transformation - 3.14 - Dropping All ID Based column:",df.shape)
            # Calculate Order Density by Hour
            columns_to_drop = ["Date","CarrierID","RelationID","Trip Nr","Order type","CustomerID","PlannedDateTime","Planned_Week"]
            df = df.drop(columns=columns_to_drop)
            logging.info("3.14 - Dropped all ID based columns successfully.")
            print("Shape before Transformation - 3.14 - Dropping All ID Based column:",df.shape)
################################################
#3.15 - Selecting the final features that have been shortlisted manually
            print("Shape before Feature Selection - 3.15 - :",df.shape)
            # Calculate Order Density by Hour
            selected_features = [
                                "OrderType_freq",
                                "CustomerID_freq",
                                "Customer_Carrier_Interaction",
                                "Carrier_Relation_Interaction",
                                "CarrierID_freq",
                                "Planned_Hour_cos",
                                "Planned_Hour",
                                "Planned_Month_sin",
                                "Planned_Year",
                                "Carrier_Relation_Order_Frequency",
                                "Planned_Hour_freq",
                                "Carrier_Relation_Customer_Interaction",
                                "RelationID_freq",
                                "Planned_Weekday",
                                "Planned_Month",
                                "Customer_Relation_Interaction",
                                #"Planned_TimeOfDay_Morning",
                                "Planned_Weekday_cos",
                                "Planned_Day_sin",
                                "Planned_Week_freq",
                                "Planned_Month_freq",
                                "Planned_Week_cos"
                                #"TargetVariable"
                            ]

            # Select the specified columns
            df = df[selected_features]                          
            logging.info("3.15 - Features successfully.")
            print("Shape before Feature Selection - 3.15 - :",df.shape)
            return df
        except Exception as e:
            raise CustomException(e, sys)
    
    def fit_transform(self, X, y=None):
        """
        Combines fit and transform steps.
        """
        return self.fit(X, y).transform(X) # Return the DataFrame with the new columns
        

#We just want the input to the DataTransformationConfig
@dataclass
class DataTransformationConfig3:
    train_data_final_path: str=os.path.join("artifacts","train_final.csv") #all data will be saved in artifacts path, filename train.csv
    #os.path dynamically adjust / or \ based on os
    #output => artifacts\train.csv
    test_data_final_path: str=os.path.join("artifacts","test_final.csv") #all data will be saved in artifacts path
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl") #Create models and want to save in a pkl file

class DataTransformation3:
    def __init__(self):
        self.data_transformation_config3 = DataTransformationConfig3()

    def get_data_transformer_object(self):
        """
        Returns a preprocessor pipeline for deriving the target variable.
        """
        try:
            # Define the pipeline with the custom transformer
            target_variable_pipeline = Pipeline([
                ("complete_transformation_including_target_variable", CompleteTransformer())
            ])
            

            return target_variable_pipeline
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation2(self,train_data_path,test_data_path) -> pd.DataFrame:
        '''
        This function initiates the data transformation using the pipeline
        '''
        try:
            logging.info(f"Reading path for CSV files for train and test")
            #Step-1: Loading the Train and Test data from the path provided
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            #Step-1.5: Preprocessing which is just specific with regards 
            # to model building and will not be required in preprocessor.pkl
            # Preprocessing steps to derive the Target Variable [Remember, 
            # it is not present in dataset]
################################################
#T.0 - Transforming the date and time columns to form PlannedDateTime & ArrivedDateTime
            print("Shape Before Transformation - T.0 - Forming PlannedDateTime & ArrivedDateTime:",train_df.shape,test_df.shape)
            train_df["PlannedDateTime"] = pd.to_datetime(
                train_df["Planned Date"] + " " + train_df["Planned Time"],
                format="%d/%m/%Y %I:%M:%S %p",
            )
            train_df["ArrivedDateTime"] = pd.to_datetime(
                train_df["Arrival Date"] + " " + train_df["Arrival Time"],
                format="%d/%m/%Y %I:%M:%S %p",
            )

            print("Shape Before Transformation - 3.0 - Forming PlannedDateTime & ArrivedDateTime:",train_df.shape,test_df.shape)
            test_df["PlannedDateTime"] = pd.to_datetime(
                test_df["Planned Date"] + " " + test_df["Planned Time"],
                format="%d/%m/%Y %I:%M:%S %p",
            )
            test_df["ArrivedDateTime"] = pd.to_datetime(
                test_df["Arrival Date"] + " " + test_df["Arrival Time"],
                format="%d/%m/%Y %I:%M:%S %p",
            )
            print("Shape After Transformation - T.0 - Forming PlannedDateTime & ArrivedDateTime:",train_df.shape,test_df.shape)

################################################
#T.1 - Deriving the Target Variable
        
            # TimeDifference = ArrivedTime - PlannedTime in minutes
            print("Shape Before Transformation - T.1 - Derive Target Variable for Train:",train_df.shape)
            train_df['Delay'] = (train_df['ArrivedDateTime'] - train_df['PlannedDateTime']).dt.total_seconds() / 60  # in minutes
            train_df["TargetVariable"] = np.where(train_df["Delay"] >= 15, 1, 0)  # Target variable derived
            logging.info("Target variable derived successfully.")
            print("Shape After Transformation - T.1- Derive Target Variable for Train:",train_df.shape)
             
             # TimeDifference = ArrivedTime - PlannedTime in minutes
            print("Shape Before Transformation - T.1 - Derive Target Variable for Test:",test_df.shape)
            test_df['Delay'] = (test_df['ArrivedDateTime'] - test_df['PlannedDateTime']).dt.total_seconds() / 60  # in minutes
            test_df["TargetVariable"] = np.where(test_df["Delay"] >= 15, 1, 0)  # Target variable derived
            logging.info("Target variable derived successfully.")
            print("Shape After Transformation - T.1 - Derive Target Variable for Test:",test_df.shape)
 
################################################
#T.2- Dropping the columns = PlannedDateTime,ArrivedDateTime,Delay
            to_drop = ["PlannedDateTime","ArrivedDateTime","Delay"]
            train_df = train_df.drop(columns = to_drop,axis = 1)
            test_df = test_df.drop(columns = to_drop,axis = 1)


            #Step-2: Removing the Target Variable from the dataframe and sending it for preprocessing
            target_column_name = "TargetVariable"
            input_feature_train_df = train_df.drop(columns = [target_column_name],axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name],axis = 1)
            target_feature_test_df = test_df[target_column_name]    
            
            print("Information of DataFrame")
            print(train_df.info())

            #Step-2: Sending the DataFrame with Input Features for preprocessing
            preprocessing_obj = self.get_data_transformer_object()
            input_feature_train_df_final = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_df_final = preprocessing_obj.fit_transform(input_feature_test_df)

            #Step-3: Rejoining the train_df_final & test_df_final back with the Target Variable
            train_df_final = pd.concat([input_feature_train_df_final,target_feature_train_df],axis = 1)
            test_df_final = pd.concat([input_feature_test_df_final,target_feature_test_df],axis = 1)


            # Step 4: Save the transformed data to CSV files - this is just a precautionary measure
            logging.info("Saving transformed train and test datasets.")
            train_df_final.to_csv(
                self.data_transformation_config3.train_data_final_path, index=False, header=True
            )

            test_df_final.to_csv(
                self.data_transformation_config3.test_data_final_path, index=False, header=True
            )
        

            #Step 5: Saving the preprocessing object

              #we write this function save_object in utils
            save_object(
                #to save the pickle file
                file_path = self.data_transformation_config3.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info("Transformation of the datasets into final CSV files is completed.")
            return (
                # Step 4: Returning the path of the final csv file paths for next step
                train_df_final,
                test_df_final,
                self.data_transformation_config3.train_data_final_path,
                self.data_transformation_config3.test_data_final_path,
                self.data_transformation_config3.preprocessor_obj_file_path,

            )
        except Exception as e:
            raise CustomException(e, sys)