import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

#Here, you are setting up 
#sets up the logging system globally for the whole Python application. 
# #This means that once the configuration is set, any subsequent logging calls 
# #(even in different scripts) will use that configuration.
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,


)

#Testing code for logger only
#if __name__ =="__main__":
 #   logging.info("logging has started")

#Testing code for Exception