from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import os
from datetime import datetime

#from src.model import load_model, predict_delay
#from src.load_preprocessor import load_preprocessor
#from src.data_preprocessing import preprocess_and_engineer, save_preprocessor

from src.pipelines.predict_pipeline import CustomData,PredictPipeline



app = Flask(__name__)

UPLOAD_FOLDER = os.path.abspath('uploads')
PROCESSED_FOLDER = os.path.abspath('processed')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created directory: {UPLOAD_FOLDER}")
else:
    print(f"Directory already exists: {UPLOAD_FOLDER}")

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)
    print(f"Created directory: {PROCESSED_FOLDER}")
else:
    print(f"Directory already exists: {PROCESSED_FOLDER}")



PREPROCESSOR_PATH = 'artifacts/preprocessor.pkl'

if not os.path.exists(PREPROCESSOR_PATH):
    print(f"{PREPROCESSOR_PATH} not found. Saving preprocessor...")
    # Save the preprocessor if it doesn't exist
    #save_preprocessor(preprocess_and_engineer)
#model = load_model()
#preprocessor = load_preprocessor()


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        
        if 'file' not in request.files:
            return "No file part in the request"
        file = request.files['file']
        
        
        if file.filename == '':
            return "No selected file"
        if file and file.filename.endswith('.csv'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            print(f"Saving uploaded file to: {file_path}")
            file.save(file_path)

            df = pd.read_csv(file_path)
            orignal_df = df.copy()

            try:
                
                #processed_data = preprocessor(df)
                #df['Delay_Percentage'] = predict_delay(model, processed_data)

                pred_df = df
                predict_pipeline=PredictPipeline()
                results,proba=predict_pipeline.predict(pred_df)
                #df['Status'] = (df['Delay_Percentage']>=50).map({True: 'Delayed', False: 'On Time'})

                pred_df["Delay_Percentage"] = proba[:,1]*100 #Converting Probability of class 1 to percentage
                pred_df["Status"] = (df["Delay_Percentage"]>=50).map({True: "Delayed", False: "On Time"})
                processed_file_name = f"results_{file.filename}"
                processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_file_name)

                print(f"Processed file will be saved to: {processed_file_path}")
                columns_to_append = df[['Delay_Percentage', 'Status']]
                final_df = pd.concat([orignal_df, columns_to_append], axis=1)
                final_df.to_csv(processed_file_path, index=False)

                message = f"The final CSV file has been saved in the processed folder as: {processed_file_name}"

                return render_template('results.html', file_name = processed_file_name ,message = message)
            except Exception as e:
                return f"Error during processing: {e}"
    return render_template('upload.html')

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    print(f"Attempting to download file from: {file_path}")  # Debugging
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
        