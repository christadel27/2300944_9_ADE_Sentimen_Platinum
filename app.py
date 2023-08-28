"""
Flask API Application

"""
from flask import Flask, jsonify, request
from flasgger import Swagger, swag_from, LazyJSONEncoder, LazyString
import pandas as pd
import numpy as np
from time import perf_counter
import re
import flask
from cleansing_analisis import cleansing, analisis_file, lstm
from db import (
    create_connection, 
    insert_result_to_db, show_analisis_result,
    insert_upload_result_to_db
)

# initializze flask application
app = Flask(__name__)

#Assign LazyJSONEncoder to app.json_encoder for swagger UI
app.json_encoder = LazyJSONEncoder
# create swagger config & swagger template
Swagger_template ={
    "info":{
        "title": LazyString(lambda: "Membersihkan Teks dan Menganalisis Sentimen API"),
        "version": LazyString(lambda: "1.0.0"),
        "description": LazyString(lambda: "Dokumentasi API untuk Membersihkan Teks dan Menganalisisnya")
    },
    "host": LazyString(lambda: request.host)
}
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
# intiliazeSwagger from tempalte & config
swagger = Swagger(app, template=Swagger_template, config=swagger_config)

# homepage
@swag_from('docs/home.yml', methods=['GET'])
@app.route('/', methods=['GET'])
def home():
    welcome_msg = {
        "version": "1.0.0",
        "message": "Welcome to Flask API",
        "author": "Adelia Christyanti dan Sony Dertha Setiawan"
    }
    return jsonify(welcome_msg)

# Show analisis result
@swag_from('docs/show_analisis_result.yml', methods=['GET'])
@app.route('/show_analisis_result', methods=['GET'])
def show_analisis_result_api():
    db_connection = create_connection()
    analisis_result = show_analisis_result(db_connection)
    return jsonify(analisis_result)

#cleansing text using form
@swag_from('docs/lstm.yml', methods=['POST'])
@app.route('/lstm', methods=['POST'])
def lstm_endpoint():
    # get text from input user
    raw_text = request.form["raw_text"]
    # cleansing text
    start = perf_counter()
    cleansing_text = cleansing(raw_text)
    analisis_text = lstm(cleansing_text)
    end = perf_counter()
    time = end - start
    print(f'processing time :{time}')
    result_response ={"raw_text": raw_text, "clean_text": cleansing_text,"Sentiment": analisis_text, "processing time": time}
    # insert result to database
    db_connection = create_connection()
    insert_result_to_db(db_connection, raw_text, cleansing_text, analisis_text)
    return jsonify(result_response)

# Cleansing text using csv upload
@swag_from('docs/lstm_upload.yml', methods=['POST'])
@app.route('/lstm_upload', methods=['POST'])
def cleansing_upload():
    # Get file from upload to dataframe
    uploaded_file = request.files['upload_file']
    # Read csv file to dataframe
    df_upload = pd.read_csv(uploaded_file,encoding ='latin-1').head(1000)
    print('Read dataframe Upload success!')
    start = perf_counter()
    df_cleansing = analisis_file(df_upload)
    end = perf_counter()
    time = end - start
    print(f'processing time :{time}')
    
    # Upload result to database
    db_connection = create_connection()
    insert_upload_result_to_db(db_connection, df_cleansing)    
    print("Upload result to database success!")
    result_response = df_cleansing.to_dict(orient='records')
    return jsonify(result_response)

if __name__ == '__main__':
    app.run()