"""
Function untuk membersihkan data teks
"""
import re
import pandas as pd
import pickle, re
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

max_features = 96
file = open('tokenizer/tokenizer.pickle','rb') # tokenizer
tokenizer = pickle.load(file)
file.close()
sentiment = ['negative', 'neutral', 'positive']

def cleansing(text):
    # Bersihkan tanda baca (selain huruf dan angka)
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # ubah teks menjadi lower case
    clean_text = clean_text.lower()
    # menghilangkan emoji
    clean_text = re.sub(r'xf0\S+', '', clean_text)
    clean_text = re.sub(r'xe\S+', '', clean_text)
    #remove repeated character
    clean_text = re.sub(r'(.)\1+', r'\1', clean_text)
    # menggantikan kata alay dengan kata formal
    replacement_words = pd.read_csv('csv_data/new_kamusalay.csv')
    replacement_dict = dict(zip(replacement_words['alay_word'], replacement_words['formal_word']))
    words = clean_text.split()
    replaced_words = [replacement_dict.get(word, word) for word in words]
    clean_text = ' '.join(replaced_words)
    # menghilangkan spasi di awal dan akhir teks
    clean_text = clean_text.strip()
    return clean_text

model_file_path = "model_of_lstm/model.h5"
model_file_from_lstm = load_model(model_file_path)
print('Model Loaded successfully !')
model_file_from_lstm.summary()

def lstm(text):
    text = [cleansing(text)]
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=max_features)

    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]
    return get_sentiment

def analisis_file(file_upload):
    # read csv file upload, jika eror dengan metode biasa
    df_upload = pd.DataFrame(file_upload.iloc[:,[0]])
    # rename kolom menjadi "raw_text"
    df_upload.columns = ["raw_text"]
    # bersihkan teks menggunakan fungsi cleansing
    # simpan di kolom "clean_text"
    df_upload["clean_text"] = df_upload["raw_text"].apply(cleansing)
    df_upload["sentiment"] = df_upload["clean_text"].apply(lstm)
    # mensensor kata abusive sesuai dari data dalam bentuk csv
    
    print("Cleansing text succes!")
    return df_upload