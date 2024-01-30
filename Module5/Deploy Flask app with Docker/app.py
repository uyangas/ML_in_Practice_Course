from flask import Flask, request, render_template, jsonify
import os
import pickle
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

with open("config.json", "rb") as f:
     config_json = json.load(f)

app = Flask('songpop')

# өгөгдлийг стандартжуулах
def scale_process_data(df):
    selected_cols = []
    X = df[config_json['DATA_COLUMNS']['X_COLUMNS']]
    y = df[config_json['DATA_COLUMNS']['Y_COLUMN']]

    num_columns = [col for col in X.columns if X[col].dtype in ['float','int']]
    cat_columns = [col for col in X.columns if X[col].dtype not in ['float','int']]
    selected_cols.extend(cat_columns)
    selected_cols.extend(num_columns)

    # scaler-г оруулж ирэх
    with open(os.path.join(config_json['MODEL_DIR'], 'scaler.pickle'),'rb') as f:
            scaler = pickle.load(f)

    # тоон өгөгдлийг scale хийх
    X_new = pd.DataFrame(scaler.transform(X[num_columns]), columns = num_columns)
    
    print("Тоон хувьсагчдыг scale хийсэн")

    # категори өгөгдлийг encode хийх
    labelencoder = LabelEncoder()
    for col in cat_columns:
        encoder_name = 'labelencoder_'+col+'.pickle'
        with open(os.path.join(config_json['MODEL_DIR'], encoder_name),'rb') as f:
            labelencoder = pickle.load(f)

        X_new[col] = labelencoder.transform(X[col])

    print("Категори хувьсагчдыг encode хийсэн")

    return X_new[selected_cols]


def model_inference(data):

    with open(os.path.join(config_json['MODEL_DIR'], 'model_RF.pickle'),'rb') as f:
            model = pickle.load(f)

    print("Сургасан машин сургалтын загварыг оруулав")
    
    y_pred = model.predict(data)
    y_pred_proba = model.predict_proba(data)[:,1]

    print("Өгөгдлийг таамаглав")

    return y_pred, y_pred_proba


def predict_data(data):
    # requested_song = {'singer':"Maroon 5",
    #                 'song_name':"Memories - Dillon Francis Remix"}
    requested_song = data

    main_df = pd.read_csv("test_data.csv")
    requested_data = main_df[(main_df['track_artist'] == requested_song['singer'])&(main_df['track_name'] == requested_song['song_name'])]

    processed_data = scale_process_data(requested_data)
    inference = model_inference(processed_data)

    return int(inference[0]), round(float(inference[1])*100)

@app.route("/")
def home():
    return render_template('index2.html')

@app.route('/predict', methods = ['POST'])
def result():
    
    singer = str(request.form['singer'])
    song_name = str(request.form['song_name'])
    print(singer, song_name)

    inference = predict_data({'singer':singer,
                              'song_name':song_name})   
      
    if int(inference[0]) == 1:
        prediction = 'The song is likely to be popular'
    else:
        prediction = 'The song is not likely to be popular'

    prediction_probability = str(inference[1])+"%"
    return render_template("result2.html", result = prediction, probability=prediction_probability)

if __name__ == "__main__":
     app.run(debug=True, host="0.0.0.0", port=5002)

