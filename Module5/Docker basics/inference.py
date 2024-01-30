import os
import joblib
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

with open("config.json", "rb") as f:
     config_json = json.load(f)

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
            scaler = joblib.load(f)

    # тоон өгөгдлийг scale хийх
    X_new = pd.DataFrame(scaler.transform(X[num_columns]), columns = num_columns)

    # категори өгөгдлийг encode хийх
    for col in cat_columns:
        encoder_name = 'labelencoder_'+col+'.pickle'
        with open(os.path.join(config_json['MODEL_DIR'], encoder_name),'rb') as f:
            labelencoder = joblib.load(f)

        X_new[col] = labelencoder.transform(X[col])

    return X_new[selected_cols]


def model_inference(data):

    with open(os.path.join(config_json['MODEL_DIR'], 'model_RF.pickle'),'rb') as f:
            model = joblib.load(f)
    
    y_pred = model.predict(data)
    y_pred_proba = model.predict_proba(data)[:,1]

    return y_pred, y_pred_proba


def predict_data(data):

    processed_data = scale_process_data(data)
    inference = model_inference(processed_data)

    return int(inference[0]), round(float(inference[1])*100)

def result():
    
    main_df = pd.read_csv("test_data.csv")

    for row in main_df.iterrows():
        rowline = row[1]
        singer = rowline['track_artist']
        song = rowline['track_name']
        data = main_df[(main_df['track_artist'] == singer)&(main_df['track_name'] == song)]

        inference = predict_data(data)   
        
        if int(inference[0]) == 1:
            prediction = 'The song is likely to be popular'
        else:
            prediction = 'The song is not likely to be popular'

        prediction_probability = str(inference[1])+"%"
        
        print(f">>> {singer} - {song}")
        print(f"--- {prediction}; Магадлал: {prediction_probability}")

if __name__ == "__main__":
     result()

