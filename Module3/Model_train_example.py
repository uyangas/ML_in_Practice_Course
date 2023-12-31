import json
import pandas as pd
import argparse
import pickle
import os

with open('./Module3/config.json', 'r') as file:
    config_json = json.load(file)

local_data_dir = config_json['LOCAL_DATA_DIR']

# Өгөгдөл хадгалах folder нээх
if not os.path.exists(local_data_dir):
    os.mkdir(local_data_dir)
else:
    pass

# spotify өгөгдлийг оруулж ирэх
def load_spotify_data():
    import pandas as pd

    DATA_PATH = config_json['DATA_DIR']
    df = pd.read_csv(os.path.join(DATA_PATH, 'Spotify/spotify_songs.csv'))
    
    return df

# өгөгдлийг сургалтын, тестийн гэж хуваах
def split_data(X, y, test_size=0.3):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=12)

    print("Өгөгдлийг {}% тест, {}% сургалтын гэж хуваасан".format(test_size, 1-test_size))
    print("Сургалтын Х-н хэмжээ: ", X_train.shape, "; y-н хэмжээ: ", y_train.shape)
    print("Сургалтын Х-н хэмжээ: ", X_test.shape, "; y-н хэмжээ: ", y_test.shape)

    return X_train, X_test, y_train, y_test

# y хувьсагчийг категори болгох
def y_to_cat(y, thresh=70):
    y = y.map(lambda x: 1 if x>=thresh else 0)

    print("Хэрэв `track_popularity` нь {}-с дээш бол 1 үгүй бол 0".format(thresh))

    return y

# өгөгдлийг стандартжуулах
def scale_process_data(df, scale='MinMax'):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

    df = load_spotify_data()

    if scale == 'MinMax':
        scaler = MinMaxScaler()
    
    elif scale == 'Standard':
        scaler = StandardScaler()

    X_train, X_test, y_train, y_test = split_data(df[config_json["DATA_COLUMNS"]["X_COLUMNS"]], df[config_json["DATA_COLUMNS"]["Y_COLUMN"]])

    num_columns = [col for col in X_train.columns if X_train[col].dtype in ['float','int']]
    cat_columns = [col for col in X_train.columns if X_train[col].dtype not in ['float','int']]

    # тоон өгөгдлийг scale хийх
    scaler.fit(X_train[num_columns])
    X_train[num_columns] = scaler.transform(X_train[num_columns])
    X_test[num_columns] = scaler.transform(X_test[num_columns])

    with open(os.path.join(config_json['MODEL_DIR'], 'scaler.pickle'),'wb') as f:
            pickle.dump(scaler, f)
    
    print("Тоон хувьсагчдыг scale хийсэн")

    # категори өгөгдлийг encode хийх
    labelencoder = LabelEncoder()
    for col in cat_columns:
        labelencoder.fit(X_train[col])
        X_train[col] = labelencoder.transform(X_train[col])
        X_test[col] = labelencoder.transform(X_test[col])
        encoder_name = 'labelencoder_'+col+'.pickle'

        with open(os.path.join(config_json['MODEL_DIR'], encoder_name),'wb') as f:
            pickle.dump(labelencoder, f)

    print("Категори хувьсагчдыг encode хийсэн")

    # таргет хувьсагчийг категори хувьсагч болгох
    y_train = y_to_cat(y_train)
    y_test = y_to_cat(y_test)

    X_train.to_csv(os.path.join(local_data_dir,"X_train.csv"), index=False)
    X_test.to_csv(os.path.join(local_data_dir,"X_test.csv"), index=False)
    y_train.to_csv(os.path.join(local_data_dir,"y_train.csv"), index=False)
    y_test.to_csv(os.path.join(local_data_dir,"y_test.csv"), index=False)

    return X_train, X_test, y_train, y_test

# загварыг авах
def build_model(max_depth, min_samples_leaf, min_samples_split):
    from sklearn.ensemble import RandomForestClassifier
    
    classifier = RandomForestClassifier(random_state=123, 
                                        max_depth=max_depth,
                                        min_samples_leaf=min_samples_leaf,
                                        min_samples_split=min_samples_split)
    
    print("Машин сургалтын загвар: ")
    print(classifier)

    return classifier

# загварыг сургах
def train_evaluate_model(model, data):
    from sklearn.metrics import recall_score, accuracy_score, precision_score

    model.fit(data[0], data[2])
    y_pred = model.predict(data[1])
    print('Recall score: ', recall_score(data[3], y_pred))
    print('Accuracy score: ', accuracy_score(data[3], y_pred))
    print('Precision score: ', precision_score(data[3], y_pred))

    return model

# загварыг хадгалах
def save_model(model):
    with open(os.path.join(config_json['MODEL_DIR'], 'model.pickle'), 'wb') as f:
        pickle.dump(model, f)
    
    print("Сургасан загварыг хадгалав")


def main():
    # Hyperparameter-уудыг environment-с авах
    max_depth = config_json["MODEL_CONFIG"]["MAX_DEPTH"]
    min_samples_leaf = config_json["MODEL_CONFIG"]["MIN_SAMPLES_LEAF"]
    min_samples_split = config_json["MODEL_CONFIG"]["MIN_SAMPLES_SPLIT"]

    df = load_spotify_data()
    data = scale_process_data(df, scale='Standard')
    model = build_model(max_depth, min_samples_leaf, min_samples_split)
    model_trained = train_evaluate_model(model, data)
    save_model(model_trained)
    
if __name__ == "__main__":
    main()