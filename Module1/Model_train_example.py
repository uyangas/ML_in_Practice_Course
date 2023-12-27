import json
import pandas as pd
import argparse
import pickle
import os

from sklearn.ensemble import RandomForestClassifier

X_columns = ['playlist_genre', 'playlist_subgenre','danceability', 'energy', 
                  'key', 'loudness', 'mode', 'speechiness','acousticness', 
                  'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
y_column = 'track_popularity'

# spotify өгөгдлийг оруулж ирэх
def load_spotify_data():
    import pandas as pd

    DATA_PATH = os.getenv('DATA_DIR')
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

    X_train, X_test, y_train, y_test = split_data(df[X_columns], df[y_column])

    num_columns = [col for col in X_train.columns if X_train[col].dtype in ['float','int']]
    cat_columns = [col for col in X_train.columns if X_train[col].dtype not in ['float','int']]

    # тоон өгөгдлийг scale хийх
    scaler.fit(X_train[num_columns])
    X_train[num_columns] = scaler.transform(X_train[num_columns])
    X_test[num_columns] = scaler.transform(X_test[num_columns])

    with open(os.path.join(os.getenv('MODEL_DIR'), 'scaler.pickle'),'wb') as f:
            pickle.dump(scaler, f)
    
    print("Тоон хувьсагчдыг scale хийсэн")

    # категори өгөгдлийг encode хийх
    labelencoder = LabelEncoder()
    for col in cat_columns:
        labelencoder.fit(X_train[col])
        X_train[col] = labelencoder.transform(X_train[col])
        X_test[col] = labelencoder.transform(X_test[col])
        encoder_name = 'labelencoder_'+col+'.pickle'

        with open(os.path.join(os.getenv('MODEL_DIR'), encoder_name),'wb') as f:
            pickle.dump(labelencoder, f)

    print("Категори хувьсагчдыг encode хийсэн")

    # таргет хувьсагчийг категори хувьсагч болгох
    y_train = y_to_cat(y_train)
    y_test = y_to_cat(y_test)

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
    with open(os.path.join(os.getenv('MODEL_DIR'), 'model.pickle'), 'wb') as f:
        pickle.dump(model, f)
    
    print("Сургасан загварыг хадгалав")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument("--max_depth", type=int, default=30, help="The depth of the tree")
    parser.add_argument("--min_samples_leaf", type=int, default=3, help="Min number of data points in leaf")
    parser.add_argument("--min_samples_split", type=int, default=3, help="Min number of data points required for split")
    args = parser.parse_args()

    # Hyperparameter-уудыг environment-с авах
    max_depth = int(os.environ.get("MAX_DEPTH", args.max_depth))
    min_samples_leaf = int(os.environ.get("MIN_SAMPLES_LEAF", args.min_samples_leaf))
    min_samples_split = int(os.environ.get("MIN_SAMPLES_SPLIT", args.min_samples_split))

    df = load_spotify_data()
    data = scale_process_data(df)
    model = build_model(max_depth, min_samples_leaf, min_samples_split)
    model_trained = train_evaluate_model(model, data)
    save_model(model_trained)
    
if __name__ == "__main__":
    main()