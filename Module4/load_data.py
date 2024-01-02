import pickle
import os

def load_data(config):
    import pandas as pd

    DATA_PATH = config['DATA_DIR']
    df = pd.read_csv(os.path.join(DATA_PATH, 'Spotify/spotify_songs.csv'))

    print(">>> Өгөгдлийг импортлосон")

    return df

def split_data(X, y, test_size=0.3):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=test_size, 
                                                        random_state=12)

    print("-> Өгөгдлийг {}% тест, {}% сургалтын гэж хуваасан".format(test_size, 1-test_size))
    print("-- Сургалтын Х-н хэмжээ: ", X_train.shape, "; y-н хэмжээ: ", y_train.shape)
    print("-- Тестийн Х-н хэмжээ: ", X_test.shape, "; y-н хэмжээ: ", y_test.shape)

    return X_train, X_test, y_train, y_test