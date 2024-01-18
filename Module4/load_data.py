import os


class LoadSplitData:

    def __init__(self, config, test_size=0.3):
        self.config = config
        self.test_size = test_size

    def execute(self):
        self.load_data()
        self.split_data()

    def load_data(self):
        import pandas as pd

        DATA_PATH = self.config['DATA_DIR']
        self.df = pd.read_csv(os.path.join(DATA_PATH, 'Spotify/spotify_songs.csv'))

        print(">>> Өгөгдлийг импортлосон")

    def split_data(self):
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df[self.config['DATA_COLUMNS']['X_COLUMNS']], 
            self.df[self.config['DATA_COLUMNS']['Y_COLUMN']], 
            test_size=self.test_size, 
            random_state=12)

        print("-> Өгөгдлийг {}% тест, {}% сургалтын гэж хуваасан".format(self.test_size, 1-self.test_size))
        print("-- Сургалтын Х-н хэмжээ: ", self.X_train.shape, "; y-н хэмжээ: ", self.y_train.shape)
        print("-- Тестийн Х-н хэмжээ: ", self.X_test.shape, "; y-н хэмжээ: ", self.y_test.shape)

        self.X_train.to_csv(os.path.join(self.config['LOCAL_DATA_DIR'], "X_train_orig.csv"), index=False)
        self.X_test.to_csv(os.path.join(self.config['LOCAL_DATA_DIR'], 'X_test_orig.csv'), index=False)
        self.y_train.to_csv(os.path.join(self.config['LOCAL_DATA_DIR'], "y_train_orig.csv"), index=False)
        self.y_test.to_csv(os.path.join(self.config['LOCAL_DATA_DIR'], 'y_test_orig.csv'), index=False)

        