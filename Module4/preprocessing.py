import pickle
import os


class Preprocessing:

    def __init__(self, config, X, y, train_data=True, thresh=70, scale='MinMax'):
        self.config = config
        self.X = X
        self.y = y
        self.train_data = train_data
        self.thresh = thresh
        self.scale = scale

    def execute(self):
        print(">>> Data Preprocessing started ...")

        self.scale_data()
        self.target_category()

        if self.train_data:
            X_name = "X_train.csv"
            y_name = "y_train.csv"
        else:
            X_name = "X_test.csv"
            y_name = "y_test.csv"

        self.X.to_csv(os.path.join(self.config['LOCAL_DATA_DIR'], X_name), index=False)
        self.y.to_csv(os.path.join(self.config['LOCAL_DATA_DIR'],y_name), index=False)

        print("-- Өгөгдлийн хэмжээ: ", self.X.shape, "; y-н хэмжээ: ", self.y.shape)
        print("-> Өгөгдөл хадгалагдав")      


    def scale_data(self):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

        if self.scale == 'MinMax':
            scaler = MinMaxScaler()
        
        elif self.scale == 'Standard':
            scaler = StandardScaler()

        num_columns = [col for col in self.X.columns if self.X[col].dtype in ['float','int']]
        cat_columns = [col for col in self.X.columns if self.X[col].dtype not in ['float','int']]

        print(f"-- {len(num_columns)} тоон хувьсагч тодорхойлогдлоо")
        print(f"-- {len(cat_columns)} тоон хувьсагч тодорхойлогдлоо")

        if self.train_data:
            scaler.fit(self.X[num_columns])
            with open(os.path.join(self.config['MODEL_DIR'], 'scaler.pickle'),'wb') as f:
                pickle.dump(scaler, f)
            
        else:
            with open(os.path.join(self.config['MODEL_DIR'], 'scaler.pickle'),'rb') as f:
                scaler=pickle.load(f)

        self.X[num_columns] = scaler.transform(self.X[num_columns])
        
        print(f"-> Тоон хувьсагчдыг scale хийсэн: `{scaler}`")


        for col in cat_columns:
            encoder_name = 'labelencoder_'+col+'.pickle'
            if self.train_data:
                labelencoder = LabelEncoder()
                labelencoder.fit(self.X[col])
                with open(os.path.join(self.config['MODEL_DIR'], encoder_name),'wb') as f:
                    pickle.dump(labelencoder, f)
            else:
                with open(os.path.join(self.config['MODEL_DIR'], encoder_name),'rb') as f:
                    labelencoder=pickle.load(f)

            self.X[col] = labelencoder.transform(self.X[col])

        print("-> Категори хувьсагчдыг encode хийсэн: `Label Encoder`")
    
    def target_category(self):
        # таргет хувьсагчийг категори хувьсагч болгох
        self.y = self.y.map(lambda x: 1 if x>=self.thresh else 0)
        
        print(f"-> Таргет хувьсагчийг `Thresh={self.thresh}` байхаар категори хувьсагч болгосон")