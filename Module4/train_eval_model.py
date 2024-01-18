import pickle
import os
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, precision_score


class TrainEvalModel:

    def __init__(self, model_type, params, config, task='Train', model_name=None):
        self.model_type = model_type
        self.params = params
        self.config = config
        self.task = task
        self.model_name = model_name

    def execute(self):

        if self.task=='Train':
            self.build_model()
            self.train_model()
            self.save_model()

        elif self.task=='Train_eval':
            self.build_model()
            self.train_model()
            self.save_model()
            self.evaluate_model()

        elif self.task=='Eval':
            self.evaluate_model()

    def build_model(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        
        if self.model_type=='RF':
            self.model = RandomForestClassifier(**self.params)
        elif self.model_type=='LR':
            self.model = LogisticRegression(**self.params)
        elif self.model_type=='DT':
            self.model = DecisionTreeClassifier(**self.params)
        
        print(">>> Машин сургалтын загвар: ")
        print("--", self.model)

    def train_model(self):
        print(">>> Загвар сургалт эхлэв")
        
        self.X_train = pd.read_csv(os.path.join(self.config['LOCAL_DATA_DIR'], "X_train.csv"))
        self.y_train = pd.read_csv(os.path.join(self.config['LOCAL_DATA_DIR'], "y_train.csv"))['track_popularity']

        self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_train)

        print('Сургалтын Recall score: ', recall_score(self.y_train, y_pred))
        print('Сургалтын Accuracy score: ', accuracy_score(self.y_train, y_pred))
        print('Сургалтын Precision score: ', precision_score(self.y_train, y_pred))

        print("-- Загварыг сургав")

    def save_model(self):
        model_save_name = 'model_'+self.model_type+'.pickle'
        with open(os.path.join(self.config['MODEL_DIR'], model_save_name), 'wb') as f:
            pickle.dump(self.model, f)
        
        print("-- Сургасан загварыг хадгалав")

    def evaluate_model(self):
        
        self.X_test = pd.read_csv(os.path.join(self.config['LOCAL_DATA_DIR'], 'X_test.csv'))
        self.y_test = pd.read_csv(os.path.join(self.config['LOCAL_DATA_DIR'], 'y_test.csv'))['track_popularity']

        print(">>> Загварыг үнэлж байна")
        assert len(self.X_test)>0, "Тестийн өгөгдөл өгөөгүй байна"

        if self.task == 'Eval':
            self.model_name = self.model_name + '.pickle'
            with open(os.path.join(self.config['MODEL_DIR'], self.model_name), 'rb') as f:
                self.model = pickle.load(f)
        else:
            pass

        y_pred = self.model.predict(self.X_test)
        
        pd.DataFrame(y_pred, index=range(len(y_pred))).to_csv('./Module4/Data/y_pred.csv', index=False)

        print('Recall score: ', recall_score(self.y_test, y_pred))
        print('Accuracy score: ', accuracy_score(self.y_test, y_pred))
        print('Precision score: ', precision_score(self.y_test, y_pred))