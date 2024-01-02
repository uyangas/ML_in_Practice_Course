import pickle
import os


class TrainEvalModel:

    def __init__(self, model_type, params, X_train, y_train, config, task='Train', X_test=None, y_test=None, model_name=None):
        self.model_type = model_type
        self.params = params
        self.X_train = X_train
        self.y_train = y_train
        self.config = config
        self.task = task
        self.X_test = X_test
        self.y_test = y_test
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
        self.model.fit(self.X_train, self.y_train)

        print("-- Загварыг сургав")

    def save_model(self):
        model_name = 'model_'+self.model_type+'.pickle'
        with open(os.path.join(self.config['MODEL_DIR'], model_name), 'wb') as f:
            pickle.dump(self.model, f)
        
        print("-- Сургасан загварыг хадгалав")

    def evaluate_model(self):
        from sklearn.metrics import recall_score, accuracy_score, precision_score
        
        print(">>> Загварыг үнэлж байна")
        assert len(self.X_test)>0, "Тестийн өгөгдөл өгөөгүй байна"

        if not self.model_name:
            self.model_name=[i for i in os.listdir(self.config['MODEL_DIR']) if 'model' in i][0]
                        
            with open(os.path.join(self.config['MODEL_DIR'], self.model_name), 'rb') as f:
                self.model = pickle.load(f)

        y_pred = self.model.predict(self.X_test)
        import pandas as pd
        pd.DataFrame(y_pred, index=range(len(y_pred))).to_csv('./Module4/Data/y_pred.csv')

        print('Recall score: ', recall_score(self.y_test, pd.DataFrame(y_pred,columns=['Prediction'])))
        print('Accuracy score: ', accuracy_score(self.y_test, pd.DataFrame(y_pred,columns=['Prediction'])))
        print('Precision score: ', precision_score(self.y_test, pd.DataFrame(y_pred,columns=['Prediction'])))