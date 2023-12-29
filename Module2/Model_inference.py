import json
import pandas as pd
import os

with open('./Module2/config.json', 'r') as file:
    config_json = json.load(file)

# spotify өгөгдлийг оруулж ирэх
def load_spotify_data():
    import pandas as pd

    df = pd.read_csv(os.path.join(config_json['DATA_DIR'], 'Spotify/spotify_songs.csv'))
    
    return df

# y хувьсагчийг категори болгох
def y_to_cat(y, thresh=70):
    y = y.map(lambda x: 1 if x>=thresh else 0)

    print("Хэрэв `track_popularity` нь {}-с дээш бол 1 үгүй бол 0".format(thresh))

    return y

# өгөгдлийг стандартжуулах
def scale_process_data(df, scale='MinMax'):
    import pickle
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

    df = load_spotify_data()

    if scale == 'MinMax':
        scaler = MinMaxScaler()
    
    elif scale == 'Standard':
        scaler = StandardScaler()
    
    X = df[config_json['DATA_COLUMNS']['X_COLUMNS']]
    y = df[config_json['DATA_COLUMNS']['Y_COLUMN']]

    num_columns = [col for col in X.columns if X[col].dtype in ['float','int']]
    cat_columns = [col for col in X.columns if X[col].dtype not in ['float','int']]

    # scaler-г оруулж ирэх
    with open(os.path.join(os.getenv('MODEL_DIR'), 'scaler.pickle'),'rb') as f:
            scaler = pickle.load(f)

    # тоон өгөгдлийг scale хийх
    X[num_columns] = scaler.transform(X[num_columns])    
    
    print("Тоон хувьсагчдыг scale хийсэн")

    # категори өгөгдлийг encode хийх
    labelencoder = LabelEncoder()
    for col in cat_columns:
        encoder_name = 'labelencoder_'+col+'.pickle'
        with open(os.path.join(config_json['MODEL_DIR'], encoder_name),'rb') as f:
            labelencoder = pickle.load(f)

        X[col] = labelencoder.transform(X[col])

    print("Категори хувьсагчдыг encode хийсэн")

    # таргет хувьсагчийг категори хувьсагч болгох
    y = y_to_cat(y)

    return X, y

# загварыг авах
def load_model():
    import pickle

    with open(os.path.join(config_json['MODEL_DIR'], 'model.pickle'),'rb') as f:
            model = pickle.load(f)

    print("Сургасан машин сургалтын загвар: ")
    print(model)

    return model

# загварыг сургах
def inference(model, data):

    from sklearn.metrics import recall_score, accuracy_score, precision_score
    
    y_pred = model.predict(data[0])
    y_pred_proba = model.predict_proba(data[0])[::,1]
    print("Inference хийсэн")
    print('Recall score: ', recall_score(data[1], y_pred))
    print('Accuracy score: ', accuracy_score(data[1], y_pred))
    print('Precision score: ', precision_score(data[1], y_pred))

    return y_pred, y_pred_proba

# загварыг хадгалах
def visualize_inference(y, y_pred, y_pred_proba):

    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    import matplotlib.pyplot as plt

    if not os.path.exists('./Charts'):
         os.mkdir('./Module2/Charts')
    else:
         pass

    fpr, tpr, thresholds = roc_curve(y, y_pred)

    roc_auc = auc(fpr, tpr)

    # method I: plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('./Module2/Charts/ROC_Curve.png')
    plt.show()
    
    precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
    plt.plot(recall, precision, 'r')
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title("Precision-Recall curve")
    plt.savefig('./Module2/Charts/PRC.png')
    plt.show()


def main():
    df = load_spotify_data()
    data = scale_process_data(df)
    model = load_model()
    y_pred, y_pred_proba = inference(model, data)
    visualize_inference(data[1], y_pred, y_pred_proba)
    
if __name__ == "__main__":
    main()