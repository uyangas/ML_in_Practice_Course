import json
import os

from load_data import load_data, split_data
from preprocessing import Preprocessing
from train_eval_model import TrainEvalModel

with open('./Module4/config.json', 'r') as file:
    config_json = json.load(file)

local_data_dir = config_json['LOCAL_DATA_DIR']

# Өгөгдөл хадгалах folder нээх
if not os.path.exists(local_data_dir):
    os.mkdir(local_data_dir)
else:
    pass


def main():

    df = load_data(config_json)
    X_train, X_test, y_train, y_test = split_data(df[config_json['DATA_COLUMNS']['X_COLUMNS']], 
                                                  df[config_json['DATA_COLUMNS']['Y_COLUMN']])

    # process train data
    Preprocessing(config=config_json,
                  X=X_train,
                  y=y_train,
                  train_data=True                 
                  ).execute()
    
    # process test data
    Preprocessing(config=config_json,
                  X=X_test,
                  y=y_test,
                  train_data=False                 
                  ).execute()

    # Hyperparameter-уудыг environment-с авах
    params={'random_state':123,
            'max_depth':config_json["MODEL_CONFIG"]["MAX_DEPTH"],
            'min_samples_leaf':config_json["MODEL_CONFIG"]["MIN_SAMPLES_LEAF"],
            'min_samples_split':config_json["MODEL_CONFIG"]["MIN_SAMPLES_SPLIT"]}

    TrainEvalModel(model_type='RF', 
                   params=params, 
                   X_train=X_train, 
                   y_train=y_train, 
                   config=config_json, 
                   task='Train_eval', 
                   X_test=X_test, 
                   y_test=y_test
                   ).execute()
    
if __name__ == "__main__":
    main()