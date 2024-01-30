import json
import os

from load_data import LoadSplitData
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

    if len(os.listdir(local_data_dir))==0:

        LoadSplitData(config=config_json
                      ).execute()

        # process train data
        Preprocessing(config=config_json,
                      train_data=True                 
                      ).execute()
        
        # process test data
        Preprocessing(config=config_json,
                      train_data=False                 
                      ).execute()
    else:
        print(">>> Өгөгдөл боловсруулагдсан байна")

    # Hyperparameter-уудыг environment-с авах
    params={'random_state':123,
            'max_depth':config_json["MODEL_CONFIG"]["MAX_DEPTH"],
            'min_samples_leaf':config_json["MODEL_CONFIG"]["MIN_SAMPLES_LEAF"],
            'min_samples_split':config_json["MODEL_CONFIG"]["MIN_SAMPLES_SPLIT"]}

    TrainEvalModel(model_type='RF', 
                   params=params, 
                   config=config_json, 
                   task='Train_eval',
                   model_name='model_rf'
                   ).execute()
    
if __name__ == "__main__":
    main()