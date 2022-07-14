#!/usr/bin/env python

import os, warnings, sys
import pprint
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

import numpy as np, pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.cluster import KMeans

import algorithm.preprocessing.pipeline as pp_pipe
import algorithm.preprocessing.preprocess_utils as pp_utils
import algorithm.utils as utils
# import algorithm.scoring as scoring
from algorithm.model.clustering import ClusteringModel as Model, get_data_based_model_params
from algorithm.utils import get_model_config


# get model configuration parameters 
model_cfg = get_model_config()


def get_trained_model(train_data, data_schema, hyper_params, num_clusters):  
    
    # set random seeds
    utils.set_seeds()    
    
    # preprocess data
    print("Pre-processing data...")
    train_X, _, preprocess_pipe = preprocess_data(train_data, None, data_schema)  
    # print('train_X shape:',  train_X.shape)  ; sys.exit()
                  
    # Create model   
       
    # get model hyper-paameters parameters 
    data_based_params = get_data_based_model_params(train_X)
    model_params = { **hyper_params, **data_based_params }
    print(model_params) #; sys.exit()
    
    model = Model(  **model_params )  
    # train and get clusters
    transformed_X = model.fit_predict(train_X)
    
    
    # return the prediction df with the id and prediction fields
    id_field_name = data_schema["inputDatasets"]["clusteringBaseMainInput"]["idField"] 
    preds_df = train_data[[id_field_name]].copy()
    preds_df['prediction'] = transformed_X
    print(preds_df['prediction'].value_counts()) #; sys.exit()
    
    
    return preprocess_pipe, model, preds_df



def preprocess_data(train_data, valid_data, data_schema):
    # print('Preprocessing train_data of shape...', train_data.shape)
    pp_params = pp_utils.get_preprocess_params(train_data, data_schema, model_cfg) 
    # pprint.pprint(pp_params) 
    
    preprocess_pipe = pp_pipe.get_preprocess_pipeline(pp_params, model_cfg)
    train_data = preprocess_pipe.fit_transform(train_data)
    # print("Processed train X/y data shape", train_data['X'].shape, train_data['y'].shape)
      
    if valid_data is not None:
        valid_data = preprocess_pipe.transform(valid_data)
    # print("Processed valid X/y data shape", valid_data['X'].shape, valid_data['y'].shape)
    return train_data, valid_data, preprocess_pipe 


