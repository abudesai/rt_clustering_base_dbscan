import numpy as np
import os, sys

import algorithm.utils as utils
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.clustering as clustering


# get model configuration parameters 
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path): 
        self.model_path = model_path
        self.preprocessor = None
        self.model = None
    
    
    def _get_preprocessor(self): 
        if self.preprocessor is None: 
            try: 
                self.preprocessor = pipeline.load_preprocessor(self.model_path)
                return self.preprocessor
            except: 
                print(f'Could not load preprocessor from {self.model_path}. Did you train the model first?')
                return None
        else: return self.preprocessor
    
    def _get_model(self): 
        if self.model is None: 
            try: 
                self.model = clustering.load_model(self.model_path)
                return self.model
            except: 
                print(f'Could not load model from {self.model_path}. Did you train the model first?')
                return None
        else: return self.model
        
    
    def _get_predictions(self, data, data_schema):  
        preprocessor = self._get_preprocessor()
        model = self._get_model()
        
        if preprocessor is None:  raise Exception("No preprocessor found. Did you train first?")
        if model is None:  raise Exception("No model found. Did you train first?")
                    
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        pred_X = preprocessor.transform(data)  
        preds = model.predict( pred_X )
        return preds    
    
        
    def predict(self, data, data_schema):
        preds = self._get_predictions(data, data_schema) 
        
        # return the prediction df with the id and prediction fields
        id_field_name = data_schema["inputDatasets"]["clusteringBaseMainInput"]["idField"]  
        preds_df = data[[id_field_name]].copy()
        preds_df['prediction'] = preds
        
        return preds_df
