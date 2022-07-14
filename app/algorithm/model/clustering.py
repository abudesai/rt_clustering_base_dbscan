
from dataclasses import replace
import dis
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances

warnings.filterwarnings('ignore') 


model_fname = "model.save"

MODEL_NAME = "DBSCAN"


class ClusteringModel(): 
    
    def __init__(self, eps, min_samples) -> None:
        self.eps = eps     
        self.min_samples = min_samples          
        self.model = self.build_model()
        
        
    def build_model(self): 
        model = DBSCAN(
            eps=self.eps, 
            min_samples=self.min_samples,
        )
        return model
    

    def __getattr__(self, name):
        # https://github.com/faif/python-patterns
        # model.predict() instead of model.model.predict()
        # same for fit(), transform(), fit_transform(), etc.
        attr = getattr(self.model, name)

        if not callable(attr): return attr

        def wrapper(*args, **kwargs):
            return getattr(self.model, attr.__name__)(*args, **kwargs)

        return wrapper    
    
    
    def evaluate(self, x_test): 
        """Evaluate the model and return the loss and metrics"""
        raise NotImplementedError

    
    def save(self, model_path): 
        joblib.dump(self.model, os.path.join(model_path, model_fname))


    @classmethod
    def load(cls, model_path):         
        clusterer = joblib.load(os.path.join(model_path, model_fname))
        return clusterer


def save_model(model, model_path):    
    model.save(model_path) 
    

def load_model(model_path): 
    try: 
        model = ClusteringModel.load(model_path)        
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model


def get_data_based_model_params(data): 
    ''' 
        Set any model parameters that are data dependent. 
        For example, number of layers or neurons in a neural network as a function of data shape.
    '''  
    # https://stats.stackexchange.com/questions/88872/a-routine-to-choose-eps-and-minpts-for-dbscan
    min_samples = 3 * data.shape[1]
    
    eps = get_percentile_distance(data)
    return {"min_samples": min_samples, "eps": eps}


def get_percentile_distance(data): 
    N = data.shape[0]
    num_samples = int(min(N, 100, max(500, N * 0.1)))
    # print("num_samples", num_samples)
    
    samples = data.sample(n=num_samples, replace=False, axis = 0)
    # print(samples.shape); sys.exit()
    
    distances = euclidean_distances(samples, samples).flatten()
    # print(distances.shape, distances.mean())
    
    perc_value = np.percentile(distances, 5.0)
    return perc_value