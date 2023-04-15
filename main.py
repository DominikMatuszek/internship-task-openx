# A really simple REST API that takes one-dimensional array of features and returns its prediction.

import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel, conlist

from scikit_models import train_svc, train_knn_model
from train_neural_net import get_model_from_directory
from load_data import get_preprocessed_data
from heuristic import HeuristicClassifier


path = "covtype.data"
train_features, train_labels, _, _ = get_preprocessed_data(path)

del _ 

svc_model = train_svc(train_features, train_labels)
knn_model = train_knn_model(train_features, train_labels)
neural_net_model = get_model_from_directory("models/neural_net_model")
heuristic_model = HeuristicClassifier()

def get_prediction(model, features):
    features = np.array(features)

    # Add an artificial dimension to the features
    features = features.reshape(1, -1)

    prediction = model.predict(features) # The model returns a list of predictions 

    print(prediction)
    print(prediction.shape)
    print(prediction[0])

    # We need to convert the prediction to an integer, because otherwise JSON will have a problem with serializing it (it's a numpy.int64 object)
    return {"result": int(prediction)} 

class Features(BaseModel):
    data: conlist(int, min_items=54, max_items=54) 

app = FastAPI()

@app.get("/api/heuristic")
async def heuristic(features: Features):
    model = heuristic_model
    
    return get_prediction(model, features.data)

@app.get("/api/svc")
async def svc(features: Features):
    model = svc_model
    
    return get_prediction(model, features.data)

@app.get("/api/knn")
async def knn(features: Features):
    model = knn_model
    
    return get_prediction(model, features.data)

@app.get("/api/neural_net")
async def neural_net(features: Features):
    model = neural_net_model
    
    return get_prediction(model, features.data)
