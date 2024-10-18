import numpy as np
import joblib
from tensorflow.keras.models import load_model # type: ignore
import requests
import json

class Prediction:
    def __init__(self, model_path, scaler_path):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.input_shape = self.model.input_shape

    def predict(self, features):
        X = np.array(features)
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=2)

        if X.shape[1] != 8 or X.shape[2] != 1:
            return {'error': 'Formato dos dados incorreto.'}, 400

        prediction = self.model.predict(X)
        return {'prediction': prediction.tolist()}

    def is_attack(self, features):
        X = np.array(features)
        X = np.expand_dims(X, axis=0)
        normalized_features = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        url = 'http://localhost:5000/predict'
        data = {'features': normalized_features.tolist()}
        response = requests.post(url, json=data)

        if 'prediction' not in response.json():
            return False

        normalized_prediction = response.json()['prediction'][0][0]
        desnormalized_prediction = self.scaler.inverse_transform([[normalized_prediction]])[0][0]

        threshold = 200.0
        return desnormalized_prediction < threshold
