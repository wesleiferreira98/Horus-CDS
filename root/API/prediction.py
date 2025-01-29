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

        # Verifica se os dados têm o formato esperado
        if X.shape[1] != 8 or X.shape[2] != 1:
            return {'error': f"Formato dos dados incorreto. Esperado (None, 8, 1), recebido {X.shape}."}, 400

        prediction = self.model.predict(X)
        return {'prediction': prediction.tolist()}

    def is_attack(self, features):
        try:
            # Processamento local direto sem chamada HTTP
            X = np.array(features)
            X = np.expand_dims(X, axis=0)
            
            # Verificação de dimensionalidade
            if len(X.shape) == 2:
                X = np.expand_dims(X, axis=2)
            
            # Pré-processamento
            normalized_features = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            # Previsão direta
            prediction = self.model.predict(normalized_features)
            desnormalized_prediction = self.scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
            
            threshold = 200.0
            # ... (código de logging permanece igual)
            
            return desnormalized_prediction < threshold
        except Exception as e:
            print(f"Erro na predição: {str(e)}")
            return False
