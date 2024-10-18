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
            # Escalonamento dos dados
            X = np.array(features)
            X = np.expand_dims(X, axis=0)
            normalized_features = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

            # Enviar requisição para a API local
            url = 'http://localhost:5000/predict'
            data = {'features': normalized_features.tolist()}
            response = requests.post(url, json=data)

            if response.status_code != 200:
                return {'error': f"Erro ao acessar a API: {response.status_code}"}, response.status_code

            response_json = response.json()
            if 'prediction' not in response_json:
                return {'error': 'A resposta da API não contém uma predição válida.'}, 500

            normalized_prediction = response_json['prediction'][0][0]
            desnormalized_prediction = self.scaler.inverse_transform([[normalized_prediction]])[0][0]

            threshold = 200.0

            # Salvar as informações em um arquivo de log
            with open('./logs/predictions_log.txt', 'a') as log_file:
                log_file.write(f"{normalized_prediction},{desnormalized_prediction},{'Ataque' if desnormalized_prediction < threshold else 'Permitido'}\n")

            return desnormalized_prediction < threshold
        except requests.RequestException as e:
            return {'error': f"Erro de comunicação com a API: {str(e)}"}, 500
        except Exception as e:
            return {'error': f"Erro no processamento: {str(e)}"}, 500
