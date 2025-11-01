import numpy as np
import joblib
from tensorflow.keras.models import load_model # type: ignore
import requests
import json

# Importar TCN para carregar modelos personalizados
try:
    from tcn import TCN
except ImportError:
    print("Aviso: Biblioteca 'tcn' não encontrada. Modelos TCN não poderão ser carregados.")
    TCN = None

class Prediction:
    def __init__(self, model_path, scaler_path):
        # Carregar modelo com custom_objects se for TCN
        if 'tcn' in model_path.lower() and TCN is not None:
            self.model = load_model(model_path, custom_objects={'TCN': TCN})
        else:
            self.model = load_model(model_path)
        
        self.scaler = joblib.load(scaler_path)
        self.input_shape = self.model.input_shape

    def predict(self, features):
        try:
            X = np.array(features)
            
            # Ajuste de dimensionalidade
            if len(X.shape) == 1:
                X = np.expand_dims(X, axis=0)
            if len(X.shape) == 2:
                X = np.expand_dims(X, axis=2)
            
            # Verificação do formato
            if X.shape[1] != 8 or X.shape[2] != 1:
                return {'error': f"Formato incorreto. Esperado (None, 8, 1), recebido {X.shape}."}, 400

            # Normalização das features
            original_shape = X.shape
            X_reshaped = X.reshape(-1, original_shape[-1])
            X_normalized = self.scaler.transform(X_reshaped).reshape(original_shape)

            # Predição e desnormalização
            prediction = self.model.predict(X_normalized)
            desnormalized_prediction = self.scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]

            # Determinação do status
            threshold = 200.0
            status = "Ataque" if desnormalized_prediction < threshold else "Permitido"

            return {
                "prediction": round(float(desnormalized_prediction), 2),
                "status": status
            }

        except Exception as e:
            return {"error": str(e)}, 400

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
            
            return desnormalized_prediction < threshold
        except Exception as e:
            print(f"Erro na predição: {str(e)}")
            return False
