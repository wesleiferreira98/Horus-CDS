import sys
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models, optimizers # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


class TrainingThread(QThread):
    show_test_accuracy = pyqtSignal(float)  # Novo sinal para indicar o término do treinamento
    update_progress = pyqtSignal(float)  # Sinal para atualizar a barra de progresso
    update_prediction_chart = pyqtSignal(np.ndarray, np.ndarray)  # Sinal para atualizar o gráfico de previsão
    update_metrics_chart = pyqtSignal(float, float)  # Sinal para atualizar o gráfico de métricas

    def __init__(self, train_data, train_labels, test_data, test_labels):
        super().__init__()
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.test_mse = None
        
    def build_model(self, num_hidden_layers, num_neurons):
        model = models.Sequential()
        model.add(layers.Dense(num_neurons, activation='relu', input_shape=(self.train_data.shape[1],)))

        for _ in range(num_hidden_layers - 1):
            model.add(layers.Dense(num_neurons, activation='relu'))

        model.add(layers.Dense(1))  # Camada de saída

        return model
    def preprocess_data(self, data):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        return data_scaled
    
    def apply_smote(self, X, y):
        """
        Aplica SMOTE para balanceamento de dados em problemas de regressão.
        Para regressão, primeiro discretiza o target em bins e depois aplica SMOTE.
        """
        try:
            # Para regressão, precisamos discretizar o target primeiro
            # Criamos bins baseados nos quartis
            quartiles = np.quantile(y, [0.25, 0.5, 0.75])
            y_binned = np.digitize(y, quartiles)
            
            # Aplicamos SMOTE nos dados com target discretizado
            smote = SMOTE(random_state=42, k_neighbors=min(5, len(np.unique(y_binned))-1))
            X_resampled, y_binned_resampled = smote.fit_resample(X, y_binned)
            
            # Para reconstruir o y contínuo, usamos interpolação baseada nos bins
            # Mapeamos os bins de volta para valores contínuos usando a média dos valores originais em cada bin
            bin_means = {}
            for bin_val in np.unique(y_binned):
                mask = y_binned == bin_val
                if np.any(mask):
                    bin_means[bin_val] = np.mean(y[mask])
                else:
                    bin_means[bin_val] = np.mean(y)  # fallback
            
            # Reconstruímos o y contínuo
            y_resampled = np.array([bin_means[bin_val] for bin_val in y_binned_resampled])
            
            # Adicionamos ruído gaussiano para restaurar a variabilidade contínua
            noise_std = np.std(y) * 0.1  # 10% do desvio padrão original
            y_resampled += np.random.normal(0, noise_std, len(y_resampled))
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"Erro ao aplicar SMOTE: {e}")
            print("Mantendo dados originais...")
            return X, y

    def run(self):
        # Pré-processamento dos dados
        self.train_data_scaled = self.preprocess_data(self.train_data)
        self.test_data_scaled = self.preprocess_data(self.test_data)
        
        # Aplicar SMOTE para balanceamento dos dados
        self.train_data_scaled, self.train_labels = self.apply_smote(self.train_data_scaled, self.train_labels)

        # Define uma grade de hiperparâmetros para ajustar
        param_grid = {
            'num_hidden_layers': [1, 2, 3],
            'num_neurons': [32, 64, 128],
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [32, 64, 128]
        }

        best_mse = float('inf')
        best_model = None
        total_combinations = len(param_grid['num_hidden_layers']) * len(param_grid['num_neurons']) * len(param_grid['learning_rate']) * len(param_grid['batch_size'])
        current_progress = 0
        
        for num_hidden_layers in param_grid['num_hidden_layers']:
            for num_neurons in param_grid['num_neurons']:
                for learning_rate in param_grid['learning_rate']:
                    for batch_size in param_grid['batch_size']:
                        # Constrói o modelo com os hiperparâmetros atuais
                        current_progress += 1
                        self.update_progress.emit(current_progress / total_combinations)

                        model = self.build_model(num_hidden_layers, num_neurons)
                        
                        # Compila o modelo
                        optimizer = optimizers.Adam(learning_rate=learning_rate)
                        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

                        # Treina o modelo
                        history = model.fit(self.train_data_scaled, self.train_labels, epochs=5, batch_size=batch_size, validation_split=0.2, verbose=0)

                        # Avalia o modelo
                        test_mse = model.evaluate(self.test_data_scaled, self.test_labels, verbose=0)[1]

                        # Verifica se é o melhor modelo até agora
                        if test_mse < best_mse:
                            best_mse = test_mse
                            best_model = model
                        

                       

        # Emite sinais para indicar o término do treinamento e o melhor MSE
        test_results = best_model.predict(self.test_data_scaled)
        self.update_prediction_chart.emit(self.test_labels, test_results)
        self.show_test_accuracy.emit(best_mse)
        self.update_metrics_chart.emit(best_mse, np.sqrt(best_mse))