import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from PyQt5.QtCore import QThread, pyqtSignal
from modelSummary.RelatorioDosModelos import RelatorioDosModelos

class TrainingThreadARIMA(QThread):
    show_test_accuracy = pyqtSignal(float)
    update_progress = pyqtSignal(float)
    update_prediction_chart = pyqtSignal(np.ndarray, np.ndarray, np.ndarray,str)
    update_metrics_chart_boxplot = pyqtSignal(np.ndarray, np.ndarray,str)
    update_metrics_chart = pyqtSignal(float, float,str)

    def __init__(self, data_set):
        super().__init__()
        self.data_set = data_set
        self.train_data = None
        self.test_data = None

    def build_data(self, data):
        data['TXTDATE'] = pd.to_datetime(data['TXTDATE'])
        data = data.set_index('TXTDATE')
        data = data[~data.index.duplicated(keep='first')]  # Remover duplicatas no índice
        data = data.asfreq('D')  # Ajustar frequência diária

        # Dividir os dados em treinamento e teste
        split_index = int(len(data) * 0.8)
        self.train_data = data[:split_index]
        self.test_data = data[split_index:]

    def run(self):
        self.build_data(self.data_set)

        # Treinamento do modelo ARIMA
        model = ARIMA(self.train_data['LONGTIME'], order=(5, 1, 0))  # Ajustar os parâmetros (p, d, q) conforme necessário
        model_fit = model.fit()

        # Fazer previsões
        predictions = model_fit.forecast(steps=len(self.test_data))
        self.test_data['Predicted'] = predictions

        mse = np.mean((self.test_data['LONGTIME'] - self.test_data['Predicted']) ** 2)
        rmse = np.sqrt(mse)

        models_and_results = {
            'ARIMA_model': (self.test_data[['LONGTIME', 'Predicted']].reset_index(), None)  # Não há sumário de modelo específico
        }

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
        }

        relatorio = RelatorioDosModelos(model_fit, models_and_results, metrics)
        relatorio.save_metrics_pdf_KNN_ARIMA_RF("ARIMA_metrics.pdf")
        relatorio.save_reports_CSV()

        mse_list = []
        rmse_list = []
        mse_list.append(mse)
        rmse_list.append(rmse)
        mse_array = np.array(mse_list)  # Convertendo lista para array NumPy
        rmse_array = np.array(rmse_list)  # Convertendo lista para array NumPy
        
        
        self.update_prediction_chart.emit(self.test_data.index.values, self.test_data['LONGTIME'].values, self.test_data['Predicted'].values,"Modelo ARIMA")
        self.show_test_accuracy.emit(mse)
        self.update_metrics_chart.emit(mse, rmse,"Modelo ARIMA")
        self.update_metrics_chart_boxplot.emit(mse_array,rmse_array,"ARIMA")
    def add_moving_average_std(self, data, window_size=3):
        data['LONGTIME_MA'] = data['LONGTIME'].rolling(window=window_size).mean()
        data['LONGTIME_STD'] = data['LONGTIME'].rolling(window=window_size).std()
        return data

    def add_lag_features(self, data, lags=[1, 2, 3]):
        for lag in lags:
            data[f'LONGTIME_LAG_{lag}'] = data['LONGTIME'].shift(lag)
        return data

    def augment_data(self, X, y):
        augmented_X = np.copy(X)
        augmented_y = np.copy(y)

        for i in range(X.shape[0]):
            if np.random.rand() < 0.5:
                augmented_X = np.vstack([augmented_X, np.flipud(X[i:i+1])])
                augmented_y = np.hstack([augmented_y, y[i:i+1]])

        return augmented_X, augmented_y

    def save_model(self,model):
        self.output_directory = "ModelosComplilados"
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)
        h5_filename = os.path.join(self.output_directory,"tcn_model.h5")
        model.save(h5_filename)