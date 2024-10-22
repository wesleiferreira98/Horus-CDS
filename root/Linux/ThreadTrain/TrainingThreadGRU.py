import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import GRU, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from View.LogTrain import LogTrain
from modelSummary.ModelSummary import ModelSummary
from modelSummary.RelatorioDosModelos import RelatorioDosModelos

class KerasGRURegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_shape, units=50, activation='tanh', dropout=0.0):
        self.input_shape = input_shape
        self.units = units
        self.activation = activation
        self.dropout = dropout
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            GRU(self.units, activation=self.activation, input_shape=self.input_shape, dropout=self.dropout),
            Dense(1)
        ])
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
        return model

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.evaluate(X, y)[1]

class TrainingThreadGRU(QThread):
    show_test_accuracy = pyqtSignal(float)
    update_progress = pyqtSignal(float)
    update_prediction_chart = pyqtSignal(np.ndarray, np.ndarray, np.ndarray,str)
    update_metrics_chart_boxplot = pyqtSignal(np.ndarray, np.ndarray,str)
    update_metrics_chart = pyqtSignal(float, float,str)

    def __init__(self, data_set):
        super().__init__()
        self.data_set = data_set
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.logModel = LogTrain("GRU","08:30")

    def build_data(self, data):
        data['TXTDATE'] = pd.to_datetime(data['TXTDATE'])
        data['Dia_da_Semana'] = data['TXTDATE'].dt.dayofweek
        data['Mês'] = data['TXTDATE'].dt.month
        data['Hora'] = pd.to_datetime(data['TXTTIME'], format='%H:%M:%S').dt.hour

        # Add moving average and std
        data = self.add_moving_average_std(data, window_size=3)
        # Add lag features
        data = self.add_lag_features(data, lags=[1, 2, 3])

        features = ['Dia_da_Semana', 'Mês', 'Hora', 'LONGTIME_MA', 'LONGTIME_STD', 'LONGTIME_LAG_1', 'LONGTIME_LAG_2', 'LONGTIME_LAG_3', 'TXTDATE']
        target = 'LONGTIME'

        data = data.dropna()  # Remove rows with NaN values resulting from rolling and shifting

        X = data[features]
        y = data[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.dates_test = self.X_test['TXTDATE']

        numeric_features = ['Dia_da_Semana', 'Mês', 'Hora', 'LONGTIME_MA', 'LONGTIME_STD', 'LONGTIME_LAG_1', 'LONGTIME_LAG_2', 'LONGTIME_LAG_3']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ])

        self.X_train = preprocessor.fit_transform(self.X_train)
        self.X_test = preprocessor.transform(self.X_test)

        # Augment data
        self.X_train, self.y_train = self.augment_data(self.X_train, self.y_train)

        # Expand dimensions for GRU input
        self.X_train = np.expand_dims(self.X_train, axis=-1)
        self.X_test = np.expand_dims(self.X_test, axis=-1)

        # Ensure y_train and y_test are 2D
        self.y_train = np.expand_dims(self.y_train, axis=-1)
        self.y_test = np.expand_dims(self.y_test, axis=-1)
    
    def custom_scorer(self, model, X, y):
        y_pred = model.predict(X)
        return -mean_squared_error(y, y_pred)

    def run(self):
        self.build_data(self.data_set)

        param_distributions = {
            'units': [32, 64, 128],
            'activation': ['relu', 'tanh'],
            'dropout': [0.0, 0.2, 0.5]
        }
        self.logModel.show()
        model = KerasGRURegressor(input_shape=(self.X_train.shape[1], 1))

        search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=3, scoring=self.custom_scorer, verbose=1)
        search.fit(self.X_train, self.y_train)

        best_model = search.best_estimator_
        keras_model = best_model.model
        mse_list = []
        rmse_list = []
        # Gerar sumário do modelo
        self.logModel.hide()
        modelSu = ModelSummary(keras_model, 'GRU_model_summary.pdf', self.X_test.shape, self.y_test.shape)
        for epoch in range(40):
            best_model.fit(self.X_train, self.y_train, epochs=1, batch_size=64, validation_split=0.2, verbose=0)
            test_loss = best_model.model.evaluate(self.X_test, self.y_test, verbose=0)
            self.test_mse = test_loss[1]
            mse_list.append(self.test_mse)
            rmse_list.append(np.sqrt(self.test_mse))
            self.update_progress.emit((epoch + 1) / 40)
        self.save_model(best_model.model)
        test_loss = best_model.model.evaluate(self.X_test, self.y_test)
        test_mse = test_loss[1]
        mse_array = np.array(mse_list)  # Convertendo lista para array NumPy
        rmse_array = np.array(rmse_list)

        test_results = best_model.model.predict(self.X_test)
        dates_test_array = np.array(self.dates_test)
        y_test_array = np.array(self.y_test)

        # Garantir que y_true e y_pred são unidimensionais
        y_trueA = np.ravel(y_test_array)
        y_predA = np.ravel(test_results)

        # Calcular a diferença entre previstos e reais
        difference = y_predA - y_trueA

        r_squared = r2_score(self.y_test, test_results)

        df_results = pd.DataFrame({
            'Data': dates_test_array,
            'Valor_Real': y_test_array.flatten(),
            'Valor_Previsto': test_results.flatten()
        })

        df_results['Valor_Previsto_SMA'] = df_results['Valor_Previsto'].rolling(window=3, min_periods=1).mean()
        # Preparar os dados para o relatório
        models_and_results = {
            'GRU_model': (df_results, modelSu)
        }

        # Colete as métricas
        metrics = {
            'MSE': test_mse,
            'RMSE': np.sqrt(test_mse),
            'R²': r_squared
            # Adicione outras métricas conforme necessário
        }

        # Instanciar e salvar o relatório
        relatorio = RelatorioDosModelos(best_model, models_and_results, metrics)
        relatorio.save_reports_CSV_PDF()
        relatorio.save_shared_metrics()  # Adiciona as métricas ao CSV compartilhado
        relatorio.save_shared_metrics_list(mse_list,rmse_list,"Modelo GRU")
        relatorio.save_shared_difference_list(difference,"Modelo GRU")

        self.update_prediction_chart.emit(y_test_array, test_results, dates_test_array,"Modelo GRU")
        self.show_test_accuracy.emit(test_mse)
        self.update_metrics_chart.emit(test_mse, np.sqrt(test_mse),"Modelo GRU")
        self.update_metrics_chart_boxplot.emit(mse_array,rmse_array,"GRU")

    def add_moving_average_std(self, data, window_size=3):
        data['TXTDATE'] = pd.to_datetime(data['TXTDATE'])
        data.set_index('TXTDATE', inplace=True)
        data['LONGTIME_MA'] = data['LONGTIME'].rolling(window=window_size).mean()
        data['LONGTIME_STD'] = data['LONGTIME'].rolling(window=window_size).std()
        data.reset_index(inplace=True)
        return data

    def add_lag_features(self, data, lags=[1, 2, 3]):
        for lag in lags:
            data[f'LONGTIME_LAG_{lag}'] = data['LONGTIME'].shift(lag)
        return data

    def augment_data(self, X, y, augmentation_factor=5, noise_level=0.1):
        augmented_X, augmented_y = [X], [y]
        for _ in range(augmentation_factor):
            noise = np.random.normal(loc=0.0, scale=noise_level, size=X.shape)
            augmented_X.append(X + noise)
            augmented_y.append(y + np.random.normal(loc=0.0, scale=noise_level, size=y.shape))
        
        return np.concatenate(augmented_X), np.concatenate(augmented_y)
    
    def save_model(self,model):
        self.output_directory = "./ModelosComplilados"
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)
        h5_filename = os.path.join(self.output_directory,"gru_model.h5")
        model.save(h5_filename)
