import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from PyQt5.QtCore import QThread, pyqtSignal

from View.LogTrain import LogTrain
from modelSummary.RelatorioDosModelos import RelatorioDosModelos

class RandomForestRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.model = self._build_model()

    def _build_model(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

class TrainingThreadRandomForest(QThread):
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
        self.logModel = LogTrain("RF","03:10")

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

    def run(self):
        self.build_data(self.data_set)
        self.logModel.show()
        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        model = RandomForestRegressorWrapper()

        search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=1)
        search.fit(self.X_train, self.y_train)

        best_model = search.best_estimator_
        self.logModel.hide()
        for epoch in range(1):
            best_model.fit(self.X_train, self.y_train)
            test_score = best_model.score(self.X_test, self.y_test)
            self.test_score = test_score
            self.update_progress.emit(1)

        test_score = best_model.score(self.X_test, self.y_test)

        test_results = best_model.predict(self.X_test)
        dates_test_array = np.array(self.dates_test)
        y_test_array = np.array(self.y_test)
        self.save_model(best_model.model)

        df_results = pd.DataFrame({
            'Data': dates_test_array,
            'Valor_Real': y_test_array.flatten(),
            'Valor_Previsto': test_results.flatten()
        })

        df_results['Valor_Previsto_SMA'] = df_results['Valor_Previsto'].rolling(window=3, min_periods=1).mean()
       
        print("Valores previstos pelo modelo:")
        for i, prediction in enumerate(test_results):
            print(f"Data: {dates_test_array[i]}, Valor real: {y_test_array[i]}, Valor previsto: {prediction}")
        mse = np.mean(( df_results['Valor_Real'] -  df_results['Valor_Previsto']) ** 2)
        # Calculando o RMSE
        rmse = np.sqrt(mse)

        mse_list = []
        rmse_list = []
        mse_list.append(mse)
        rmse_list.append(rmse)
        mse_array = np.array(mse_list)  # Convertendo lista para array NumPy
        rmse_array = np.array(rmse_list)  # Convertendo lista para array NumPy
        self.update_prediction_chart.emit(y_test_array, test_results, dates_test_array,"Modelo RF")
        self.show_test_accuracy.emit(mse)
        self.update_metrics_chart.emit(mse,rmse,"Modelo RF")
        self.update_metrics_chart_boxplot.emit(mse_array,rmse_array,"RF")

        models_and_results = {
            'RandomForest_model': (df_results, None)  # Não há sumário de modelo específico
        }

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
        }

        relatorio = RelatorioDosModelos(best_model, models_and_results, metrics)
        relatorio.save_metrics_pdf_KNN_ARIMA_RF("RandomForest_metrics.pdf")
        relatorio.save_reports_CSV_KNN_ARIMA_RF("RandomForest_results.csv")
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

    def augment_data(self, X, y):
        augmented_X = np.copy(X)
        augmented_y = np.copy(y)

        for i in range(X.shape[0]):
            if np.random.rand() < 0.5:
                augmented_X = np.vstack([augmented_X, np.flipud(X[i:i+1])])
                augmented_y = np.hstack([augmented_y, y[i:i+1]])

        return augmented_X, augmented_y
    
    def save_model(self,model):
        self.output_directory = "./ModelosComplilados"
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)
        h5_filename = os.path.join(self.output_directory,"rf_model.h5")
        model.save(h5_filename)
