import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras import layers, optimizers  # type: ignore
from tcn import TCN
from PyQt5.QtCore import QThread, pyqtSignal

class TrainingThreadTCN(QThread):
    show_test_accuracy = pyqtSignal(float)
    update_progress = pyqtSignal(float)
    update_prediction_chartCNN = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    update_metrics_chart = pyqtSignal(float, float)

    def __init__(self, data_set):
        super().__init__()
        self.data_set = data_set
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def build_data(self, data):
        data['TXTDATE'] = pd.to_datetime(data['TXTDATE'])
        data['Dia_da_Semana'] = data['TXTDATE'].dt.dayofweek
        data['Mês'] = data['TXTDATE'].dt.month
        data['Hora'] = pd.to_datetime(data['TXTTIME'], format='%H:%M:%S').dt.hour

        features = ['Dia_da_Semana', 'Mês', 'Hora','TXTDATE']
        target = 'LONGTIME'

        X = data[features]
        y = data[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.dates_test = self.X_test['TXTDATE']

        numeric_features = ['Dia_da_Semana', 'Mês', 'Hora']
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

    def run(self):
        self.build_data(self.data_set)

        param_distributions = {
            'nb_filters': [32, 64, 128],
            'kernel_size': [2, 3, 4],
            'nb_stacks': [1, 2, 3],
            'dilations': [[1, 2, 4, 8], [1, 2, 4, 8, 16]],
            'activation': ['relu', 'tanh'],
            'use_skip_connections': [True, False]
        }

        model = Sequential([
            TCN(input_shape=(self.X_train.shape[1], 1)),
            layers.Dense(1)
        ])

        optimizer = optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

        search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=3, scoring='neg_mean_squared_error', verbose=1)
        search.fit(self.X_train, self.y_train)

        best_model = search.best_estimator_

        for epoch in range(40):
            best_model.fit(self.X_train, self.y_train, epochs=1, batch_size=64, validation_split=0.2, verbose=0)
            test_loss = best_model.evaluate(self.X_test, self.y_test, verbose=0)
            self.test_mse = test_loss[1]
            self.update_progress.emit((epoch + 1) / 40)

        test_loss = best_model.evaluate(self.X_test, self.y_test)
        test_mse = test_loss[1]

        test_results = best_model.predict(self.X_test)
        dates_test_array = np.array(self.dates_test)
        y_test_array = np.array(self.y_test)

        df_results = pd.DataFrame({
            'Data': dates_test_array,
            'Valor_Real': y_test_array.flatten(),
            'Valor_Previsto': test_results.flatten()
        })

        df_results['Valor_Previsto_SMA'] = df_results['Valor_Previsto'].rolling(window=3, min_periods=1).mean()
        df_results.to_csv('previsoes_tcn.csv', index=False)

        print("Valores previstos pelo modelo:")
        for i, prediction in enumerate(test_results):
            print(f"Data: {dates_test_array[i]}, Valor real: {y_test_array[i]}, Valor previsto: {prediction}")

        self.update_prediction_chartCNN.emit(y_test_array, test_results, dates_test_array)
        self.show_test_accuracy.emit(test_mse)
        self.update_metrics_chart.emit(test_mse, np.sqrt(test_mse))
