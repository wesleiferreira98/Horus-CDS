import os
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras import Input  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GaussianNoise  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.metrics import r2_score
from View.LogTrain import LogTrain
from modelSummary.ModelSummary import ModelSummary
from modelSummary.RelatorioDosModelos import RelatorioDosModelos


class KerasMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim, num_hidden_layers=2, num_neurons=64, activation='relu', dropout=0.0, noise_std=0.35):
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons = num_neurons
        self.activation = activation
        self.dropout = dropout
        self.noise_std = noise_std
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))
        model.add(GaussianNoise(self.noise_std))
        model.add(Dense(self.num_neurons, activation=self.activation))

        for _ in range(self.num_hidden_layers - 1):
            model.add(Dense(self.num_neurons, activation=self.activation))
            if self.dropout > 0:
                model.add(Dropout(self.dropout))

        model.add(Dense(1))
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
        return model

    def fit(self, X, y, **kwargs):
        kwargs.setdefault('verbose', 0)
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]


class TrainingThreadMLP(QThread):
    show_test_accuracy = pyqtSignal(float)
    update_progress = pyqtSignal(float)
    update_prediction_chart = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, str)
    update_metrics_chart_boxplot = pyqtSignal(np.ndarray, np.ndarray, str)
    update_metrics_chart = pyqtSignal(float, float, str)

    def __init__(self, data_set):
        super().__init__()
        self.data_set = data_set
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.dates_test = None
        self.categories_test = None
        self.category_thresholds = None
        self.logModel = LogTrain("Horus-V0", "02:30")
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_type = "Horus-V0"
        self.fast_mode = os.getenv("HORUS_FAST_CURVES", "0") == "1"

    def temporal_split(self, data, test_ratio=0.2):
        data = data.copy()
        data['DATETIME'] = pd.to_datetime(data['TXTDATE'].astype(str) + ' ' + data['TXTTIME'].astype(str))
        data_clean = data.drop_duplicates(subset=['DATETIME'], keep='first').copy()
        data_sorted = data_clean.sort_values('DATETIME').reset_index(drop=True)
        split_idx = int(len(data_sorted) * (1 - test_ratio))
        train_data = data_sorted.iloc[:split_idx].copy()
        test_data = data_sorted.iloc[split_idx:].copy()

        if self.fast_mode:
            train_cap = min(len(train_data), 1200)
            test_cap = min(len(test_data), 300)
            train_data = train_data.tail(train_cap).copy()
            test_data = test_data.head(test_cap).copy()

        return train_data, test_data

    def add_temporal_features_safe(self, data, is_train=True):
        data = data.copy()
        data['TXTDATE'] = pd.to_datetime(data['TXTDATE'])
        data['Dia_da_Semana'] = data['TXTDATE'].dt.dayofweek
        data['Mês'] = data['TXTDATE'].dt.month
        data['Hora'] = pd.to_datetime(data['TXTTIME'], format='%H:%M:%S').dt.hour
        data = data.sort_values('DATETIME').reset_index(drop=True)
        data['LONGTIME_MA'] = data['LONGTIME'].rolling(window=3, min_periods=1).mean()
        data['LONGTIME_STD'] = data['LONGTIME'].rolling(window=3, min_periods=1).std().fillna(0)

        for lag in [1, 2, 3]:
            data[f'LONGTIME_LAG_{lag}'] = data['LONGTIME'].shift(lag)

        data = data.dropna().reset_index(drop=True)
        return data

    def build_data(self, data):
        self.category_thresholds = self.compute_category_thresholds(data)
        train_data, test_data = self.temporal_split(data)
        train_data = self.add_temporal_features_safe(train_data, is_train=True)
        test_data = self.add_temporal_features_safe(test_data, is_train=False)

        features = [
            'Dia_da_Semana',
            'Mês',
            'Hora',
            'LONGTIME_MA',
            'LONGTIME_STD',
            'LONGTIME_LAG_1',
            'LONGTIME_LAG_2',
            'LONGTIME_LAG_3',
            'TXTDATE'
        ]
        target = 'LONGTIME'

        self.dates_test = test_data['TXTDATE']
        self.categories_test = test_data['CATEGORY'].values
        self.y_train = train_data[target]
        self.y_test = test_data[target]

        X_train = train_data[features]
        X_test = test_data[features]

        numeric_features = [
            'Dia_da_Semana',
            'Mês',
            'Hora',
            'LONGTIME_MA',
            'LONGTIME_STD',
            'LONGTIME_LAG_1',
            'LONGTIME_LAG_2',
            'LONGTIME_LAG_3'
        ]
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ]
        )

        self.X_train = preprocessor.fit_transform(X_train)
        self.X_test = preprocessor.transform(X_test)
        self.X_train, self.y_train = self.augment_data(self.X_train, self.y_train)

        self.X_train = np.asarray(self.X_train, dtype=np.float32)
        self.X_test = np.asarray(self.X_test, dtype=np.float32)
        self.y_train = np.asarray(self.y_train, dtype=np.float32).reshape(-1, 1)
        self.y_test = np.asarray(self.y_test, dtype=np.float32).reshape(-1, 1)

    def run(self):
        self.build_data(self.data_set)

        if self.fast_mode:
            print("HORUS_FAST_CURVES=1 detectado: treino reduzido para gerar curvas ROC/PR.")
            n_iter = 3
            epochs = 8
            batch_size = 32
            self.logModel.update_estimated_time("00:40")
        else:
            n_iter = 10
            epochs = 40
            batch_size = 64

        param_distributions = {
            'num_hidden_layers': [1, 2, 3],
            'num_neurons': [32, 64, 128],
            'activation': ['relu', 'tanh'],
            'dropout': [0.1, 0.2, 0.3],
            'noise_std': [0.2, 0.3, 0.4]
        }

        self.logModel.show()
        model = KerasMLPRegressor(input_dim=self.X_train.shape[1])
        search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=n_iter,
            cv=3,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=1
        )
        search.fit(self.X_train, self.y_train)

        best_model = search.best_estimator_
        keras_model = best_model.model
        mse_list = []
        rmse_list = []
        modelSu = ModelSummary(keras_model, 'horus_v0_model_summary.pdf', self.X_test.shape, self.y_test.shape)
        self.logModel.hide()

        for epoch in range(epochs):
            best_model.fit(self.X_train, self.y_train, epochs=1, batch_size=batch_size, validation_split=0.2, verbose=0)
            test_loss = best_model.model.evaluate(self.X_test, self.y_test, verbose=0)
            self.test_mse = test_loss[1]
            mse_list.append(self.test_mse)
            rmse_list.append(np.sqrt(self.test_mse))
            self.update_progress.emit((epoch + 1) / epochs)

        self.save_model(best_model.model)
        test_loss = best_model.model.evaluate(self.X_test, self.y_test, verbose=0)
        test_mse = test_loss[1]
        test_results = best_model.model.predict(self.X_test, verbose=0)
        dates_test_array = np.array(self.dates_test)
        y_test_array = np.array(self.y_test)
        mse_array = np.array(mse_list)
        rmse_array = np.array(rmse_list)
        r_squared = r2_score(self.y_test, test_results)

        y_trueA = np.ravel(y_test_array)
        y_predA = np.ravel(test_results)
        difference = y_predA - y_trueA

        df_results = pd.DataFrame({
            'Data': dates_test_array,
            'Valor_Real': y_test_array.flatten(),
            'Valor_Previsto': test_results.flatten()
        })
        df_results['Valor_Previsto_SMA'] = df_results['Valor_Previsto'].rolling(window=3, min_periods=1).mean()

        models_and_results = {
            'Horus_V0_model': (df_results, modelSu)
        }

        metrics = {
            'MSE': test_mse,
            'RMSE': np.sqrt(test_mse),
            'R²': r_squared
        }

        relatorio = RelatorioDosModelos(best_model, models_and_results, metrics, model_type="Old")
        relatorio.save_reports_CSV_PDF()
        relatorio.save_shared_metrics()
        relatorio.save_shared_metrics_list(mse_list, rmse_list, "Horus-V0")
        relatorio.save_shared_difference_list(difference, "Horus-V0")
        relatorio.save_roc_pr_curves_regression(self.categories_test, test_results, "Horus-V0", self.category_thresholds)

        self.update_prediction_chart.emit(y_test_array, test_results, dates_test_array, "Horus-V0")
        self.show_test_accuracy.emit(test_mse)
        self.update_metrics_chart.emit(test_mse, np.sqrt(test_mse), "Horus-V0")
        self.update_metrics_chart_boxplot.emit(mse_array, rmse_array, "Horus-V0")

    def save_model(self, model):
        self.output_directory = os.path.join(self.base_dir, "DadosDoPostreino/ModelosOlds/ModelosComplilados")
        os.makedirs(self.output_directory, exist_ok=True)
        h5_filename = os.path.join(self.output_directory, "horus_v0_model.h5")
        model.save(h5_filename)
        print(f"Modelo Horus-V0 salvo em: {h5_filename}")

    def augment_data(self, X, y, augmentation_factor=3, noise_level=0.25):
        augmented_X, augmented_y = [X], [np.asarray(y)]
        for _ in range(augmentation_factor):
            noise = np.random.normal(loc=0.0, scale=noise_level, size=X.shape)
            augmented_X.append(X + noise)
            augmented_y.append(
                np.asarray(y) + np.random.normal(loc=0.0, scale=noise_level * 0.2, size=np.asarray(y).shape)
            )
        return np.concatenate(augmented_X), np.concatenate(augmented_y)

    def compute_category_thresholds(self, data):
        categorized = data.dropna(subset=['LONGTIME', 'CATEGORY'])
        ilegal = categorized[categorized['CATEGORY'] == 'ilegal']['LONGTIME']
        suspeito = categorized[categorized['CATEGORY'] == 'suspeito']['LONGTIME']
        valido = categorized[categorized['CATEGORY'] == 'válido']['LONGTIME']

        low_threshold = (ilegal.max() + suspeito.min()) / 2 if not ilegal.empty and not suspeito.empty else categorized['LONGTIME'].quantile(0.2)
        high_threshold = (suspeito.max() + valido.min()) / 2 if not suspeito.empty and not valido.empty else categorized['LONGTIME'].quantile(0.8)
        return float(low_threshold), float(high_threshold)
