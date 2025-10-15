import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, SimpleRNN # type: ignore
from tensorflow.keras import optimizers # type: ignore
from PyQt5.QtCore import QThread, pyqtSignal
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from contextlib import redirect_stdout
from sklearn.metrics import r2_score

from View.LogTrain import LogTrain
from modelSummary.ModelSummary import ModelSummary
from modelSummary.RelatorioDosModelos import RelatorioDosModelos

class KerasRNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_shape, units=32):
        self.input_shape = input_shape
        self.units = units
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            SimpleRNN(units=self.units, input_shape=self.input_shape),
            Dense(1)
        ])
        optimizer = optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
        return model

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.evaluate(X, y)[1]

class TrainingThreadRNN(QThread):
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
        self.logModel = LogTrain("RNN","01:51")

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

        # Apply SMOTE for data balancing
        self.X_train, self.y_train = self.apply_smote(self.X_train, self.y_train)

        # Expand dimensions for RNN input
        self.X_train = np.expand_dims(self.X_train, axis=-1)
        self.X_test = np.expand_dims(self.X_test, axis=-1)

        # Ensure y_train and y_test are 2D
        self.y_train = np.expand_dims(self.y_train, axis=-1)
        self.y_test = np.expand_dims(self.y_test, axis=-1)
    
    

    def update_progressIN(self,search):
        means = search.cv_results_['mean_test_score']
        stds = search.cv_results_['std_test_score']
        for mean, std in zip(means, stds):
            progress = mean * -1  # Invert sign
            print(progress)
            self.update_progress.emit(progress)

    def run(self):
        self.build_data(self.data_set)
        self.logModel.show()
        param_distributions = {
            'units': [32, 64, 128]
        }

        model = KerasRNNRegressor(input_shape=(self.X_train.shape[1], 1))


        search = RandomizedSearchCV(model, param_distributions, n_iter=5, cv=3, scoring='neg_mean_squared_error', verbose=1)
        search.on_epoch_end = self.update_progressIN  # Connect the signal
        search.fit(self.X_train, self.y_train)

        best_model = search.best_estimator_

         # Extract the Keras model from the best_model
        mse_list = []
        rmse_list = []
        keras_model = best_model.model
        modelSu = ModelSummary(keras_model,'rnn_model_summary.pdf',self.X_test.shape,self.y_test.shape)
        self.logModel.hide()
        for epoch in range(40):
            best_model.fit(self.X_train, self.y_train, epochs=1, batch_size=64, validation_split=0.2, verbose=0)
            test_loss = keras_model.evaluate(self.X_test, self.y_test, verbose=0)
            self.test_mse = test_loss[1]
            mse_list.append(self.test_mse)
            rmse_list.append(np.sqrt(self.test_mse))
            self.update_progress.emit((epoch + 1) / 40)

        test_loss = keras_model.evaluate(self.X_test, self.y_test, verbose=0)

        print(mse_list)
        test_mse = test_loss[1]
        self.save_model(best_model.model)
        test_results = best_model.model.predict(self.X_test)
        dates_test_array = np.array(self.dates_test)
        y_test_array = np.array(self.y_test)
        mse_array = np.array(mse_list)  # Convertendo lista para array NumPy
        rmse_array = np.array(rmse_list)  # Convertendo lista para array NumPy
        
        # Garantir que y_true e y_pred são unidimensionais
        y_trueA = np.ravel(y_test_array)
        y_predA = np.ravel(test_results)

        # Calcular a diferença entre previstos e reais
        difference = y_predA - y_trueA
        df_results = pd.DataFrame({
            'Data': dates_test_array,
            'Valor_Real': y_test_array.flatten(),
            'Valor_Previsto': test_results.flatten()
        })
        r_squared = r2_score(self.y_test, test_results)

        df_results['Valor_Previsto_SMA'] = df_results['Valor_Previsto'].rolling(window=3, min_periods=1).mean()
        # Preparar os dados para o relatório
        models_and_results = {
            'RNN_model': (df_results, modelSu)
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
        relatorio.save_shared_metrics()
        relatorio.save_shared_metrics_list(mse_list,rmse_list,"Modelo RNN")
        relatorio.save_shared_difference_list(difference,"Modelo RNN")
        
        self.show_test_accuracy.emit(test_mse)
        self.update_metrics_chart.emit(test_mse, np.sqrt(test_mse),"Modelo RNN")
        self.update_prediction_chart.emit(y_test_array, test_results, dates_test_array,"Modelo RNN")
        self.update_metrics_chart_boxplot.emit(mse_array,rmse_array,"RNN")

    def add_moving_average_std(self,data, window_size=3):
        data['TXTDATE'] = pd.to_datetime(data['TXTDATE'])
        data.set_index('TXTDATE', inplace=True)
        data['LONGTIME_MA'] = data['LONGTIME'].rolling(window=window_size).mean()
        data['LONGTIME_STD'] = data['LONGTIME'].rolling(window=window_size).std()
        data.reset_index(inplace=True)
        return data

    def add_lag_features(self,data, lags=[1, 2, 3]):
        for lag in lags:
            data[f'LONGTIME_LAG_{lag}'] = data['LONGTIME'].shift(lag)
        return data

    def create_windows(self,data, window_size=3):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:(i + window_size)])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

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
            print("Aplicando data augmentation tradicional como fallback...")
            return self.augment_data(X, y)

    def augment_data(self,X, y, augmentation_factor=5, noise_level=0.8):
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
        h5_filename = os.path.join(self.output_directory,"rnn_model.h5")
        model.save(h5_filename)



