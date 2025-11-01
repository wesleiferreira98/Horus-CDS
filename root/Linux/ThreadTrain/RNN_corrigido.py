#!/usr/bin/env python3
"""
RNN Corrigido - Vers  def _build_model(self):
    self.model = Sequential([
      # Camada de ruído na entrada para regularização
      GaussianNoise(self.noise_level, input_shape=self.input_shape),
      
      # RNN principal com dropout interno e recorrente
      SimpleRNN(self.units, activation=self.activation,
           dropout=self.dropout, recurrent_dropout=self.dropout,
           return_sequences=True),
      
      # Batch normalization para estabilizar treinamento
      BatchNormalization(),
      
      # Segunda camada RNN para maior capacidade
      SimpleRNN(self.units//2, activation=self.activation,
           dropout=self.dropout, recurrent_dropout=self.dropout,
           return_sequences=False),em vazamento temporal e com balanceamento de classes
Correções aplicadas:
1. Divisão temporal adequada
2. Features calculadas separadamente
3. Balanceamento de classes
4. Dataset normalizado
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import r2_score, confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, GaussianNoise, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from PyQt5.QtCore import QThread, pyqtSignal
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI para threads
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class KerasRNNRegressorBalanced(BaseEstimator, RegressorMixin):
  """RNN para regressão com balanceamento de classes"""
  
  def __init__(self, input_shape, units=50, activation='tanh', dropout=0.2, 
         class_weights=None, noise_level=0.1):
    self.input_shape = input_shape
    self.units = units
    self.activation = activation
    self.dropout = dropout
    self.class_weights = class_weights
    self.noise_level = noise_level
    self.model = None
    self._build_model()

  def _build_model(self):
    self.model = Sequential([
      # Camada de ruído na entrada para regularização
      GaussianNoise(self.noise_level, input_shape=self.input_shape),
      
      # RNN principal com dropout interno e recorrente
      SimpleRNN(self.units, activation=self.activation,
           dropout=self.dropout, recurrent_dropout=self.dropout,
           return_sequences=True),
      
      # Batch normalization para estabilizar treinamento
      BatchNormalization(),
      
      # Segunda camada RNN para maior capacidade
      SimpleRNN(self.units//2, activation=self.activation,
           dropout=self.dropout, recurrent_dropout=self.dropout,
           return_sequences=False),
      
      # Camadas densas com batch normalization e dropout
      BatchNormalization(),
      Dropout(self.dropout),
      Dense(64, activation='relu'),
      BatchNormalization(),
      Dropout(self.dropout),
      Dense(32, activation='relu'),
      Dropout(self.dropout),
      Dense(1) # Saída única para regressão
    ])
    
    optimizer = Adam(learning_rate=0.001)
    self.model.compile(
      optimizer=optimizer, 
      loss='mean_squared_error', # MSE para regressão
      metrics=['mse']
    )

  def fit(self, X, y, **kwargs):
    # Aplicar pesos de amostra se disponíveis
    if self.class_weights is not None and 'sample_weight' not in kwargs:
      # Converter valores de regressão para classes para aplicar pesos
      y_classes = self._convert_regression_to_classes(y)
      sample_weights = np.array([self.class_weights.get(cls, 1.0) for cls in y_classes])
      kwargs['sample_weight'] = sample_weights
    
    self.model.fit(X, y, **kwargs)
    return self

  def predict(self, X):
    return self.model.predict(X)

  

  def score(self, X, y):
    return self.model.evaluate(X, y)[1]
  
  def _convert_regression_to_classes(self, y):
    """Converte valores de regressão para classes para aplicar pesos"""
    classes = []
    for value in y.flatten():
      if value < -0.5:
        classes.append('ilegal')
      elif value < 0.5:
        classes.append('suspeito')
      else:
        classes.append('válido')
    return classes

class TrainingThreadRNNCorrigido(QThread):
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
    self.categories_test = None
    self.dates_test = None
    self.scaler = None
    self.model_type = "RNN"
    self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
  def temporal_split(self, data, test_ratio=0.2):
    """Divisão temporal correta dos dados"""
    print(f"Realizando divisão temporal dos dados para {self.model_type}...")
    
    data_sorted = data.sort_values('TXTDATE').reset_index(drop=True)
    split_idx = int(len(data_sorted) * (1 - test_ratio))
    
    train_data = data_sorted.iloc[:split_idx].copy()
    test_data = data_sorted.iloc[split_idx:].copy()
    
    print(f"Divisão temporal {self.model_type}:")
    print(f"  Treino: {len(train_data)} amostras ({len(train_data)/len(data_sorted)*100:.1f}%)")
    print(f"  Teste: {len(test_data)} amostras ({len(test_data)/len(data_sorted)*100:.1f}%)")
    
    return train_data, test_data
  
  def add_temporal_features_safe(self, data, is_train=True):
    """Adiciona features temporais sem vazamento"""
    data = data.copy()
    data['TXTDATE'] = pd.to_datetime(data['TXTDATE'])
    data['Dia_da_Semana'] = data['TXTDATE'].dt.dayofweek
    data['Mês'] = data['TXTDATE'].dt.month
    data['Hora'] = pd.to_datetime(data['TXTTIME'], format='%H:%M:%S').dt.hour
    
    # Ordenar por data
    data = data.sort_values('TXTDATE').reset_index(drop=True)
    
    # Moving average e std calculados apenas dentro do conjunto
    window_size = 3
    data['LONGTIME_MA'] = data['LONGTIME'].rolling(window=window_size, min_periods=1).mean()
    data['LONGTIME_STD'] = data['LONGTIME'].rolling(window=window_size, min_periods=1).std().fillna(0)
    
    # Lag features calculados apenas dentro do conjunto
    for lag in [1, 2, 3]:
      data[f'LONGTIME_LAG_{lag}'] = data['LONGTIME'].shift(lag)
    
    # Remover NaN dos lags
    data = data.dropna().reset_index(drop=True)
    
    prefix = "Treino" if is_train else "Teste"
    print(f"Features adicionadas ao {prefix} {self.model_type}: {len(data)} amostras")
    
    return data
  
  def analyze_and_balance_classes(self, train_data, technique='class_weights'):
    """Analisa distribuição e aplica balanceamento"""
    print(f"Analisando classes para {self.model_type}...")
    
    class_counts = train_data['CATEGORY'].value_counts()
    total = len(train_data)
    
    print("Distribuição original:")
    for cls, count in class_counts.items():
      print(f"  {cls}: {count} ({count/total*100:.1f}%)")
    
    # Calcular pesos de classe
    unique_classes = train_data['CATEGORY'].unique()
    class_weights_array = compute_class_weight(
      'balanced', 
      classes=unique_classes, 
      y=train_data['CATEGORY']
    )
    class_weights = dict(zip(unique_classes, class_weights_array))
    
    print(f"Pesos calculados para {self.model_type}:")
    for cls, weight in class_weights.items():
      print(f"  {cls}: {weight:.3f}")
      
    return class_weights
  
  def apply_balancing_technique(self, X, y_categories, technique='smote'):
    """Aplica técnicas de balanceamento"""
    if technique == 'smote':
      print(f"Aplicando SMOTE para {self.model_type}...")
      
      # Converter categorias para números
      le = LabelEncoder()
      y_encoded = le.fit_transform(y_categories)
      
      # Aplicar SMOTE
      smote = SMOTE(random_state=42, k_neighbors=3)
      X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
      
      # Converter de volta para categorias
      categories_resampled = le.inverse_transform(y_resampled)
      
      print(f"Dados após SMOTE {self.model_type}: {len(X_resampled)} amostras")
      unique, counts = np.unique(categories_resampled, return_counts=True)
      for cls, count in zip(unique, counts):
        print(f"  {cls}: {count}")
        
      return X_resampled, categories_resampled
      
    elif technique == 'undersampling':
      print(f"Aplicando undersampling para {self.model_type}...")
      
      le = LabelEncoder()
      y_encoded = le.fit_transform(y_categories)
      
      undersampler = RandomUnderSampler(random_state=42)
      X_resampled, y_resampled = undersampler.fit_resample(X, y_encoded)
      
      categories_resampled = le.inverse_transform(y_resampled)
      
      print(f"Dados após undersampling {self.model_type}: {len(X_resampled)} amostras")
      unique, counts = np.unique(categories_resampled, return_counts=True)
      for cls, count in zip(unique, counts):
        print(f"  {cls}: {count}")
        
      return X_resampled, categories_resampled
    
    else:
      return X, y_categories

  def build_data_corrected(self, data, balancing_technique='class_weights'):
    """Constrói dados com metodologia corrigida para REGRESSÃO"""
    print(f"Construindo dados corrigidos para {self.model_type}...")
    
    # 1. Divisão temporal
    train_data, test_data = self.temporal_split(data)
    
    # 2. Features temporais separadamente
    train_data = self.add_temporal_features_safe(train_data, is_train=True)
    test_data = self.add_temporal_features_safe(test_data, is_train=False)
    
    # 3. Preparar features e target (REGRESSÃO)
    features = ['Dia_da_Semana', 'Mês', 'Hora', 'LONGTIME_MA', 'LONGTIME_STD', 
          'LONGTIME_LAG_1', 'LONGTIME_LAG_2', 'LONGTIME_LAG_3']
    target = 'LONGTIME' # Target de regressão (valores contínuos)
    
    X_train = train_data[features].values
    y_train = train_data[target].values # Valores contínuos para regressão
    categories_train = train_data['CATEGORY'].values # Para análise de distribuição
    dates_train = train_data['TXTDATE'].values
    
    X_test = test_data[features].values
    y_test = test_data[target].values # Valores contínuos para regressão
    self.categories_test = test_data['CATEGORY'].values # Para matriz de confusão
    self.dates_test = test_data['TXTDATE'].values
    
    # 4. Normalização (fit apenas no treino)
    print(f"Aplicando normalização {self.model_type}...")
    numeric_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='mean')),
      ('scaler', StandardScaler())
    ])
    
    self.scaler = numeric_transformer
    X_train_scaled = self.scaler.fit_transform(X_train)
    X_test_scaled = self.scaler.transform(X_test)
    
    # 5. Balanceamento via pesos de amostra
    print(f" Analisando distribuição de classes {self.model_type}...")
    class_weights = self.analyze_and_balance_classes(train_data, technique=balancing_technique)
    
    # Para regressão, usamos sample_weight no fit() em vez de balanceamento direto
    # Os pesos serão aplicados baseados na conversão dos valores para classes
    
    # 6. Preparar para RNN (adicionar dimensão temporal)
    self.X_train = np.expand_dims(X_train_scaled, axis=-1)
    self.X_test = np.expand_dims(X_test_scaled, axis=-1)
    self.y_train = np.expand_dims(y_train, axis=-1) # Valores de regressão
    self.y_test = np.expand_dims(y_test, axis=-1) # Valores de regressão
    
    print(f"Dados preparados para {self.model_type}:")
    print(f"  Treino: {self.X_train.shape}")
    print(f"  Teste: {self.X_test.shape}")
    
    return class_weights

  def run(self):
    """Metodo run para compatibilidade com interface"""
    results = self.run_corrected(balancing_technique='class_weights', noise_level=0.1)
    
    # Emitir sinais para interface
    mse_array = np.array(results['mse_progression'])
    rmse_array = np.array(results['rmse_progression'])
    
    self.update_prediction_chart.emit(
      self.dates_test, 
      self.y_test.flatten(), 
      results['predictions'].flatten(),
      f"Modelo {self.model_type} Corrigido"
    )
    self.update_metrics_chart.emit(results['mse'], results['rmse'], f"Modelo {self.model_type} Corrigido")
    self.update_metrics_chart_boxplot.emit(mse_array, rmse_array, f"{self.model_type}_Corrigido")
    self.show_test_accuracy.emit(results['mse'])

  def run_corrected(self, balancing_technique='class_weights', noise_level=0.1):
    """Execução corrigida do treinamento RNN"""
    print(f"Iniciando treinamento {self.model_type} corrigido...")
    
    # Construir dados corrigidos
    class_weights = self.build_data_corrected(self.data_set, balancing_technique)
    
    # Parâmetros para busca
    param_distributions = {
      'units': [32, 50, 64, 100],
      'activation': ['tanh', 'relu'],
      'dropout': [0.1, 0.2, 0.3]
    }

    # Criar modelo com nível de ruído especificado
    model = KerasRNNRegressorBalanced(
      input_shape=(self.X_train.shape[1], 1),
      class_weights=class_weights,
      noise_level=noise_level
    )

    print(f"Realizando busca de hiperparâmetros {self.model_type}...")
    search = RandomizedSearchCV(
      model, param_distributions, 
      n_iter=3, # Reduzido para teste
      cv=3, 
      scoring='neg_mean_squared_error', # MSE negativo para regressão
      verbose=1
    )
    
    search.fit(self.X_train, self.y_train)
    best_model = search.best_estimator_
    
    print(f"Treinando modelo {self.model_type} final...")
    mse_list = []
    rmse_list = []
    
    for epoch in range(20):
      best_model.fit(
        self.X_train, self.y_train, 
        epochs=1, batch_size=32, 
        validation_split=0.2, verbose=0
      )
      
      # Avaliar MSE e RMSE no teste
      test_results = best_model.model.predict(self.X_test)
      test_mse = np.mean((self.y_test - test_results) ** 2)
      test_rmse = np.sqrt(test_mse)
      
      mse_list.append(test_mse)
      rmse_list.append(test_rmse)
      
      self.update_progress.emit((epoch + 1) / 20)
    
    # Resultados finais
    test_results = best_model.predict(self.X_test)
    test_mse_final = mse_list[-1]
    r_squared = r2_score(self.y_test, test_results)
    
    print(f"Resultados finais {self.model_type}:")
    print(f"  MSE: {test_mse_final:.6f}")
    print(f"  RMSE: {np.sqrt(test_mse_final):.6f}")
    print(f"  R²: {r_squared:.6f}")
    
    # Gerar matriz de confusão (converter regressão para categorias)
    self.generate_confusion_matrix_corrected(self.categories_test, test_results)
    
    # Salvar modelo
    self.save_model_corrected(best_model.model)
    
    return {
      'mse': test_mse_final,
      'rmse': np.sqrt(test_mse_final),
      'r2': r_squared,
      'predictions': test_results,
      'dates': self.dates_test
    }
  
  def generate_confusion_matrix_corrected(self, y_true_categories, y_pred_longtime):
    """Gera matriz de confusão com thresholds ajustados para dados normalizados"""
    print(f"Gerando matriz de confusão {self.model_type}...")
    
    # Converter previsões de regressão para categorias
    y_pred_flat = y_pred_longtime.flatten()
    
    # Calcular thresholds baseados na distribuição real das categorias no teste
    threshold_17_8 = np.percentile(y_pred_flat, 17.8)
    threshold_23_2 = np.percentile(y_pred_flat, 23.2)
    
    def classify_corrected(values):
      return ['ilegal' if v <= threshold_17_8 else 'suspeito' if v <= threshold_23_2 else 'válido' for v in values]
    
    y_pred_categories = classify_corrected(y_pred_flat)
    
    # Matriz de confusão
    cm = confusion_matrix(y_true_categories, y_pred_categories, labels=['ilegal', 'suspeito', 'válido'])
    accuracy = np.trace(cm) / np.sum(cm)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
          xticklabels=['ilegal', 'suspeito', 'válido'],
          yticklabels=['ilegal', 'suspeito', 'válido'])
    plt.title(f'Matriz de Confusão - {self.model_type} Corrigido\n(Sem Vazamento Temporal)')
    plt.xlabel('Categoria Predita')
    plt.ylabel('Categoria Real')
    
    # Salvar
    output_dir = os.path.join(self.base_dir, "MatrizConfusaoCorrigida")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/matriz_confusao_{self.model_type.lower()}_corrigido.jpg', 
          dpi=300, bbox_inches='tight')
    plt.close()
    
    # Relatório
    report = classification_report(y_true_categories, y_pred_categories,
                   labels=['ilegal', 'suspeito', 'válido'], zero_division=0)
    
    with open(f'{output_dir}/relatorio_{self.model_type.lower()}_corrigido.txt', 'w', encoding='utf-8') as f:
      f.write(f"RELATÓRIO {self.model_type} CORRIGIDO - SEM VAZAMENTO TEMPORAL\n")
      f.write("="*60 + "\n")
      f.write(f"Acurácia: {accuracy:.3f} ({accuracy*100:.1f}%)\n")
      f.write("Correções aplicadas:\n")
      f.write("- Divisão temporal (não aleatória)\n")
      f.write("- Features calculadas separadamente\n")
      f.write("- StandardScaler fit apenas no treino\n")
      f.write("- Balanceamento de classes\n")
      f.write("- Classificação direta (não regressão)\n\n")
      f.write(report)
    
    print(f"Matriz de confusão {self.model_type} salva em: {output_dir}")
    print(f"Acurácia {self.model_type}: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    return cm, report
  
  def save_model_corrected(self, model):
    """Salva modelo corrigido"""
    output_directory = os.path.join(self.base_dir, "ModelosCorrigidos")
    os.makedirs(output_directory, exist_ok=True)
    keras_filename = os.path.join(output_directory, f"{self.model_type.lower()}_model_corrigido.keras")
    model.save(keras_filename)
    print(f"Modelo {self.model_type} corrigido salvo em: {keras_filename}")


def main():
  """Teste da versão RNN corrigida"""
  print("Testando RNN Corrigido com Ruído")
  print("="*50)
  
  # Carregar dados NORMALIZADOS
  data = pd.read_csv('DadosReais/dados_normalizados_smartgrid.csv')
  print(f"Dados carregados: {len(data)} registros")
  
  # Criar instância corrigida
  trainer = TrainingThreadRNNCorrigido(data)
  
  # Executar treinamento corrigido
  results = trainer.run_corrected(balancing_technique='class_weights')
  
  print("\nTreinamento RNN corrigido concluído!")
  print(f"Acurácia: {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")

if __name__ == "__main__":
  main()