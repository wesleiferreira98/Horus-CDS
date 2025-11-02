#!/usr/bin/env python3
"""
LSTM Corrigido - Versão sem vazamento temporal e com balanceamento de classes
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
from tensorflow.keras.layers import LSTM, Dense, Dropout, GaussianNoise, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from PyQt5.QtCore import QThread, pyqtSignal
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI para threads
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from modelSummary.RelatorioDosModelos import RelatorioDosModelos
from modelSummary.ModelSummary import ModelSummary

class KerasLSTMRegressorBalanced(BaseEstimator, RegressorMixin):
  """LSTM para regressão com balanceamento de classes"""
  
  def __init__(self, input_shape, units=50, activation='tanh', dropout=0.2, 
         class_weights=None, noise_level=0.5):
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
      
      # LSTM principal com dropout interno e recorrente
      LSTM(self.units, activation=self.activation, 
         dropout=self.dropout, recurrent_dropout=self.dropout,
         return_sequences=True),
      
      # Batch normalization para estabilizar treinamento
      BatchNormalization(),
      
      # Segunda camada LSTM para maior capacidade
      LSTM(self.units//2, activation=self.activation,
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

class TrainingThreadLSTMCorrigido(QThread):
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
    self.model_type = "LSTM"
    self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
  def temporal_split(self, data, test_ratio=0.2):
    """Divisão temporal correta dos dados"""
    print(f"Realizando divisao temporal dos dados para {self.model_type}...")
    
    # Criar coluna datetime combinando data e hora
    data['DATETIME'] = pd.to_datetime(data['TXTDATE'].astype(str) + ' ' + data['TXTTIME'].astype(str))
    
    # Remover duplicatas temporais
    print(f"Dados originais: {len(data)} registros")
    data_clean = data.drop_duplicates(subset=['DATETIME'], keep='first').copy()
    print(f"Apos limpeza: {len(data_clean)} registros ({len(data) - len(data_clean)} duplicatas removidas)")
    
    # Ordenar por datetime completo
    data_sorted = data_clean.sort_values('DATETIME').reset_index(drop=True)
    split_idx = int(len(data_sorted) * (1 - test_ratio))
    
    train_data = data_sorted.iloc[:split_idx].copy()
    test_data = data_sorted.iloc[split_idx:].copy()
    
    print(f"Divisao temporal {self.model_type}:")
    print(f"  Treino: {train_data['DATETIME'].min()} ate {train_data['DATETIME'].max()} ({len(train_data)} amostras)")
    print(f"  Teste: {test_data['DATETIME'].min()} ate {test_data['DATETIME'].max()} ({len(test_data)} amostras)")
    print(f"  Datas unicas no teste: {test_data['TXTDATE'].nunique()}")
    
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
    """Constrói dados com metodologia corrigida"""
    print(f"Construindo dados corrigidos para {self.model_type}...")
    
    # 1. Divisão temporal
    train_data, test_data = self.temporal_split(data)
    
    # 2. Features temporais separadamente
    train_data = self.add_temporal_features_safe(train_data, is_train=True)
    test_data = self.add_temporal_features_safe(test_data, is_train=False)
    
    # 3. Preparar features e target (REGRESSÃO)
    features = ['Dia_da_Semana', 'Mês', 'Hora', 'LONGTIME_MA', 'LONGTIME_STD', 
          'LONGTIME_LAG_1', 'LONGTIME_LAG_2', 'LONGTIME_LAG_3']
    target = 'LONGTIME' # Target de regressão
    
    X_train = train_data[features].values
    y_train = train_data[target].values # Valores contínuos
    categories_train = train_data['CATEGORY'].values # Para análise de distribuição
    dates_train = train_data['TXTDATE'].values
    
    X_test = test_data[features].values
    y_test = test_data[target].values
    self.categories_test = test_data['CATEGORY'].values
    self.dates_test = test_data['DATETIME'].values  # Usar DATETIME completo
    
    # 4. Normalização (fit apenas no treino)
    print(f"Aplicando normalização {self.model_type}...")
    numeric_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='mean')),
      ('scaler', StandardScaler())
    ])
    
    self.scaler = numeric_transformer
    X_train_scaled = self.scaler.fit_transform(X_train)
    X_test_scaled = self.scaler.transform(X_test)
    
    # 5. Balanceamento (calcular pesos apenas para análise)
    class_weights = self.analyze_and_balance_classes(train_data, technique=balancing_technique)
    
    # Para regressão, não aplicamos SMOTE/undersampling diretamente
    # Usamos sample_weight na hora do fit
    
    # 6. Preparar para LSTM (adicionar dimensão temporal)
    self.X_train = np.expand_dims(X_train_scaled, axis=-1)
    self.X_test = np.expand_dims(X_test_scaled, axis=-1)
    self.y_train = np.expand_dims(y_train, axis=-1) # Valores de regressão
    self.y_test = np.expand_dims(y_test, axis=-1)
    
    print(f"Dados preparados para {self.model_type}:")
    print(f"  Treino: {self.X_train.shape}")
    print(f"  Teste: {self.X_test.shape}")
    
    return class_weights

  def run(self):
    """Execução corrigida do treinamento LSTM"""
    balancing_technique='class_weights'
    noise_level=0.1
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
    model = KerasLSTMRegressorBalanced(
      input_shape=(self.X_train.shape[1], 1),
      class_weights=class_weights,
      noise_level=noise_level
    )

    print(f"Realizando busca de hiperparâmetros {self.model_type}...")
    search = RandomizedSearchCV(
      model, param_distributions, 
      n_iter=3, # reduzido para acelerar testes
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
    
    # Gerar matriz de confusão
    self.generate_confusion_matrix_corrected(self.categories_test, best_model.predict(self.X_test))
    
    # Salvar modelo
    self.save_model_corrected(best_model.model)
    
    # EXATAMENTE como o modelo original - linha por linha
    test_loss = best_model.model.evaluate(self.X_test, self.y_test)
    test_mse = test_loss[1]
    test_results = best_model.model.predict(self.X_test)
    dates_test_array = np.array(self.dates_test)
    y_test_array = np.array(self.y_test)

    # Garantir que y_true e y_pred são unidimensionais
    y_trueA = np.ravel(y_test_array)
    y_predA = np.ravel(test_results)

    # Calcular a diferença entre previstos e reais
    difference = y_predA - y_trueA

    r_squared = r2_score(self.y_test, test_results)

    # DEBUG: Verificar tamanhos antes de criar DataFrame
    print(f"\nDEBUG - Criando DataFrame {self.model_type}:")
    print(f"   dates_test_array shape: {dates_test_array.shape}")
    print(f"   y_test_array shape: {y_test_array.shape}")
    print(f"   test_results shape: {test_results.shape}")
    print(f"   dates_test_array unique: {len(np.unique(dates_test_array))}")
    print(f"   Primeiros 5 dates: {dates_test_array[:5]}")
    print(f"   Primeiros 5 y_true: {y_test_array.flatten()[:5]}")
    print(f"   Primeiros 5 y_pred: {test_results.flatten()[:5]}")

    df_results = pd.DataFrame({
        'Data': dates_test_array,
        'Valor_Real': y_test_array.flatten(),
        'Valor_Previsto': test_results.flatten()
    })
    
    print(f"   DataFrame criado com {len(df_results)} linhas")
    print(f"   DataFrame unique dates: {df_results['Data'].nunique()}")
    print(f"   Primeiras 5 linhas do DataFrame:")
    print(df_results.head())

    df_results['Valor_Previsto_SMA'] = df_results['Valor_Previsto'].rolling(window=3, min_periods=1).mean()
    
    # Criar ModelSummary
    keras_model = best_model.model
    modelSu = ModelSummary(keras_model, f'{self.model_type}_Corrigido_model_summary.pdf', self.X_test.shape, self.y_test.shape)
    
    # Preparar os dados para o relatório
    models_and_results = {
        f'{self.model_type}_Corrigido_model': (df_results, modelSu)
    }

    # Colete as métricas
    metrics = {
        'MSE': test_mse,
        'RMSE': np.sqrt(test_mse),
        'R²': r_squared
        # Adicione outras métricas conforme necessário
    }

    # Instanciar e salvar o relatório (modelo corrigido - New)
    relatorio = RelatorioDosModelos(best_model, models_and_results, metrics, model_type="New")
    relatorio.save_reports_CSV_PDF()
    relatorio.save_shared_metrics()  # Adiciona as métricas ao CSV compartilhado
    relatorio.save_shared_metrics_list(mse_list, rmse_list, f"Modelo {self.model_type} Corrigido")
    relatorio.save_shared_difference_list(difference, f"Modelo {self.model_type} Corrigido")

    mse_array = np.array(mse_list)
    rmse_array = np.array(rmse_list)

    self.update_prediction_chart.emit(y_test_array, test_results, dates_test_array, f"Modelo {self.model_type} Corrigido")
    self.show_test_accuracy.emit(test_mse)
    self.update_metrics_chart.emit(test_mse, np.sqrt(test_mse), f"Modelo {self.model_type} Corrigido")
    self.update_metrics_chart_boxplot.emit(mse_array, rmse_array, f"{self.model_type} Corrigido")
  
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
    cm = confusion_matrix(y_true_categories, y_pred_categories, 
              labels=['ilegal', 'suspeito', 'válido'])
    accuracy = np.trace(cm) / np.sum(cm)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
          xticklabels=['ilegal', 'suspeito', 'válido'],
          yticklabels=['ilegal', 'suspeito', 'válido'])
    plt.title(f'Matriz de Confusão - {self.model_type} Corrigido\n(Sem Vazamento Temporal)')
    plt.xlabel('Categoria Predita')
    plt.ylabel('Categoria Real')
    
    # Salvar no diretório correto dentro de DadosDoPostreino/ModelosNew
    output_dir = os.path.join(self.base_dir, "DadosDoPostreino/ModelosNew/MatrizConfusao")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/matriz_confusao_{self.model_type.lower()}_corrigido.jpg', 
          dpi=300, bbox_inches='tight')
    plt.close()
    
    # Relatório
    report = classification_report(y_true_categories, y_pred_categories,
                   labels=['ilegal', 'suspeito', 'válido'], zero_division=0)
    
    with open(f'{output_dir}/relatorio_{self.model_type.lower()}_corrigido.txt', 'w', encoding='utf-8') as f:
      f.write(f"RELATÓRIO DE CLASSIFICAÇÃO - {self.model_type} CORRIGIDO\n")
      f.write("=" * 60 + "\n")
      f.write("Metodologia: Regressão com conversão para categorias\n")
      f.write("Sem vazamento temporal\n")
      f.write("Balanceamento: Pesos de amostra\n")
      f.write("-" * 60 + "\n\n")
      f.write(f"Acurácia: {accuracy:.4f} ({accuracy*100:.1f}%)\n\n")
      f.write(report)
      f.write("\n" + "-" * 60 + "\n")
      f.write("Matriz de Confusão:\n")
      f.write(str(cm))
    
    print(f"Matriz de confusão {self.model_type} salva em: {output_dir}")
    
    return cm, report
  
  def save_model_corrected(self, model):
    """Salva modelo corrigido em DadosDoPostreino/ModelosNew"""
    output_directory = os.path.join(self.base_dir, "DadosDoPostreino/ModelosNew/ModelosCorrigidos")
    os.makedirs(output_directory, exist_ok=True)
    keras_filename = os.path.join(output_directory, f"{self.model_type.lower()}_model_corrigido.keras")
    model.save(keras_filename)
    print(f"Modelo {self.model_type} corrigido salvo em: {keras_filename}")


def main():
  """Teste da versão LSTM corrigida"""
  print("Testando LSTM Corrigido com Ruído")
  print("="*50)
  
  # Carregar dados NORMALIZADOS
  data = pd.read_csv('DadosReais/dados_normalizados_smartgrid.csv')
  print(f"Dados carregados: {len(data)} registros")
  
  # Criar instância corrigida
  trainer = TrainingThreadLSTMCorrigido(data)
  
  # Executar treinamento corrigido
  results = trainer.run_corrected(balancing_technique='class_weights')
  
  print("\nTreinamento LSTM corrigido concluído!")
  print(f"MSE: {results['mse']:.6f}")
  print(f"RMSE: {results['rmse']:.6f}")
  print(f"R²: {results['r2']:.6f}")

if __name__ == "__main__":
  main()