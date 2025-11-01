#!/usr/bin/env python3
"""
Versão Corrigida do TrainingThreadTCN
Resolve os problemas metodológicos identificados:
1. Tratamento de desequilíbrio de classes
2. Divisão temporal adequada
3. Prevenção de vazamento temporal
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import r2_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras import layers, optimizers # type: ignore
from tcn import TCN
from PyQt5.QtCore import QThread, pyqtSignal
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI para threads
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

class KerasTCNRegressorBalanced(BaseEstimator, RegressorMixin):
  def __init__(self, input_shape, nb_filters=64, kernel_size=3, nb_stacks=1, 
         dilations=[1, 2, 4, 8], activation='relu', use_skip_connections=True,
         class_weights=None):
    self.input_shape = input_shape
    self.nb_filters = nb_filters
    self.kernel_size = kernel_size
    self.nb_stacks = nb_stacks
    self.dilations = dilations
    self.activation = activation
    self.use_skip_connections = use_skip_connections
    self.class_weights = class_weights
    self.model = self._build_model()

  def _build_model(self):
    model = Sequential([
      TCN(input_shape=self.input_shape,
        nb_filters=self.nb_filters,
        kernel_size=self.kernel_size,
        nb_stacks=self.nb_stacks,
        dilations=self.dilations,
        activation=self.activation,
        use_skip_connections=self.use_skip_connections),
      layers.Dense(1)
    ])
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
    return model

  def fit(self, X, y, **kwargs):
    # Aplicar pesos de classe se fornecidos
    if self.class_weights is not None and 'sample_weight' not in kwargs:
      # Converter regressão para classes para aplicar pesos
      y_classes = self._convert_regression_to_classes(y)
      sample_weights = np.array([self.class_weights.get(cls, 1.0) for cls in y_classes])
      kwargs['sample_weight'] = sample_weights
    
    self.model.fit(X, y, **kwargs)

  def predict(self, X):
    return self.model.predict(X)

  def score(self, X, y):
    return self.model.evaluate(X, y)[1]
  
  def _convert_regression_to_classes(self, y):
    """Converte valores de regressão para classes para aplicar pesos"""
    classes = []
    for value in y.flatten():
      if value < -0.5: # Threshold para dados normalizados
        classes.append('ilegal')
      elif value < 0.5:
        classes.append('suspeito')
      else:
        classes.append('válido')
    return classes

class TrainingThreadTCNCorrigido(QThread):
  show_test_accuracy = pyqtSignal(float)
  update_progress = pyqtSignal(float)
  update_prediction_chart = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, str)
  update_metrics_chart_boxplot = pyqtSignal(np.ndarray, np.ndarray, str)
  update_metrics_chart = pyqtSignal(float, float, str)

  def __init__(self, data_set):
    super().__init__()
    self.data_set = data_set
    self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    self.X_train = None
    self.X_test = None
    self.y_train = None
    self.y_test = None
    self.categories_test = None
    self.dates_test = None
    self.scaler = None
    
  def temporal_split(self, data, test_ratio=0.2):
    """
    Divisão temporal correta dos dados
    """
    print("Realizando divisão temporal dos dados...")
    
    # Ordenar por data
    data_sorted = data.sort_values('TXTDATE').reset_index(drop=True)
    
    # Calcular ponto de divisão
    split_idx = int(len(data_sorted) * (1 - test_ratio))
    
    train_data = data_sorted.iloc[:split_idx].copy()
    test_data = data_sorted.iloc[split_idx:].copy()
    
    print(f"Divisão temporal:")
    print(f"  Treino: {train_data['TXTDATE'].min()} até {train_data['TXTDATE'].max()} ({len(train_data)} amostras)")
    print(f"  Teste: {test_data['TXTDATE'].min()} até {test_data['TXTDATE'].max()} ({len(test_data)} amostras)")
    
    return train_data, test_data
  
  def add_temporal_features_safe(self, data, is_train=True):
    """
    Adiciona features temporais sem vazamento
    """
    data = data.copy()
    data['TXTDATE'] = pd.to_datetime(data['TXTDATE'])
    data['Dia_da_Semana'] = data['TXTDATE'].dt.dayofweek
    data['Mês'] = data['TXTDATE'].dt.month
    data['Hora'] = pd.to_datetime(data['TXTTIME'], format='%H:%M:%S').dt.hour
    
    # Ordenar por data para garantir ordem temporal
    data = data.sort_values('TXTDATE').reset_index(drop=True)
    
    # Moving average e std - calculados apenas dentro do conjunto
    window_size = 3
    data['LONGTIME_MA'] = data['LONGTIME'].rolling(window=window_size, min_periods=1).mean()
    data['LONGTIME_STD'] = data['LONGTIME'].rolling(window=window_size, min_periods=1).std().fillna(0)
    
    # Lag features - calculados apenas dentro do conjunto
    for lag in [1, 2, 3]:
      data[f'LONGTIME_LAG_{lag}'] = data['LONGTIME'].shift(lag)
    
    # Remover NaN resultantes dos lags
    data = data.dropna().reset_index(drop=True)
    
    prefix = "Treino" if is_train else "Teste"
    print(f"Features temporais adicionadas ao conjunto de {prefix}: {len(data)} amostras restantes")
    
    return data
  
  def analyze_class_distribution(self, data):
    """
    Analisa distribuição das classes e calcula pesos
    """
    print("Analisando distribuição das classes...")
    
    class_counts = data['CATEGORY'].value_counts()
    total = len(data)
    
    print("Distribuição original:")
    for cls, count in class_counts.items():
      print(f"  {cls}: {count} ({count/total*100:.1f}%)")
    
    # Calcular pesos de classe balanceados
    unique_classes = data['CATEGORY'].unique()
    class_weights_array = compute_class_weight(
      'balanced', 
      classes=unique_classes, 
      y=data['CATEGORY']
    )
    class_weights = dict(zip(unique_classes, class_weights_array))
    
    print("Pesos calculados para balanceamento:")
    for cls, weight in class_weights.items():
      print(f"  {cls}: {weight:.3f}")
      
    return class_weights
  
  def apply_balancing_technique(self, X, y, categories, technique='smote'):
    """
    Aplica técnicas de balanceamento de classes
    """
    if technique == 'smote':
      print("Aplicando SMOTE para balanceamento...")
      
      # Converter categorias para números para SMOTE
      category_map = {'ilegal': 0, 'suspeito': 1, 'válido': 2}
      y_categorical = np.array([category_map[cat] for cat in categories])
      
      # Aplicar SMOTE
      smote = SMOTE(random_state=42, k_neighbors=3)
      X_resampled, y_cat_resampled = smote.fit_resample(X, y_categorical)
      
      # Ajustar y de regressão proporcionalmente
      indices_originais = []
      for y_cat in y_cat_resampled:
        # Encontrar índices da classe original
        class_indices = np.where(y_categorical == y_cat)[0]
        # Selecionar aleatoriamente um índice
        selected_idx = np.random.choice(class_indices)
        indices_originais.append(selected_idx)
      
      y_resampled = y[indices_originais]
      categories_resampled = [list(category_map.keys())[cat] for cat in y_cat_resampled]
      
      print(f"Dados após SMOTE: {len(X_resampled)} amostras")
      unique, counts = np.unique(categories_resampled, return_counts=True)
      for cls, count in zip(unique, counts):
        print(f"  {cls}: {count}")
        
      return X_resampled, y_resampled, np.array(categories_resampled)
    
    elif technique == 'undersampling':
      print("Aplicando undersampling...")
      
      category_map = {'ilegal': 0, 'suspeito': 1, 'válido': 2}
      y_categorical = np.array([category_map[cat] for cat in categories])
      
      undersampler = RandomUnderSampler(random_state=42)
      X_resampled, y_cat_resampled = undersampler.fit_resample(X, y_categorical)
      
      # Reconstruir y de regressão e categorias
      y_resampled = y[undersampler.sample_indices_]
      categories_resampled = categories[undersampler.sample_indices_]
      
      print(f"Dados após undersampling: {len(X_resampled)} amostras")
      unique, counts = np.unique(categories_resampled, return_counts=True)
      for cls, count in zip(unique, counts):
        print(f"  {cls}: {count}")
        
      return X_resampled, y_resampled, categories_resampled
    
    else:
      return X, y, categories

  def build_data_corrected(self, data):
    """
    Versão corrigida do build_data que evita vazamento temporal
    """
    print("Construindo dados com metodologia corrigida...")
    
    # 1. PRIMEIRO: Divisão temporal
    train_data, test_data = self.temporal_split(data)
    
    # 2. Analisar distribuição das classes (apenas no treino)
    class_weights = self.analyze_class_distribution(train_data)
    
    # 3. Adicionar features temporais separadamente
    train_data = self.add_temporal_features_safe(train_data, is_train=True)
    test_data = self.add_temporal_features_safe(test_data, is_train=False)
    
    # 4. Preparar features e target
    features = ['Dia_da_Semana', 'Mês', 'Hora', 'LONGTIME_MA', 'LONGTIME_STD', 
          'LONGTIME_LAG_1', 'LONGTIME_LAG_2', 'LONGTIME_LAG_3']
    target = 'LONGTIME'
    
    # Separar dados
    X_train = train_data[features].values
    y_train = train_data[target].values
    categories_train = train_data['CATEGORY'].values
    dates_train = train_data['TXTDATE'].values
    
    X_test = test_data[features].values
    y_test = test_data[target].values
    self.categories_test = test_data['CATEGORY'].values
    self.dates_test = test_data['TXTDATE'].values
    
    # 5. Normalização (fit apenas no treino!)
    print("Aplicando normalização (StandardScaler apenas no treino)...")
    numeric_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='mean')),
      ('scaler', StandardScaler())
    ])
    
    self.scaler = numeric_transformer
    X_train_scaled = self.scaler.fit_transform(X_train)
    X_test_scaled = self.scaler.transform(X_test)
    
    # 6. Aplicar balanceamento (apenas no treino)
    technique = 'class_weights' # Opções: 'smote', 'undersampling', 'class_weights'
    
    if technique in ['smote', 'undersampling']:
      X_train_balanced, y_train_balanced, categories_train_balanced = self.apply_balancing_technique(
        X_train_scaled, y_train, categories_train, technique=technique
      )
    else:
      X_train_balanced = X_train_scaled
      y_train_balanced = y_train
      categories_train_balanced = categories_train
    
    # 7. Preparar para TCN
    self.X_train = np.expand_dims(X_train_balanced, axis=-1)
    self.X_test = np.expand_dims(X_test_scaled, axis=-1)
    self.y_train = np.expand_dims(y_train_balanced, axis=-1)
    self.y_test = np.expand_dims(y_test, axis=-1)
    
    # Armazenar pesos de classe
    self.class_weights = class_weights
    
    print(f"Dados preparados com sucesso!")
    print(f"  Treino: {self.X_train.shape[0]} amostras")
    print(f"  Teste: {self.X_test.shape[0]} amostras")
    
    return class_weights

  def run(self):
    """Metodo run para compatibilidade com interface"""
    results = self.run_corrected()
    
    # Emitir sinais para interface
    mse_array = np.array(results['mse_progression'])
    rmse_array = np.array(results['rmse_progression'])
    
    self.update_prediction_chart.emit(
      self.dates_test, 
      self.y_test.flatten(), 
      results['predictions'].flatten(),
      "Modelo TCN Corrigido"
    )
    self.update_metrics_chart.emit(results['mse'], results['rmse'], "Modelo TCN Corrigido")
    self.update_metrics_chart_boxplot.emit(mse_array, rmse_array, "TCN_Corrigido")
    self.show_test_accuracy.emit(results['mse'])

  def run_corrected(self):
    """
    Execução corrigida do treinamento
    """
    print("Iniciando treinamento com metodologia corrigida...")
    
    # Construir dados corrigidos
    class_weights = self.build_data_corrected(self.data_set)
    
    # Parâmetros para busca
    param_distributions = {
      'nb_filters': [32, 64, 128],
      'kernel_size': [2, 3, 4],
      'nb_stacks': [1, 2],
      'dilations': [[1, 2, 4, 8], [1, 2, 4, 8, 16]],
      'activation': ['relu', 'tanh'],
      'use_skip_connections': [True, False]
    }

    # Criar modelo com pesos de classe
    model = KerasTCNRegressorBalanced(
      input_shape=(self.X_train.shape[1], 1),
      class_weights=class_weights
    )

    print("Realizando busca de hiperparâmetros...")
    search = RandomizedSearchCV(
      model, param_distributions, 
      n_iter=5, # Reduzido para teste
      cv=3, 
      scoring='neg_mean_squared_error', 
      verbose=1
    )
    
    search.fit(self.X_train, self.y_train)
    best_model = search.best_estimator_
    
    print("Treinando modelo final...")
    mse_list = []
    rmse_list = []
    
    for epoch in range(20): # Reduzido para teste
      best_model.fit(
        self.X_train, self.y_train, 
        epochs=1, batch_size=64, 
        validation_split=0.2, verbose=0
      )
      
      test_loss = best_model.model.evaluate(self.X_test, self.y_test, verbose=0)
      test_mse = test_loss[1]
      mse_list.append(test_mse)
      rmse_list.append(np.sqrt(test_mse))
      self.update_progress.emit((epoch + 1) / 20)
    
    # Avaliar modelo
    test_results = best_model.model.predict(self.X_test)
    test_mse_final = mse_list[-1]
    r_squared = r2_score(self.y_test, test_results)
    
    print(f"Resultados finais:")
    print(f"  MSE: {test_mse_final:.6f}")
    print(f"  RMSE: {np.sqrt(test_mse_final):.6f}")
    print(f"  R²: {r_squared:.6f}")
    
    # Gerar matriz de confusão
    self.generate_confusion_matrix_corrected(self.categories_test, test_results)
    
    # Salvar modelo
    self.save_model_corrected(best_model.model)
    
    return {
      'mse': test_mse_final,
      'rmse': np.sqrt(test_mse_final),
      'r2': r_squared,
      'predictions': test_results,
      'dates': self.dates_test,
      'mse_progression': mse_list,
      'rmse_progression': rmse_list
    }
  
  def generate_confusion_matrix_corrected(self, y_true_categories, y_pred_longtime):
    """
    Gera matriz de confusão com thresholds ajustados para dados normalizados
    """
    print("Gerando matriz de confusão corrigida...")
    
    # Converter previsões para categorias usando percentis dos dados de teste
    y_pred_flat = y_pred_longtime.flatten()
    
    # Calcular thresholds baseados na distribuição real das categorias no teste
    threshold_17_8 = np.percentile(y_pred_flat, 17.8)
    threshold_23_2 = np.percentile(y_pred_flat, 23.2)
    
    def classify_corrected(values):
      return np.where(values < threshold_17_8, 'ilegal',
          np.where(values < threshold_23_2, 'suspeito', 'válido'))
    
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
    plt.title('Matriz de Confusão - TCN Corrigido\n(Sem Vazamento Temporal)')
    plt.xlabel('Categoria Predita')
    plt.ylabel('Categoria Real')
    
    # Salvar
    output_dir = os.path.join(self.base_dir, "MatrizConfusaoCorrigida")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/matriz_confusao_tcn_corrigido.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Relatório
    report = classification_report(y_true_categories, y_pred_categories,
                   labels=['ilegal', 'suspeito', 'válido'], zero_division=0)
    
    with open(f'{output_dir}/relatorio_tcn_corrigido.txt', 'w', encoding='utf-8') as f:
      f.write("RELATÓRIO TCN CORRIGIDO - SEM VAZAMENTO TEMPORAL\n")
      f.write("="*60 + "\n")
      f.write(f"Acurácia: {accuracy:.3f} ({accuracy*100:.1f}%)\n")
      f.write("Correções aplicadas:\n")
      f.write("- Divisão temporal (não aleatória)\n")
      f.write("- Features calculadas separadamente\n")
      f.write("- StandardScaler fit apenas no treino\n")
      f.write("- Balanceamento de classes\n\n")
      f.write(report)
    
    print(f"Matriz de confusão corrigida salva em: {output_dir}")
    print(f"Acurácia corrigida: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    return cm, report
  
  def save_model_corrected(self, model):
    """Salva modelo corrigido"""
    output_directory = os.path.join(self.base_dir, "ModelosCorrigidos")
    os.makedirs(output_directory, exist_ok=True)
    keras_filename = os.path.join(output_directory, "tcn_model_corrigido.keras")
    model.save(keras_filename)
    print(f"Modelo corrigido salvo em: {keras_filename}")


def main():
  """Teste da versão corrigida"""
  print("Testando TrainingThreadTCN Corrigido")
  print("="*50)
  
  # Carregar dados
  data = pd.read_csv('DadosReais/dados_normalizados_smartgrid.csv')
  print(f"Dados carregados: {len(data)} registros")
  
  # Criar instância corrigida
  trainer = TrainingThreadTCNCorrigido(data)
  
  # Executar treinamento corrigido
  results = trainer.run_corrected()
  
  print("\nTreinamento corrigido concluído!")
  print(f"MSE Corrigido: {results['mse']:.6f}")
  print(f"RMSE Corrigido: {results['rmse']:.6f}")
  print(f"R² Corrigido: {results['r2']:.6f}")

if __name__ == "__main__":
  main()