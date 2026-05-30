#!/usr/bin/env python3
"""
Utilitários compartilhados de divisão temporal e feature engineering
para os modelos do Horus-CDS (TCN, LSTM, GRU, RNN, MLP, Transformer V5).

Extraído dos arquivos *_corrigido.py em ThreadTrain/ para eliminar
duplicação e garantir que todos os modelos usem exatamente o mesmo
pré-processamento (apples-to-apples).

Todas as funções aqui são PURAS (sem self, sem QThread) — podem ser
chamadas tanto de dentro das threads PyQt5 quanto de scripts/notebooks.

Dataset esperado: dados_normalizados_smartgrid.csv
Colunas obrigatórias: TXTDATE, TXTTIME, LONGTIME, CATEGORY
"""

import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def temporal_split(data, test_ratio=0.2, model_name="", verbose=True):
    """
    Realiza divisão temporal dos dados respeitando a ordem cronológica.

    Importante: NÃO usa train_test_split aleatório. A divisão é feita
    pela ordem do timestamp completo (TXTDATE + TXTTIME) para evitar
    vazamento temporal (dados do futuro contaminando o treino).

    Etapas:
      1. Combina TXTDATE + TXTTIME em uma coluna DATETIME
      2. Remove duplicatas temporais (mantém primeira ocorrência)
      3. Ordena por DATETIME
      4. Divide os primeiros (1 - test_ratio) para treino, resto para teste

    Args:
        data (pd.DataFrame): DataFrame com colunas TXTDATE e TXTTIME.
        test_ratio (float): Proporção para teste (default 0.2 = 80/20).
        model_name (str): Nome do modelo, usado apenas em logs.
        verbose (bool): Se True, imprime estatísticas da divisão.

    Returns:
        tuple: (train_data, test_data) — DataFrames já com a coluna
               DATETIME adicionada e ordenados cronologicamente.
    """
    if verbose:
        suffix = f" para {model_name}" if model_name else ""
        print(f"Realizando divisao temporal dos dados{suffix}...")

    # 1. Combinar data e hora em um único timestamp
    data = data.copy()
    data['DATETIME'] = pd.to_datetime(
        data['TXTDATE'].astype(str) + ' ' + data['TXTTIME'].astype(str)
    )

    # 2. Remover duplicatas temporais
    if verbose:
        print(f"Dados originais: {len(data)} registros")
    data_clean = data.drop_duplicates(subset=['DATETIME'], keep='first').copy()
    if verbose:
        removidas = len(data) - len(data_clean)
        print(f"Apos limpeza: {len(data_clean)} registros "
              f"({removidas} duplicatas removidas)")

    # 3. Ordenar por DATETIME completo (data + hora)
    data_sorted = data_clean.sort_values('DATETIME').reset_index(drop=True)

    # 4. Dividir treino / teste mantendo ordem temporal
    split_idx = int(len(data_sorted) * (1 - test_ratio))
    train_data = data_sorted.iloc[:split_idx].copy()
    test_data = data_sorted.iloc[split_idx:].copy()

    if verbose:
        prefix = f"Divisao temporal {model_name}:" if model_name else "Divisao temporal:"
        print(f"\n{prefix}")
        print(f"  Treino: {train_data['DATETIME'].min()} ate "
              f"{train_data['DATETIME'].max()} ({len(train_data)} amostras)")
        print(f"  Teste : {test_data['DATETIME'].min()} ate "
              f"{test_data['DATETIME'].max()} ({len(test_data)} amostras)")
        print(f"  DATETIME unicos no teste: {test_data['DATETIME'].nunique()}")

    return train_data, test_data


def add_temporal_features_safe(data, is_train=True, model_name="", verbose=True,
                                window_size=3, lags=(1, 2, 3)):
    """
    Adiciona features temporais derivadas SEM gerar vazamento entre
    treino e teste.

    Regra de ouro: rolling/lag são calculados APÓS a divisão temporal,
    separadamente em cada conjunto. Nunca antes — senão valores de
    teste vazam para features do treino.

    Features adicionadas:
      - Dia_da_Semana   (0=segunda ... 6=domingo)
      - Mês             (1-12)
      - Hora            (0-23)
      - LONGTIME_MA     (média móvel de janela window_size)
      - LONGTIME_STD    (desvio padrão móvel)
      - LONGTIME_LAG_n  (n em lags)

    Linhas com NaN nos lags (as primeiras max(lags)) são removidas.

    Args:
        data (pd.DataFrame): Conjunto já dividido (treino OU teste).
        is_train (bool): Apenas afeta a mensagem de log.
        model_name (str): Nome do modelo, usado em logs.
        verbose (bool): Se True, imprime estatísticas.
        window_size (int): Janela para média/desvio móveis (default 3).
        lags (tuple): Lags a calcular (default (1, 2, 3)).

    Returns:
        pd.DataFrame: Dados com as novas colunas, ordenados por TXTDATE
                      e sem linhas NaN dos lags.
    """
    data = data.copy()
    data['TXTDATE'] = pd.to_datetime(data['TXTDATE'])
    data['Dia_da_Semana'] = data['TXTDATE'].dt.dayofweek
    data['Mês'] = data['TXTDATE'].dt.month
    data['Hora'] = pd.to_datetime(data['TXTTIME'], format='%H:%M:%S').dt.hour

    # Ordenar por data para garantir ordem temporal antes de rolling/lag
    data = data.sort_values('TXTDATE').reset_index(drop=True)

    # Moving average e std — calculados apenas dentro deste conjunto
    data['LONGTIME_MA'] = data['LONGTIME'].rolling(
        window=window_size, min_periods=1).mean()
    data['LONGTIME_STD'] = data['LONGTIME'].rolling(
        window=window_size, min_periods=1).std().fillna(0)

    # Lag features — calculados apenas dentro deste conjunto
    for lag in lags:
        data[f'LONGTIME_LAG_{lag}'] = data['LONGTIME'].shift(lag)

    # Remover linhas com NaN resultantes dos lags
    data = data.dropna().reset_index(drop=True)

    if verbose:
        conjunto = "Treino" if is_train else "Teste"
        suffix = f" {model_name}" if model_name else ""
        print(f"Features temporais adicionadas ao {conjunto}{suffix}: "
              f"{len(data)} amostras restantes")

    return data


def analyze_class_distribution(data, target_col='CATEGORY', verbose=True,
                                model_name=""):
    """
    Calcula a distribuição das classes e os pesos balanceados para uso
    em modelos sensíveis a classe.

    Importante: deve ser chamada APENAS sobre o conjunto de TREINO.
    Calcular pesos sobre o conjunto completo gera vazamento.

    Args:
        data (pd.DataFrame): Conjunto de treino contendo target_col.
        target_col (str): Nome da coluna de rótulos (default 'CATEGORY').
        verbose (bool): Se True, imprime distribuição e pesos.
        model_name (str): Nome do modelo, usado em logs.

    Returns:
        dict: Mapeamento {classe: peso} usando estratégia 'balanced'
              do sklearn (n_samples / (n_classes * count_classe)).
    """
    if verbose:
        suffix = f" para {model_name}" if model_name else ""
        print(f"Analisando distribuicao das classes{suffix}...")

    class_counts = data[target_col].value_counts()
    total = len(data)

    if verbose:
        print("Distribuicao original:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} ({count / total * 100:.1f}%)")

    # np.asarray garante numpy.ndarray (sklearn >= 1.6 rejeita StringArray
    # do pandas em 'classes'). Comportamento idêntico ao código antigo em
    # sklearn antigo; corrige incompatibilidade em sklearn novo.
    unique_classes = np.asarray(data[target_col].unique())
    class_weights_array = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=data[target_col]
    )
    class_weights = dict(zip(unique_classes, class_weights_array))

    if verbose:
        print("Pesos calculados para balanceamento:")
        for cls, weight in class_weights.items():
            print(f"  {cls}: {weight:.3f}")

    return class_weights


# Lista das features padrão usadas pelo pipeline atual (TCN/LSTM/GRU/RNN/MLP).
# Mantida aqui para que todos os modelos referenciem a mesma constante.
DEFAULT_FEATURES = [
    'Dia_da_Semana', 'Mês', 'Hora',
    'LONGTIME_MA', 'LONGTIME_STD',
    'LONGTIME_LAG_1', 'LONGTIME_LAG_2', 'LONGTIME_LAG_3',
]

DEFAULT_TARGET = 'LONGTIME'
