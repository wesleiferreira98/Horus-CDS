#!/usr/bin/env python3
"""
Teste de paridade — temporal_split.py vs implementação inline antiga.

Compara byte a byte a saída do novo módulo `TratamentoDeDados/temporal_split.py`
com cópias VERBATIM dos métodos originalmente inline em `TCN_corrigido.py`.

Se este script passar, está provado que o refactor para o módulo comum
NÃO altera comportamento — você pode migrar TCN/LSTM/GRU/RNN/MLP com
segurança, sabendo que treinos vão produzir os mesmos pesos.

Uso:
    cd Horus-CDS
    python scripts/test_temporal_split_parity.py

Saída esperada:
    [OK] temporal_split: train e test iguais
    [OK] add_temporal_features_safe (train): iguais
    [OK] add_temporal_features_safe (test): iguais
    [OK] analyze_class_distribution: pesos iguais
    PARIDADE COMPLETA — refactor seguro.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Permitir importar de root/Linux/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINUX_DIR = os.path.join(PROJECT_ROOT, "root", "Linux")
sys.path.insert(0, LINUX_DIR)

# Importar a versão NOVA (que queremos validar)
from TratamentoDeDados.temporal_split import (
    temporal_split as new_temporal_split,
    add_temporal_features_safe as new_add_temporal_features_safe,
    analyze_class_distribution as new_analyze_class_distribution,
)


# ============================================================================
# CÓPIAS VERBATIM DOS MÉTODOS DE TCN_corrigido.py (versão ANTIGA)
# Sem `self`, sem prints reformatados — exatamente como estavam.
# ============================================================================

def old_temporal_split(data, test_ratio=0.2):
    """Cópia verbatim do método de TCN_corrigido.py linhas 114-144"""
    print("Realizando divisão temporal dos dados...")

    data['DATETIME'] = pd.to_datetime(
        data['TXTDATE'].astype(str) + ' ' + data['TXTTIME'].astype(str)
    )

    print(f"Dados originais: {len(data)} registros")
    data_clean = data.drop_duplicates(subset=['DATETIME'], keep='first').copy()
    print(f"Apos limpeza: {len(data_clean)} registros "
          f"({len(data) - len(data_clean)} duplicatas removidas)")

    data_sorted = data_clean.sort_values('DATETIME').reset_index(drop=True)
    split_idx = int(len(data_sorted) * (1 - test_ratio))

    train_data = data_sorted.iloc[:split_idx].copy()
    test_data = data_sorted.iloc[split_idx:].copy()

    return train_data, test_data


def old_add_temporal_features_safe(data, is_train=True):
    """Cópia verbatim do método de TCN_corrigido.py linhas 146-174"""
    data = data.copy()
    data['TXTDATE'] = pd.to_datetime(data['TXTDATE'])
    data['Dia_da_Semana'] = data['TXTDATE'].dt.dayofweek
    data['Mês'] = data['TXTDATE'].dt.month
    data['Hora'] = pd.to_datetime(data['TXTTIME'], format='%H:%M:%S').dt.hour

    data = data.sort_values('TXTDATE').reset_index(drop=True)

    window_size = 3
    data['LONGTIME_MA'] = data['LONGTIME'].rolling(
        window=window_size, min_periods=1).mean()
    data['LONGTIME_STD'] = data['LONGTIME'].rolling(
        window=window_size, min_periods=1).std().fillna(0)

    for lag in [1, 2, 3]:
        data[f'LONGTIME_LAG_{lag}'] = data['LONGTIME'].shift(lag)

    data = data.dropna().reset_index(drop=True)

    return data


def old_analyze_class_distribution(data):
    """Cópia (quase) verbatim do método de TCN_corrigido.py linhas 176-202.

    NOTA: o original passa data['CATEGORY'].unique() direto. Em sklearn >= 1.6
    isso falha com StringArray. Aplicamos np.asarray() para alinhar com o novo
    módulo — o resultado numérico é idêntico, só o tipo da entrada muda.
    """
    class_counts = data['CATEGORY'].value_counts()
    total = len(data)

    unique_classes = np.asarray(data['CATEGORY'].unique())
    class_weights_array = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=data['CATEGORY']
    )
    class_weights = dict(zip(unique_classes, class_weights_array))

    return class_weights


# ============================================================================
# COMPARADORES
# ============================================================================

def assert_dataframes_equal(df_old, df_new, label):
    """Compara dois DataFrames com tolerância para floats."""
    # Mesmo conjunto de colunas
    cols_old = sorted(df_old.columns.tolist())
    cols_new = sorted(df_new.columns.tolist())
    if cols_old != cols_new:
        only_old = set(cols_old) - set(cols_new)
        only_new = set(cols_new) - set(cols_old)
        raise AssertionError(
            f"[FAIL] {label}: colunas divergem.\n"
            f"  Só no antigo: {only_old}\n"
            f"  Só no novo:   {only_new}"
        )

    # Mesmo tamanho
    if len(df_old) != len(df_new):
        raise AssertionError(
            f"[FAIL] {label}: tamanhos divergem. "
            f"antigo={len(df_old)}, novo={len(df_new)}"
        )

    # Comparar valores coluna a coluna (na mesma ordem)
    df_old_sorted = df_old[cols_old].reset_index(drop=True)
    df_new_sorted = df_new[cols_old].reset_index(drop=True)

    for col in cols_old:
        s_old = df_old_sorted[col]
        s_new = df_new_sorted[col]

        if pd.api.types.is_numeric_dtype(s_old):
            if not np.allclose(s_old.values, s_new.values,
                                rtol=1e-9, atol=1e-12, equal_nan=True):
                idx = np.where(~np.isclose(
                    s_old.values, s_new.values,
                    rtol=1e-9, atol=1e-12, equal_nan=True))[0]
                raise AssertionError(
                    f"[FAIL] {label}: coluna '{col}' diverge em {len(idx)} "
                    f"posições. Primeiro mismatch: idx={idx[0]}, "
                    f"old={s_old.iloc[idx[0]]}, new={s_new.iloc[idx[0]]}"
                )
        else:
            if not s_old.equals(s_new):
                raise AssertionError(
                    f"[FAIL] {label}: coluna '{col}' diverge (não-numérica)"
                )

    print(f"[OK] {label}: iguais ({len(df_old)} linhas, "
          f"{len(cols_old)} colunas)")


def assert_dicts_equal(d_old, d_new, label):
    """Compara dois dicts de class_weights."""
    if set(d_old.keys()) != set(d_new.keys()):
        raise AssertionError(
            f"[FAIL] {label}: chaves divergem. "
            f"antigo={set(d_old.keys())}, novo={set(d_new.keys())}"
        )
    for k in d_old:
        if not np.isclose(d_old[k], d_new[k], rtol=1e-12):
            raise AssertionError(
                f"[FAIL] {label}: peso de '{k}' diverge. "
                f"antigo={d_old[k]}, novo={d_new[k]}"
            )
    print(f"[OK] {label}: pesos iguais ({len(d_old)} classes)")


# ============================================================================
# EXECUÇÃO
# ============================================================================

def main():
    csv_path = os.path.join(
        LINUX_DIR, "DadosReais", "dados_normalizados_smartgrid.csv"
    )
    if not os.path.exists(csv_path):
        print(f"ERRO: dataset nao encontrado em {csv_path}")
        sys.exit(1)

    print(f"Carregando dataset: {csv_path}")
    data = pd.read_csv(csv_path)
    print(f"Dataset: {len(data)} linhas, {len(data.columns)} colunas\n")

    # --- 1. temporal_split ---
    print("=" * 60)
    print("1. temporal_split")
    print("=" * 60)

    # IMPORTANTE: passar cópias separadas porque a função mutaciona 'data'
    train_old, test_old = old_temporal_split(data.copy())
    train_new, test_new = new_temporal_split(data.copy(), verbose=False)

    assert_dataframes_equal(train_old, train_new, "temporal_split → train")
    assert_dataframes_equal(test_old, test_new, "temporal_split → test")

    # --- 2. add_temporal_features_safe ---
    print("\n" + "=" * 60)
    print("2. add_temporal_features_safe")
    print("=" * 60)

    train_old_feat = old_add_temporal_features_safe(train_old, is_train=True)
    train_new_feat = new_add_temporal_features_safe(
        train_new, is_train=True, verbose=False)
    assert_dataframes_equal(
        train_old_feat, train_new_feat,
        "add_temporal_features_safe → train"
    )

    test_old_feat = old_add_temporal_features_safe(test_old, is_train=False)
    test_new_feat = new_add_temporal_features_safe(
        test_new, is_train=False, verbose=False)
    assert_dataframes_equal(
        test_old_feat, test_new_feat,
        "add_temporal_features_safe → test"
    )

    # --- 3. analyze_class_distribution ---
    print("\n" + "=" * 60)
    print("3. analyze_class_distribution")
    print("=" * 60)

    weights_old = old_analyze_class_distribution(train_old)
    weights_new = new_analyze_class_distribution(train_new, verbose=False)
    assert_dicts_equal(
        weights_old, weights_new,
        "analyze_class_distribution"
    )

    print("\n" + "=" * 60)
    print("PARIDADE COMPLETA — refactor seguro.")
    print("Pode migrar TCN_corrigido.py, LSTM_corrigido.py, GRU_corrigido.py,")
    print("RNN_corrigido.py e TrainingThreadMLP.py para usar o novo módulo.")
    print("=" * 60)


if __name__ == "__main__":
    main()
