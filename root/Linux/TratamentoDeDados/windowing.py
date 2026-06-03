#!/usr/bin/env python3
"""Geração de janelas deslizantes para modelos sequenciais (V5 / PatchTST).

Deve ser chamado após temporal_split — janelar antes da divisão vazaria dados.
"""

import numpy as np
import pandas as pd


# ── Constantes ────────────────────────────────────────────────────────────────

DEFAULT_SEQ_LEN             = 32
DEFAULT_TRANSFORMER_FEATURES = ['LONGTIME', 'INTACTIVE', 'INTMANUAL']

CATEGORY_TO_INT = {'ilegal': 0, 'suspeito': 1, 'válido': 2}
INT_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_INT.items()}


# ── Função principal ──────────────────────────────────────────────────────────

def make_windows(df, feature_cols=None, target_col='CATEGORY',
                 seq_len=DEFAULT_SEQ_LEN, stride=1, verbose=True,
                 model_name=""):
    """Converte DataFrame ordenado cronologicamente em janelas para modelo sequencial.

    Para cada posição i ∈ [seq_len, len(df)), gera:
        X[k] = df[feature_cols].iloc[i - seq_len : i]   # (seq_len, F)
        y[k] = df[target_col].iloc[i]

    Args:
        df (pd.DataFrame): conjunto já dividido temporalmente, ordenado por DATETIME.
        feature_cols (list[str] | None): colunas de entrada; usa
            DEFAULT_TRANSFORMER_FEATURES se None.
        target_col (str): coluna de rótulo. Strings são codificadas via
            CATEGORY_TO_INT; valores numéricos são usados diretamente.
        seq_len (int): tamanho da janela. Default: DEFAULT_SEQ_LEN.
        stride (int): passo entre janelas. Default: 1 (sobreposição máxima).
        verbose (bool): imprime estatísticas da geração. Default: True.
        model_name (str): prefixo para logs. Default: "".

    Returns:
        tuple[np.ndarray, np.ndarray]:
            X — shape (N, seq_len, num_features), dtype float32.
            y — shape (N,), dtype int64 (categórico) ou float32 (numérico).

    Raises:
        KeyError: se alguma coluna de feature_cols ou target_col não existir.
        ValueError: se len(df) <= seq_len ou se target_col tiver valor fora
            de CATEGORY_TO_INT.
    """
    if feature_cols is None:
        feature_cols = DEFAULT_TRANSFORMER_FEATURES

    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas ausentes: {missing}. Disponíveis: {list(df.columns)}")

    if len(df) <= seq_len:
        raise ValueError(
            f"DataFrame com {len(df)} linhas é menor ou igual ao seq_len={seq_len}."
        )

    features_arr = df[feature_cols].values.astype(np.float32)
    target_raw   = df[target_col].values

    if target_raw.dtype == object or pd.api.types.is_string_dtype(target_raw):
        try:
            y_full = np.array([CATEGORY_TO_INT[v] for v in target_raw], dtype=np.int64)
        except KeyError as e:
            raise ValueError(
                f"Valor de '{target_col}' fora do mapeamento {list(CATEGORY_TO_INT)}: {e}"
            )
    else:
        y_full = target_raw.astype(np.float32)

    indices      = np.arange(seq_len, len(df), stride)
    n_windows    = len(indices)
    num_features = len(feature_cols)

    X = np.empty((n_windows, seq_len, num_features), dtype=np.float32)
    for k, i in enumerate(indices):
        X[k] = features_arr[i - seq_len : i]
    y = y_full[indices]

    if verbose:
        suffix = f" ({model_name})" if model_name else ""
        print(f"Windowing{suffix}: {len(df)} linhas -> {n_windows} janelas")
        print(f"  X shape: {X.shape}  ({num_features} features, "
              f"seq_len={seq_len}, stride={stride})")
        print(f"  y shape: {y.shape}  (dtype={y.dtype})")
        if y.dtype == np.int64:
            unique, counts = np.unique(y, return_counts=True)
            dist = ", ".join(
                f"{INT_TO_CATEGORY.get(int(u), int(u))}={c}"
                for u, c in zip(unique, counts)
            )
            print(f"  Distribuicao de classes na saida: {dist}")

    return X, y


# ── Smoke test ────────────────────────────────────────────────────────────────

def _smoke_test():
    rng     = np.random.default_rng(seed=42)
    n       = 100
    fake_df = pd.DataFrame({
        'LONGTIME':  rng.normal(size=n).astype(np.float32),
        'INTACTIVE': rng.integers(0, 2, size=n).astype(np.float32),
        'INTMANUAL': rng.integers(0, 2, size=n).astype(np.float32),
        'CATEGORY':  rng.choice(['ilegal', 'suspeito', 'válido'], size=n),
    })

    X, y = make_windows(fake_df, seq_len=DEFAULT_SEQ_LEN, model_name="SMOKE")

    assert X.shape == (n - DEFAULT_SEQ_LEN, DEFAULT_SEQ_LEN,
                       len(DEFAULT_TRANSFORMER_FEATURES))
    assert y.shape == (n - DEFAULT_SEQ_LEN,)
    assert X.dtype == np.float32
    assert y.dtype == np.int64
    assert 0 <= y.min() and y.max() <= 2

    print("[OK]")


if __name__ == "__main__":
    _smoke_test()
