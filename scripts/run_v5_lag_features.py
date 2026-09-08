#!/usr/bin/env python3
"""Ablation: 8 features derivadas do TCN (lag/MA/STD/calendário) + config final.

Testa se as features manuais ajudam o Transformer com a nova config vencedora:
seq_len=16 + OneCycleLR + warmup 10%. Se subir, features ajudam mesmo com attention.
Se não, o Transformer aprende as dependências temporais sozinho.

Uso:
    cd Horus-CDS && python scripts/run_v5_lag_features.py
"""

import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINUX_DIR    = os.path.join(PROJECT_ROOT, "root", "Linux")
sys.path.insert(0, LINUX_DIR)

import pandas as pd
from ThreadTrain.TrainingThreadTransformer import TrainingThreadTransformer
from TratamentoDeDados.temporal_split import DEFAULT_FEATURES

CSV_PATH        = os.path.join(LINUX_DIR, "DadosReais", "dados_normalizados_smartgrid.csv")
EXPERIMENT_NAME = "exp_lag_features"


def main():
    data   = pd.read_csv(CSV_PATH)
    thread = TrainingThreadTransformer(
        data_set=data,
        seq_len=16,                          # melhor da ablation de lookback
        feature_cols=list(DEFAULT_FEATURES), # 8 features derivadas (igual TCN)
        use_scheduler=True,                  # config final do V5
        warmup_pct=0.1,
    )
    thread.output_dir = os.path.join(thread.output_dir, EXPERIMENT_NAME)
    os.makedirs(thread.output_dir, exist_ok=True)
    thread.run()


if __name__ == "__main__":
    main()
