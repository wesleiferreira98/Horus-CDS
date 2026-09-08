#!/usr/bin/env python3
"""Ablation: 11 features — 3 brutas + 8 derivadas juntas.

Isola cientificamente o efeito de "adicionar features derivadas" mantendo
as brutas (LONGTIME, INTACTIVE, INTMANUAL). Fecha a ablation iniciada em
run_v5_lag_features.py, que trocou ao invés de somar.

Uso:
    cd Horus-CDS && python scripts/run_v5_all_features.py
"""

import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINUX_DIR    = os.path.join(PROJECT_ROOT, "root", "Linux")
sys.path.insert(0, LINUX_DIR)

import pandas as pd
from ThreadTrain.TrainingThreadTransformer import TrainingThreadTransformer
from TratamentoDeDados.temporal_split import DEFAULT_FEATURES
from TratamentoDeDados.windowing    import DEFAULT_TRANSFORMER_FEATURES

CSV_PATH        = os.path.join(LINUX_DIR, "DadosReais", "dados_normalizados_smartgrid.csv")
EXPERIMENT_NAME = "exp_all_features"


def main():
    data     = pd.read_csv(CSV_PATH)
    features = list(DEFAULT_TRANSFORMER_FEATURES) + list(DEFAULT_FEATURES)  # 3 + 8 = 11
    thread   = TrainingThreadTransformer(
        data_set=data,
        seq_len=16,
        feature_cols=features,
        use_scheduler=True,
        warmup_pct=0.1,
    )
    thread.output_dir = os.path.join(thread.output_dir, EXPERIMENT_NAME)
    os.makedirs(thread.output_dir, exist_ok=True)
    thread.run()


if __name__ == "__main__":
    main()
