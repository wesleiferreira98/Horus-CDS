#!/usr/bin/env python3
"""Ablation: OneCycleLR com warmup + cosine annealing.

Ativa o scheduler (Smith 2018) — 10% warmup linear, cosine decay no resto.
Combina o melhor da ablation anterior: seq_len=16.

Uso:
    cd Horus-CDS && python scripts/run_v5_warmup.py
"""

import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINUX_DIR    = os.path.join(PROJECT_ROOT, "root", "Linux")
sys.path.insert(0, LINUX_DIR)

import pandas as pd
from ThreadTrain.TrainingThreadTransformer import TrainingThreadTransformer

CSV_PATH        = os.path.join(LINUX_DIR, "DadosReais", "dados_normalizados_smartgrid.csv")
EXPERIMENT_NAME = "exp_warmup"


def main():
    data   = pd.read_csv(CSV_PATH)
    thread = TrainingThreadTransformer(
        data_set=data,
        seq_len=16,           # melhor da ablation anterior
        use_scheduler=True,   # OneCycleLR ativo
        warmup_pct=0.1,       # 10% em warmup linear
    )
    thread.output_dir = os.path.join(thread.output_dir, EXPERIMENT_NAME)
    os.makedirs(thread.output_dir, exist_ok=True)
    thread.run()


if __name__ == "__main__":
    main()
