#!/usr/bin/env python3
"""Ablation: seq_len=16 — lookback curto.

Uso:
    cd Horus-CDS && python scripts/run_v5_seqlen16.py
"""

import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINUX_DIR    = os.path.join(PROJECT_ROOT, "root", "Linux")
sys.path.insert(0, LINUX_DIR)

import pandas as pd
from ThreadTrain.TrainingThreadTransformer import TrainingThreadTransformer

CSV_PATH        = os.path.join(LINUX_DIR, "DadosReais", "dados_normalizados_smartgrid.csv")
EXPERIMENT_NAME = "exp_seqlen16"


def main():
    data   = pd.read_csv(CSV_PATH)
    thread = TrainingThreadTransformer(data_set=data, seq_len=16)
    thread.output_dir = os.path.join(thread.output_dir, EXPERIMENT_NAME)
    os.makedirs(thread.output_dir, exist_ok=True)
    thread.run()


if __name__ == "__main__":
    main()
