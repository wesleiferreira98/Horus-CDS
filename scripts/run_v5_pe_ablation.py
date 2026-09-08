#!/usr/bin/env python3
"""Ablation: positional embedding aprendido vs senoidal.

Uso:
    cd Horus-CDS && python scripts/run_v5_pe_ablation.py [--pe learned|sinusoidal|both]
"""

import argparse, os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINUX_DIR    = os.path.join(PROJECT_ROOT, "root", "Linux")
sys.path.insert(0, LINUX_DIR)

import pandas as pd
from ThreadTrain.TrainingThreadTransformer import TrainingThreadTransformer

CSV_PATH = os.path.join(LINUX_DIR, "DadosReais", "dados_normalizados_smartgrid.csv")
BASE_DIR = os.path.join(LINUX_DIR, "DadosDoPostreino", "ModelosNew", "Transformer")

PE_EXPERIMENTS = {
    'learned':    'exp_pe_learned',
    'sinusoidal': 'exp_pe_sinusoidal',
}


def run_experiment(data, pe_type):
    thread = TrainingThreadTransformer(data_set=data, seq_len=16,
                                       pos_embed_type=pe_type)
    thread.output_dir = os.path.join(BASE_DIR, PE_EXPERIMENTS[pe_type])
    os.makedirs(thread.output_dir, exist_ok=True)
    thread.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pe', choices=['learned', 'sinusoidal', 'both'],
                        default='both')
    args = parser.parse_args()

    data   = pd.read_csv(CSV_PATH)
    to_run = ['learned', 'sinusoidal'] if args.pe == 'both' else [args.pe]
    for pe_type in to_run:
        run_experiment(data, pe_type)


if __name__ == "__main__":
    main()
