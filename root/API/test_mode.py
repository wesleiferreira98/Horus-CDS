#!/usr/bin/env python3
"""Script de teste para verificar o modo de operação da API"""

import sys
import os

# Adicionar o diretório ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from packet_sniffer import PacketSniffer
from prediction import Prediction
from logger import Logger

# Setup básico
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
LOG_DIR = os.path.join(BASE_DIR, 'logs', 'test')
os.makedirs(LOG_DIR, exist_ok=True)

try:
    # Carregar modelo TCN
    model_path = os.path.join(MODELS_DIR, "tcn_model.keras")
    
    print(f"Carregando modelo: {model_path}")
    print(f"Carregando scaler: {SCALER_PATH}")
    
    prediction_model = Prediction(model_path, SCALER_PATH)
    logger = Logger(os.path.join(LOG_DIR, "packet_logs.txt"))
    
    print("\n=== Teste com simulation_mode=None (deve usar modo REAL como padrão) ===")
    sniffer = PacketSniffer(prediction_model, logger, simulation_mode=None)
    print(f"Modo simulação: {sniffer.simulation_mode}")
    print(f"UID do processo: {os.geteuid()} (0 = root)")
    
    print("\n=== Resultado ===")
    if sniffer.simulation_mode:
        print("FALLBACK: Modo simulação ativado (sem permissões root)")
    else:
        print("PADRÃO: Modo captura real ativado (com permissões root)")
    
except Exception as e:
    print(f"Erro: {e}")
    import traceback
    traceback.print_exc()
