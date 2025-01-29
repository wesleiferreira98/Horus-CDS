import threading
import numpy as np
from scapy.all import sniff, TCP # type: ignore
import logging
import io
import matplotlib.pyplot as plt
import re

class PacketSniffer:
    def __init__(self, prediction_model, logger):
        self.prediction_model = prediction_model
        self.logger = logger
        self.lock = threading.Lock()
    
    def update_prediction_model(self, new_model):
        with self.lock:
            self.prediction_model = new_model

    def process_packet(self, packet):
        with self.lock:  # Garantir acesso seguro ao modelo
            if packet.haslayer(TCP):
                features = self.extract_features(packet)
                is_attack = self.prediction_model.is_attack(features)

                if is_attack:
                    log_message = f"Ataque detectado! Requisição negada. Pacote: {packet.summary()}"
                    self.logger.log(log_message)
                else:
                    log_message = f"Requisição normal. Permitida. Pacote: {packet.summary()}"
                    self.logger.log(log_message)
            else:
                log_message = f"Pacote não TCP: {packet.summary()}"
                self.logger.log(log_message)

    def extract_features(self, packet):
        # Gerar características aleatórias
        features = np.random.randint(30, 100, size=(8,))
        return features.reshape((8, 1))

    def start_sniffing(self):
        print("Captura iniciada")
        sniff(prn=self.process_packet, iface='lo')

    def process_logs(self, filtro):
        # Processar e retornar os logs
        return self.logger.process_logs(filtro)

    def generate_prediction_chart(self):
        return self.logger.generate_prediction_chart()

    def generate_log_chart(self):
        return self.logger.generate_log_chart()
