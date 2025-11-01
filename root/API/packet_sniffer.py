import threading
import numpy as np
import time
import os
try:
    from scapy.all import sniff, TCP # type: ignore
    SCAPY_AVAILABLE = True
except (ImportError, PermissionError):
    SCAPY_AVAILABLE = False
    print("Aviso: Scapy não disponível ou sem permissões. Modo simulação ativado.")

class PacketSniffer:
    def __init__(self, prediction_model, logger, simulation_mode=None):
        self.prediction_model = prediction_model
        self.logger = logger
        self.lock = threading.Lock()
        self.running = False
        self.simulation_thread = None
        
        # Detectar modo automaticamente
        if simulation_mode is None:
            # Modo REAL por padrão, simulação apenas como fallback
            self.simulation_mode = False
        else:
            self.simulation_mode = simulation_mode
        
        if self.simulation_mode:
            print("Modo SIMULACAO ativado (fallback - nao requer sudo)")
        else:
            print("Modo CAPTURA REAL ativado (padrao)")
    
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

    def simulate_packets(self):
        """Gera pacotes simulados para teste sem necessidade de permissões root"""
        print("Iniciando simulacao de pacotes...")
        self.running = True
        packet_count = 0
        
        while self.running:
            try:
                # Simular um pacote a cada 2-5 segundos
                time.sleep(np.random.uniform(2, 5))
                
                # Gerar características aleatórias
                features = np.random.randint(30, 100, size=(8,))
                features = features.reshape((8, 1))
                
                # Processar como se fosse um pacote real
                with self.lock:
                    is_attack = self.prediction_model.is_attack(features)
                    
                    packet_count += 1
                    packet_summary = f"Simulated TCP Packet #{packet_count}"
                    
                    if is_attack:
                        log_message = f"Ataque detectado! Requisição negada. Pacote: {packet_summary}"
                    else:
                        log_message = f"Requisição normal. Permitida. Pacote: {packet_summary}"
                    
                    self.logger.log(log_message)
                    
            except Exception as e:
                print(f"Erro na simulação: {e}")
                time.sleep(1)
    
    def start_sniffing(self):
        """Inicia captura de pacotes (real ou simulada)"""
        if self.simulation_mode:
            # Modo simulação - fallback quando sem permissões
            print("Captura SIMULADA iniciada (modo fallback)")
            self.simulation_thread = threading.Thread(target=self.simulate_packets, daemon=True)
            self.simulation_thread.start()
        else:
            # Modo real - padrão
            if not SCAPY_AVAILABLE:
                print("ERRO: Scapy nao disponivel. Alternando para modo simulacao.")
                self.simulation_mode = True
                self.start_sniffing()
                return
            
            try:
                print("Captura REAL iniciada (interface: lo)")
                sniff(prn=self.process_packet, iface='lo', store=False)
            except PermissionError:
                print("AVISO: Permissoes insuficientes para captura real.")
                print("Alternando automaticamente para modo simulacao (fallback).")
                print("Para usar modo real, execute com: sudo python app.py")
                self.simulation_mode = True
                self.start_sniffing()
    
    def stop_sniffing(self):
        """Para a captura de pacotes"""
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2)
        print("Captura interrompida")

    def process_logs(self, filtro):
        # Processar e retornar os logs
        return self.logger.process_logs(filtro)

    def generate_prediction_chart(self):
        return self.logger.generate_prediction_chart()

    def generate_log_chart(self):
        return self.logger.generate_log_chart()
