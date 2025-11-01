#!/usr/bin/env python3
"""
Script de teste para o console de treinamento
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from View.TrainingConsole import TrainingConsole, ConsoleCapture
import time

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Teste Console de Treinamento")
        self.setGeometry(100, 100, 800, 600)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Console
        self.console = TrainingConsole()
        layout.addWidget(self.console)
        
        # Botão de teste
        test_button = QPushButton("Simular Treinamento")
        test_button.clicked.connect(self.simulate_training)
        layout.addWidget(test_button)
        
        self.console_capture = None
    
    def simulate_training(self):
        """Simula saída de treinamento"""
        self.console.start_capture()
        
        # Redirecionar stdout
        self.console_capture = ConsoleCapture(self.console, sys.stdout)
        sys.stdout = self.console_capture
        
        # Simular mensagens de treinamento (com códigos ANSI)
        print("Features temporais adicionadas ao conjunto de Treino: 17160 amostras restantes")
        print("Features temporais adicionadas ao conjunto de Teste: 4288 amostras restantes")
        print("Aplicando normalização (StandardScaler apenas no treino)...")
        print("Dados preparados com sucesso!")
        print("  Treino: 17160 amostras")
        print("  Teste: 4288 amostras")
        print("")
        print("AVISO: Do not pass an input_shape/input_dim argument to a layer.")
        print("Realizando busca de hiperparâmetros...")
        print("Fitting 3 folds for each of 5 candidates, totalling 15 fits")
        print("I0000 00:00:1762035419.936330   77358 device_compiler.h:196] Compiled cluster using XLA!")
        
        # Simular barras de progresso com códigos ANSI (como aparecem no Keras)
        print("\x1B[1m139/997\x1B[0m \x1B[32m━━\x1B[0m\x1B[37m━━━━━━━━━━━━━━━━━━\x1B[0m \x1B[1m1s\x1B[0m 2ms/step - loss: 0.5641 - mse: 0.5641")
        print("\x1B[1m358/997\x1B[0m \x1B[32m━━━━━━━\x1B[0m\x1B[37m━━━━━━━━━━━━━\x1B[0m \x1B[1m0s\x1B[0m 2ms/step - loss: 0.3400 - mse: 0.3400")
        print("\x1B[1m997/997\x1B[0m \x1B[32m━━━━━━━━━━━━━━━━━━━━\x1B[0m\x1B[37m\x1B[0m \x1B[1m2s\x1B[0m 2ms/step - loss: 0.2186 - mse: 0.2186")
        
        print("179/179 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step")
        print("")
        print("SUCESSO: Modelo treinado com sucesso!")
        print("GPU: NVIDIA GeForce RTX 4050 detectada")
        
        # Restaurar stdout
        sys.stdout = self.console_capture.original_stream
        self.console.log_signal.emit("")
        self.console.log_signal.emit("=== Simulação finalizada ===")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())
