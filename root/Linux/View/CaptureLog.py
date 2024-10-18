import os
import sys
import time
import requests
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QPushButton, QInputDialog, QLabel, QMessageBox, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal

# Classe responsável por buscar e atualizar os gráficos em uma thread separada
class GraphUpdater(QThread):
    update_signal = pyqtSignal(str, str)  # Sinal para enviar os caminhos dos dois gráficos atualizados

    def __init__(self, ip, port, interval=10):
        super().__init__()
        self.ip = ip
        self.port = port
        self.interval = interval
        self.running = True  # Controle para parar a thread se necessário

    def run(self):
        log_chart_url = f'http://{self.ip}:{self.port}/log_chart'
        prediction_chart_url = f'http://{self.ip}:{self.port}/prediction_chart'

        while self.running:
            try:
                # Fazer requisição GET para a API Flask para o gráfico de logs
                log_response = requests.get(log_chart_url)
                log_response.raise_for_status()
                log_graph_path = self.save_graph(log_response.content, 'log_chart.png')

                # Fazer requisição GET para a API Flask para o gráfico de predições
                prediction_response = requests.get(prediction_chart_url)
                prediction_response.raise_for_status()
                prediction_graph_path = self.save_graph(prediction_response.content, 'prediction_chart.png')

                # Emitir o caminho dos gráficos para serem atualizados na interface
                self.update_signal.emit(log_graph_path, prediction_graph_path)

            except requests.exceptions.RequestException as e:
                print(f"Erro ao buscar os gráficos: {e}")
                self.update_signal.emit('', '')  # Emitir string vazia para lidar com erro

            time.sleep(self.interval)  # Espera o intervalo antes de buscar os gráficos novamente

    def stop(self):
        self.running = False

    def save_graph(self, graph_data, graph_name):
        # Criar o diretório de saída "LogGraph"
        output_directory = "LogGraph"
        os.makedirs(output_directory, exist_ok=True)

        # Caminho completo para salvar o gráfico
        graph_path = os.path.join(output_directory, graph_name)

        # Salvar o gráfico no diretório "LogGraph"
        with open(graph_path, 'wb') as f:
            f.write(graph_data)

        return graph_path


class CaptureLog(QWidget):
    def __init__(self):
        super().__init__()

        # Configuração da janela
        self.setWindowTitle('Análise de Requisições de Rede e Predições')
        self.setGeometry(100, 100, 1600, 600)  # Aumentar o tamanho da janela para suportar dois gráficos
        self.setStyleSheet("""
            QWidget {
                background-color: #9dcfff;
                border-radius: 10px;
            }
            QLabel {
                color: #1e25ff;
                font-size: 20px;
                padding: 10px;
                margin: 4px 2px;
            }
            QPushButton {
                background-color: #1E90FF;
                border: none;
                color: white;
                text-align: center;
                font-size: 16px;
                padding: 10px 20px;
                margin: 4px 2px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #1C86EE;
            }
            QLineEdit {
                background-color: #1E90FF;
                border: 1px solid #1C86EE;
                padding: 7px;
                border-radius: 5px;
                color: #f0f0f0;
            }
        """)

        # Layout
        layout = QVBoxLayout()

        # Botão para iniciar monitoramento
        self.button = QPushButton('Iniciar Monitoramento de Logs')
        self.button.clicked.connect(self.start_monitoring)
        layout.addWidget(self.button)

        # Layout horizontal para os gráficos
        self.graph_layout = QHBoxLayout()
        layout.addLayout(self.graph_layout)

        # Label para exibir o gráfico de logs
        self.log_graph_label = QLabel('Gráfico de Logs aparecerá aqui')
        self.graph_layout.addWidget(self.log_graph_label)

        # Label para exibir o gráfico de predições
        self.prediction_graph_label = QLabel('Gráfico de Predições aparecerá aqui')
        self.graph_layout.addWidget(self.prediction_graph_label)

        self.setLayout(layout)

        # Thread para monitoramento
        self.monitor_thread = None

        self.start_monitoring()

    def start_monitoring(self):
        # Solicitar IP e Porta no mesmo campo
        server_info, ok = QInputDialog.getText(self, 'Entrada de IP e Porta', 'Digite o IP e a porta do servidor (ex: 192.168.1.10:5000):')
        if not ok or not server_info:
            return  # Se o usuário cancelar ou não inserir, nada acontece
        
        # Separar IP e porta
        try:
            ip, port = server_info.split(':')
        except ValueError:
            self.log_graph_label.setText('Formato inválido. Use o formato IP:Porta.')
            return

        # Iniciar a thread para monitorar os gráficos periodicamente
        self.monitor_thread = GraphUpdater(ip, port, interval=10)
        self.monitor_thread.update_signal.connect(self.update_graphs)
        self.monitor_thread.start()

    def update_graphs(self, log_graph_path, prediction_graph_path):
        # Atualizar o gráfico de logs
        if log_graph_path:
            pixmap = QPixmap(log_graph_path)
            self.log_graph_label.setPixmap(pixmap)
            self.log_graph_label.setScaledContents(True)  # Ajustar o gráfico ao tamanho do label
        else:
            self.log_graph_label.setText('Erro ao buscar o gráfico de logs')

        # Atualizar o gráfico de predições
        if prediction_graph_path:
            pixmap = QPixmap(prediction_graph_path)
            self.prediction_graph_label.setPixmap(pixmap)
            self.prediction_graph_label.setScaledContents(True)  # Ajustar o gráfico ao tamanho do label
        else:
            self.prediction_graph_label.setText('Erro ao buscar o gráfico de predições')

    def closeEvent(self, event):
        if self.monitor_thread:
            self.monitor_thread.stop()  # Parar a thread quando a janela for fechada
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CaptureLog()
    window.show()
    sys.exit(app.exec_())
