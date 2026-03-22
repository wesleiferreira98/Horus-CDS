from PyQt5.QtWidgets import (
    QWidget, QTableView, QPushButton, QFileDialog, QVBoxLayout, 
    QScrollArea, QLabel, QTableWidgetItem, QMessageBox, QProgressBar,
    QRadioButton, QTableWidget, QHBoxLayout, QSpacerItem, QSizePolicy, QCheckBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, Qt, QThreadPool, QTimer, pyqtSlot
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_config import has_gpu, is_using_gpu
from Models.PandasModel import PandasModel
from View.CustomMessageBox import CustomMessageBox
from View.LoadDataThread import LoadDataThread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from ThreadTrain import (
    TrainingThreadGRU, TrainingThreadLSTM, TrainingThreadARIMA, 
    TrainingThreadTCN, TrainingThreadKNN, TrainingThreadRNN, 
    TrainingThreadRandomForest, TrainingThreadMLP,
    GRU_corrigido, LSTM_corrigido, RNN_corrigido, TCN_corrigido
)
import os
import joblib
import seaborn as sns
import csv
from View.CaptureLog import CaptureLog
from View.TrainingConsole import TrainingConsole, ConsoleCapture

class SPTI(QWidget):
    
    update_prediction_chart = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, str)
    update_metrics_chart = pyqtSignal(float, float,str)
    update_progress = pyqtSignal(float)
    update_metrics_chart_boxplot = pyqtSignal(np.ndarray, np.ndarray,str)
    update_prediction_chartCNN = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        # Definir diretório base do projeto (root/Linux)
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.initUI()
        self.processed_data = False
        self.threadpool = QThreadPool()
        self.model_select = None
        self.current_training_model = None
        
        # Configurar captura de console
        self.console_capture = None
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
    
    def closeEvent(self, event):
        """Garantir que streams sejam restaurados ao fechar"""
        try:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
        except:
            pass
        event.accept()
    
    def _get_model_directory_suffix(self, modelname):
        """
        Determina o sufixo do diretório baseado no nome do modelo.
        Modelos corrigidos vão para ModelosNew, outros para ModelosOlds.
        """
        if "Corrigido" in modelname or "corrigido" in modelname:
            return "DadosDoPostreino/ModelosNew"
        else:
            return "DadosDoPostreino/ModelosOlds"

    def initUI(self):
        self.setWindowTitle('Central de Treinamento de Modelos (Horus-CDS)')
        self.setGeometry(100, 100, 1713, 896)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        main_widget = QWidget()
        self.scroll_area.setWidget(main_widget)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.scroll_area)

        self.main_layout = QVBoxLayout(main_widget)

        # Creating layout for tables and grid
        tables_and_grid_layout = QVBoxLayout()

        self.canvas_prediction = FigureCanvas(plt.Figure())
        self.canvas_metrics = FigureCanvas(plt.Figure())

        tables_and_grid_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.main_layout.addLayout(tables_and_grid_layout)

        # Creating buttons and labels
        self.rb_RandomForest = QRadioButton("Treinar com Random Forest")
        self.rb_KNN = QRadioButton("Treinar com Modelo KNN")
        self.rb_LSTM = QRadioButton("Treinar com Rede Neural LSTM")
        self.rb_ARIMA = QRadioButton("Treinar com Modelo ARIMA")
        self.rb_TCN = QRadioButton("Treinar com rede neural TCN")
        self.rb_RNN = QRadioButton("Treinar com rede neural RNN")
        self.rb_GRU = QRadioButton("Treinar com rede neural GRU")
        self.rb_HorusV0 = QRadioButton("Treinar com rede neural MLP")
        
        # Modelos corrigidos
        self.rb_GRU_corrigido = QRadioButton("Treinar com GRU Corrigido")
        self.rb_LSTM_corrigido = QRadioButton("Treinar com LSTM Corrigido")
        self.rb_RNN_corrigido = QRadioButton("Treinar com RNN Corrigido")
        self.rb_TCN_corrigido = QRadioButton("Treinar com TCN Corrigido")

        self.train_button = QPushButton('Selecionar Data SET')
        self.train_model_button = QPushButton('Iniciar Treinamento')
        self.model_select_train = QLabel("Selecione o Modelo de treinamento")
        self.fast_curves_checkbox = QCheckBox("Modo rapido para gerar ROC/PR")
        self.fast_curves_checkbox.setToolTip("Reduz amostra, busca e iteracoes dos modelos antigos para gerar curvas rapidamente.")
        self.grafic_models_old_button = QPushButton('📊 Dados Treinamento - Modelos Antigos')
        self.grafic_models_new_button = QPushButton('📊 Dados Treinamento - Modelos Novos')
        self.grafic_logs = QPushButton('Capturar dados de Log') 
        self.label_train_test = QLabel("Dados de Treino")
        self.label_progess_test = QLabel("Progresso do Teste")

        # Label informativo sobre GPU
        self.gpu_info_label = QLabel()
        if has_gpu():
            if is_using_gpu():
                self.gpu_info_label.setText("GPU Detectada e Ativa para Treinamento")
                self.gpu_info_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.gpu_info_label.setText("GPU Detectada mas Desabilitada (modo CPU)")
                self.gpu_info_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.gpu_info_label.setText("GPU Não Detectada (usando CPU)")
            self.gpu_info_label.setStyleSheet("color: red; font-weight: bold;")

        # Organizing radio buttons and labels
        rb_train_layoutV1 = QVBoxLayout()
        rb_train_layoutV1.addWidget(self.rb_RandomForest)
        rb_train_layoutV1.addWidget(self.rb_KNN)
        rb_train_layoutV1.addWidget(self.rb_TCN)
        rb_train_layoutV1.addWidget(self.rb_ARIMA)

        rb_train_layoutV2 = QVBoxLayout()
        rb_train_layoutV2.addWidget(self.rb_LSTM)
        rb_train_layoutV2.addWidget(self.rb_RNN)
        rb_train_layoutV2.addWidget(self.rb_GRU)
        rb_train_layoutV2.addWidget(self.rb_HorusV0)
        
        rb_train_layoutV3 = QVBoxLayout()
        rb_train_layoutV3.addWidget(self.rb_GRU_corrigido)
        rb_train_layoutV3.addWidget(self.rb_LSTM_corrigido)
        rb_train_layoutV3.addWidget(self.rb_RNN_corrigido)
        rb_train_layoutV3.addWidget(self.rb_TCN_corrigido)

        rb_train_layout = QHBoxLayout()
        rb_train_layout.addLayout(rb_train_layoutV1)
        rb_train_layout.addLayout(rb_train_layoutV2)
        rb_train_layout.addLayout(rb_train_layoutV3)

        integration_rb_label = QVBoxLayout()
        integration_rb_label.addWidget(self.model_select_train)
        integration_rb_label.addLayout(rb_train_layout)
        integration_rb_label.addWidget(self.fast_curves_checkbox)
        
        # Adicionar label informativo de GPU
        gpu_info_layout = QHBoxLayout()
        gpu_info_layout.addWidget(self.gpu_info_label)
        gpu_info_layout.setAlignment(Qt.AlignCenter)
        integration_rb_label.addLayout(gpu_info_layout)

        label_train_layout = QHBoxLayout()
        label_train_layout.addWidget(self.label_train_test)
        label_train_layout.setAlignment(Qt.AlignCenter)

        label_progress_layout = QHBoxLayout()
        label_progress_layout.addWidget(self.label_progess_test)
        label_progress_layout.setAlignment(Qt.AlignCenter)

        self.progress_bar = QProgressBar()
        
        # Console de treinamento
        self.training_console = TrainingConsole()
        self.training_console.setVisible(False)  # Inicialmente oculto

       

        self.train_data_table = QTableWidget()
        self.train_data_table.setColumnCount(11)
        self.train_data_table.setFixedHeight(500)
        self.train_data_table.setFixedWidth(850)

        table_layout = QHBoxLayout()
        table_layout.addStretch()  # Adiciona espaço flexível antes da tabela
        table_layout.addWidget(self.train_data_table)  # Adiciona a tabela
        table_layout.addStretch()  # Adiciona espaço flexível depois da tabela

        # Adding widgets to the main layout
        self.main_layout.addWidget(self.train_button)
        self.main_layout.addWidget(self.train_model_button)
        
        # Layout horizontal para os dois botões de dados de treinamento
        grafic_buttons_layout = QHBoxLayout()
        grafic_buttons_layout.addWidget(self.grafic_models_old_button)
        grafic_buttons_layout.addWidget(self.grafic_models_new_button)
        self.main_layout.addLayout(grafic_buttons_layout)
        
        self.main_layout.addWidget(self.grafic_logs)
        self.main_layout.addLayout(integration_rb_label)
        self.main_layout.addLayout(label_progress_layout)
        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addWidget(self.training_console)  # Console de treinamento
        self.main_layout.addLayout(label_train_layout)
        self.main_layout.addLayout(table_layout)

        # Connecting signals to slots
        self.train_button.clicked.connect(self.select_train_data)
        self.train_model_button.clicked.connect(self.start_training)
        self.grafic_models_old_button.clicked.connect(self.plot_metrics_shared_old)
        self.grafic_models_new_button.clicked.connect(self.plot_metrics_shared_new)
        self.grafic_logs.clicked.connect(self.cap_log)
        self.rb_RandomForest.toggled.connect(self.on_radio_button_toggled)
        self.rb_KNN.toggled.connect(self.on_radio_button_toggled)
        self.rb_LSTM.toggled.connect(self.on_radio_button_toggled)
        self.rb_TCN.toggled.connect(self.on_radio_button_toggled)
        self.rb_RNN.toggled.connect(self.on_radio_button_toggled)
        self.rb_GRU.toggled.connect(self.on_radio_button_toggled)
        self.rb_ARIMA.toggled.connect(self.on_radio_button_toggled)
        self.rb_GRU_corrigido.toggled.connect(self.on_radio_button_toggled)
        self.rb_LSTM_corrigido.toggled.connect(self.on_radio_button_toggled)
        self.rb_RNN_corrigido.toggled.connect(self.on_radio_button_toggled)
        self.rb_TCN_corrigido.toggled.connect(self.on_radio_button_toggled)

        self.setStyleSheet("""
            QWidget {
                background-color: #9dcfff;
                border-radius: 10px;
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
            QProgressBar {
                border: none;
                text-align: center;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 8px;
            }
            QProgressBar::chunk {
                background-color: #1E90FF;
                width: 10px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #1C86EE;
            }
        """)
        
        # Estilos específicos para os botões de dados de treinamento
        self.grafic_models_old_button.setStyleSheet("""
            QPushButton {
                background-color: #FF6B6B;  /* Vermelho suave para modelos antigos */
                border: 2px solid #C92A2A;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 12px 20px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #FA5252;
                border: 2px solid #A61E1E;
            }
        """)
        
        self.grafic_models_new_button.setStyleSheet("""
            QPushButton {
                background-color: #51CF66;  /* Verde para modelos novos/corrigidos */
                border: 2px solid #2F9E44;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 12px 20px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #40C057;
                border: 2px solid #2B8A3E;
            }
        """)

       
    def cap_log(self):
        # Instancia e exibe a janela CaptureLog
        self.capture = CaptureLog()  # Salva como um atributo da classe para manter a referência
        self.capture.show()
    def select_train_data(self):
        file_dialog = QFileDialog()
        # Abrir diálogo na pasta DadosReais
        default_dir = os.path.join(self.base_dir, 'DadosReais')
        train_data_file, _ = file_dialog.getOpenFileName(self, 'Selecionar Data SET', default_dir, 'CSV Files (*.csv)')
        if train_data_file:
            try:
                self.train_data = pd.read_csv(train_data_file)
                self.clear_table(self.train_data_table)
                self.train_data_table.setRowCount(len(self.train_data))
                self.train_data_table.setColumnCount(len(self.train_data.columns))
                self.train_data_table.setHorizontalHeaderLabels(self.train_data.columns)

                self.message_box = CustomMessageBox(self)
                self.message_box.setWindowTitle("Carregando arquivo")
                self.message_box.setText("Por favor, espere enquanto o arquivo está sendo carregado...")
                self.message_box.setStandardButtons(QMessageBox.Cancel)
                self.message_box.show_progress_bar()
                self.message_box.show()

                self.populate_table(self.train_data, self.train_data_table)
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao carregar o arquivo: {str(e)}")

    def clear_table(self, table_widget):
        table_widget.clearContents()
        table_widget.setRowCount(0)

    def populate_table(self, data, table_widget):
        self.table_widget = table_widget
        self.data = data
        self.row = 0
        self.chunk_size = 100
        self.total_rows = len(data)
        self.total_chunks = (self.total_rows // self.chunk_size) + 1
        self.timer = QTimer()
        self.timer.timeout.connect(self.load_next_chunk)
        self.timer.start(50)

    def load_next_chunk(self):
        if self.row < self.total_rows:
            end_row = min(self.row + self.chunk_size, self.total_rows)
            for i in range(self.row, end_row):
                for j in range(len(self.data.columns)):
                    try:
                        item = QTableWidgetItem(str(self.data.iloc[i, j]))
                        self.table_widget.setItem(i, j, item)
                    except IndexError as e:
                        QMessageBox.critical(self, "Erro", f"Falha ao acessar o índice ({i}, {j}): {str(e)}")
                        self.timer.stop()
                        return

            self.row = end_row
            progress_value = int((self.row / self.total_rows) * 100)
            self.message_box.set_progress_value(progress_value)
        else:
            self.timer.stop()
            self.message_box.hide()

    def load_train_data(self, file_path):
        self.show_progress_bar()
        self.load_thread = LoadDataThread(file_path)
        self.load_thread.update_progress.connect(self.update_progress2)
        self.load_thread.data_loaded.connect(self.on_train_data_loaded)
        self.load_thread.start()

    def on_train_data_loaded(self, data):
        self.train_data = data
        self.populate_table(data, self.train_data_table)
    
    def show_progress_bar(self):
        self.progress = CustomMessageBox(self)
        self.progress.setWindowTitle("Carregando Dados")
        self.progress.setText("Por favor, aguarde enquanto os dados são carregados...")
        self.progress.show_progress_bar()
        self.progress.show()

    def update_progress(self, progress):
        progress_int = int(progress * 100)
        self.progress_bar.setValue(progress_int)

    def start_training(self):
        if not hasattr(self, 'train_data') or self.train_data is None:
            QMessageBox.critical(self, "Erro", "Nenhum dado de treino carregado.")
            return

        selected_model = self.get_selected_model()
        if selected_model is None:
            QMessageBox.critical(self, "Erro", "Nenhum modelo de treinamento selecionado.")
            return

        self.apply_fast_mode_setting()
        self.current_training_model = selected_model

        QMessageBox.information(self, "Treinamento", f"Treinamento iniciado com o modelo {selected_model}.")
        
        # Ativar console e iniciar captura de logs
        try:
            self.training_console.setVisible(True)
            self.training_console.start_capture()
            
            # Salvar referencias originais
            if not hasattr(self, '_original_stdout'):
                self._original_stdout = sys.stdout
                self._original_stderr = sys.stderr
            
            # Redirecionar stdout e stderr para o console
            self.console_capture = ConsoleCapture(self.training_console, self._original_stdout)
            sys.stdout = self.console_capture
            sys.stderr = self.console_capture
        except Exception as e:
            print(f"Erro ao configurar console: {e}")
        
        self.train_model(selected_model)

    def on_radio_button_toggled(self):
        selected_model = self.get_selected_model()
        if selected_model:
            self.model_select_train.setText(f"Treinar com Modelo {selected_model}")
            self.update_fast_mode_availability(selected_model)
    
    def get_selected_model(self):
        if self.rb_RandomForest.isChecked():
            return 'Random Forest'
        elif self.rb_KNN.isChecked():
            return 'KNN'
        elif self.rb_LSTM.isChecked():
            return 'LSTM'
        elif self.rb_ARIMA.isChecked():
            return 'ARIMA'
        elif self.rb_TCN.isChecked():
            return 'TCN'
        elif self.rb_RNN.isChecked():
            return 'RNN'
        elif self.rb_GRU.isChecked():
            return 'GRU'
        elif self.rb_HorusV0.isChecked():
            return 'Horus-V0'
        elif self.rb_GRU_corrigido.isChecked():
            return 'GRU_corrigido'
        elif self.rb_LSTM_corrigido.isChecked():
            return 'LSTM_corrigido'
        elif self.rb_RNN_corrigido.isChecked():
            return 'RNN_corrigido'
        elif self.rb_TCN_corrigido.isChecked():
            return 'TCN_corrigido'
        return None

    def apply_fast_mode_setting(self):
        if self.fast_curves_checkbox.isChecked():
            os.environ["HORUS_FAST_CURVES"] = "1"
        else:
            os.environ.pop("HORUS_FAST_CURVES", None)

    def update_fast_mode_availability(self, selected_model):
        old_models = {
            'Random Forest',
            'KNN',
            'LSTM',
            'ARIMA',
            'TCN',
            'RNN',
            'GRU',
            'Horus-V0',
        }

        is_old_model = selected_model in old_models
        self.fast_curves_checkbox.setEnabled(is_old_model)

        if is_old_model:
            self.fast_curves_checkbox.setText("Modo rapido para gerar ROC/PR")
            self.fast_curves_checkbox.setToolTip(
                "Reduz amostra, busca e iteracoes dos modelos antigos para gerar curvas rapidamente."
            )
        else:
            self.fast_curves_checkbox.setChecked(False)
            self.fast_curves_checkbox.setText("Modo rapido indisponivel para modelos corrigidos")
            self.fast_curves_checkbox.setToolTip(
                "Os modelos corrigidos nao usam o modo rapido de curvas da linha antiga."
            )

    def train_model(self, model_name):
        if model_name == 'GRU':
            self.train_thread = TrainingThreadGRU.TrainingThreadGRU(self.train_data)
        elif model_name == 'LSTM':
            self.train_thread = TrainingThreadLSTM.TrainingThreadLSTM(self.train_data)
        elif model_name == 'ARIMA':
            self.train_thread = TrainingThreadARIMA.TrainingThreadARIMA(self.train_data)
        elif model_name == 'TCN':
            self.train_thread = TrainingThreadTCN.TrainingThreadTCN(self.train_data)
        elif model_name == 'KNN':
            self.train_thread = TrainingThreadKNN.TrainingThreadKNN(self.train_data)
        elif model_name == 'RNN':
            self.train_thread = TrainingThreadRNN.TrainingThreadRNN(self.train_data)
        elif model_name == 'Random Forest':
            self.train_thread = TrainingThreadRandomForest.TrainingThreadRandomForest(self.train_data)
        elif model_name == 'Horus-V0':
            self.train_thread = TrainingThreadMLP.TrainingThreadMLP(self.train_data)
        elif model_name == 'GRU_corrigido':
            self.train_thread = GRU_corrigido.TrainingThreadGRUCorrigido(self.train_data)
        elif model_name == 'LSTM_corrigido':
            self.train_thread = LSTM_corrigido.TrainingThreadLSTMCorrigido(self.train_data)
        elif model_name == 'RNN_corrigido':
            self.train_thread = RNN_corrigido.TrainingThreadRNNCorrigido(self.train_data)
        elif model_name == 'TCN_corrigido':
            self.train_thread = TCN_corrigido.TrainingThreadTCNCorrigido(self.train_data)
        else:
            QMessageBox.critical(self, "Erro", "Modelo de treinamento inválido.")
            return

        self.train_thread.update_progress.connect(self.update_progress)
        self.train_thread.update_prediction_chart.connect(self.plot_prediction)
        self.train_thread.update_metrics_chart.connect(self.plot_metrics)
        self.train_thread.update_metrics_chart_boxplot.connect(self.update_prediction_boxplot)
        self.train_thread.finished.connect(self.on_training_finished)
        self.train_thread.start()
    
    def on_training_finished(self):
        """Callback quando treinamento finaliza (com sucesso ou erro)"""
        try:
            # Restaurar streams
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
            self.console_capture = None
            
            if hasattr(self, 'training_console'):
                self.training_console.log_signal.emit("")
                self.training_console.log_signal.emit("=== Processo finalizado ===")

            self.show_roc_pr_curves_for_current_model()
        except Exception as e:
            print(f"Erro no callback de finalizacao: {e}")

    def get_report_curve_model_names(self, selected_model):
        mapping = {
            'Random Forest': ['Modelo RF'],
            'KNN': ['Modelo KNN'],
            'LSTM': ['Modelo LSTM'],
            'ARIMA': ['Modelo ARIMA'],
            'TCN': ['Modelo TCN'],
            'RNN': ['Modelo RNN'],
            'GRU': ['Modelo GRU'],
            'Horus-V0': ['Horus-V0'],
            'GRU_corrigido': ['Modelo GRU Corrigido'],
            'LSTM_corrigido': ['Modelo LSTM Corrigido'],
            'RNN_corrigido': ['Modelo RNN Corrigido'],
            'TCN_corrigido': ['Modelo TCN Corrigido'],
        }
        return mapping.get(selected_model, [])

    def show_roc_pr_curves_for_current_model(self):
        if not self.current_training_model:
            return

        model_dir_suffix = self._get_model_directory_suffix(self.current_training_model)
        curve_directory = os.path.join(self.base_dir, model_dir_suffix, "CurvasROC_PR")

        if not os.path.isdir(curve_directory):
            return

        for report_name in self.get_report_curve_model_names(self.current_training_model):
            roc_path = os.path.join(curve_directory, f"roc_curve_{report_name}.jpg")
            pr_path = os.path.join(curve_directory, f"pr_curve_{report_name}.jpg")

            if os.path.exists(roc_path):
                self.show_image(roc_path)
            if os.path.exists(pr_path):
                self.show_image(pr_path)



    def plot_prediction(self, y_true, y_pred, dates, modelname):
        print(f"\n{'='*80}")
        print(f"   plot_prediction() chamado para: {modelname}")
        print(f"   y_true shape: {np.array(y_true).shape}")
        print(f"   y_pred shape: {np.array(y_pred).shape}")
        print(f"   dates shape: {np.array(dates).shape}")
        print(f"{'='*80}\n")
        
        # Determinar diretório baseado no tipo de modelo (Old/New)
        model_dir_suffix = self._get_model_directory_suffix(modelname)
        self.output_directory = os.path.join(self.base_dir, model_dir_suffix, "PrevisoesDosModelos")
        os.makedirs(self.output_directory, exist_ok=True)
        
        # Se for modelo corrigido, ler dados do CSV
        if "Corrigido" in modelname or "corrigido" in modelname:
            print(f"Modelo corrigido detectado: {modelname}")
            print(f"Lendo dados do CSV para gerar gráfico...")
            
            # Construir caminho do CSV
            csv_dir = os.path.join(self.base_dir, model_dir_suffix, "RelatorioDosModelos(CSV)")
            # Extrair nome base do modelo (ex: "Modelo TCN Corrigido" -> "TCN_Corrigido")
            model_parts = modelname.replace("Modelo ", "").replace(" ", "_")
            csv_filename = os.path.join(csv_dir, f"{model_parts}_model_results.csv")
            
            print(f"🔍 Buscando CSV em: {csv_filename}")
            
            try:
                # Ler CSV
                df_csv = pd.read_csv(csv_filename)
                print(f"   CSV carregado com sucesso!")
                print(f"   Arquivo: {csv_filename}")
                print(f"   Colunas: {df_csv.columns.tolist()}")
                print(f"   Total de linhas: {len(df_csv)}")
                
                # Extrair dados do CSV (já desnormalizados)
                dates = pd.to_datetime(df_csv['Data'])
                y_true = df_csv['Valor_Real'].values
                y_pred = df_csv['Valor_Previsto'].values
                
                print(f"   Valores extraídos do CSV:")
                print(f"     y_true: min={y_true.min():.4f}, max={y_true.max():.4f}, mean={y_true.mean():.4f}")
                print(f"     y_pred: min={y_pred.min():.4f}, max={y_pred.max():.4f}, mean={y_pred.mean():.4f}")
                
                # Usar apenas os 20 primeiros dados
                y_true = y_true[:20]
                y_pred = y_pred[:20]
                dates = dates[:20]
                
                print(f"Usando {len(y_true)} pontos para o gráfico")
                
                # Dados do CSV já estão desnormalizados, apenas garantir formato correto
                y_true = np.ravel(y_true)
                y_pred = np.ravel(y_pred)
                dates = np.ravel(dates)
                
                print(f"Dados do CSV prontos para plotagem (já desnormalizados)")
                
            except Exception as e:
                print(f"   ERRO ao ler CSV!")
                print(f"   Arquivo tentado: {csv_filename}")
                print(f"   Erro: {e}")
                print(f"   Traceback completo:")
                import traceback
                traceback.print_exc()
                print(f"Usando dados passados como fallback...")
                # Fallback para dados originais
                dates = pd.to_datetime(dates)
                y_true = y_true[:20]
                y_pred = y_pred[:20]
                dates = dates[:20]
                y_true = np.ravel(y_true)
                y_pred = np.ravel(y_pred)
                dates = np.ravel(dates)
        else:
            print(f"Modelo antigo (não corrigido): {modelname}")
            # Modelos antigos - comportamento original
            try:
                # Carregar o scaler salvo
                scaler_file = os.path.join(self.base_dir, "TratamentoDeDados", "scaler.pkl")
                scaler = joblib.load(scaler_file)
            except Exception as e:
                print(f"Aviso: Não foi possível carregar o scaler: {e}")
                print("Gerando gráfico com valores normalizados...")
                scaler = None

            # Convertendo as datas para o formato datetime
            dates = pd.to_datetime(dates)
            
            # Usar apenas os 20 primeiros dados
            y_true = y_true[:20]
            y_pred = y_pred[:20]
            dates = dates[:20]

            # Desnormalizar os dados (se scaler disponível) - APENAS MODELOS ANTIGOS
            if scaler is not None:
                try:
                    y_true = scaler.inverse_transform(y_true.reshape(-1, 1))
                    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
                except Exception as e:
                    print(f"Aviso: Erro ao desnormalizar dados: {e}")
                    print("Usando valores sem desnormalização...")
                    y_true = y_true.reshape(-1, 1)
                    y_pred = y_pred.reshape(-1, 1)
            else:
                y_true = y_true.reshape(-1, 1)
                y_pred = y_pred.reshape(-1, 1)

            y_true = np.ravel(y_true)
            y_pred = np.ravel(y_pred)
            dates = np.ravel(dates)

        print(f"\n📊 Preparando dados para plotagem:")
        print(f"   dates: {len(dates)} pontos")
        print(f"   y_true: {len(y_true)} pontos (min={np.min(y_true):.4f}, max={np.max(y_true):.4f})")
        print(f"   y_pred: {len(y_pred)} pontos (min={np.min(y_pred):.4f}, max={np.max(y_pred):.4f})")
        
        df = pd.DataFrame({
            'dates': dates,
            'y_true': y_true,
            'y_pred': y_pred
        })

        df_sorted = df.sort_values(by='y_true', ascending=False)
        dates_sorted = df_sorted['dates']
        y_true_sorted = df_sorted['y_true']
        y_pred_sorted = df_sorted['y_pred']
        
        print(f"\n🎨 Gerando gráfico matplotlib...")
        print(f"   Título: Previsão de LONGTIME vs. Valor Real {modelname}")

        plt.figure(figsize=(10, 6))
        date_strings_sorted = [str(date.date()) for date in dates_sorted]

        # Adicionar um pequeno deslocamento no eixo y para evitar sobreposição
        y_pred_offset = y_pred_sorted + 0.03  # Ajuste o valor conforme necessário

        # Usar círculos para valores reais e quadrados para previstos
        plt.scatter(date_strings_sorted, y_true_sorted, label='Valor Real', color='blue', s=300, marker='o')  
        plt.scatter(date_strings_sorted, y_pred_offset, label='Previsão', color='red', s=100, marker='s')

        plt.xlabel('Data')
        plt.ylabel('LONGTIME')
        plt.title(f'Previsão de LONGTIME vs. Valor Real {modelname}') 
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        prediction_filename = os.path.join(self.output_directory, f"prediction_plot_{modelname}.jpg")
        plt.savefig(prediction_filename)
        print(f"Gráfico de previsão (bolinhas azuis e quadrados vermelhos) salvo em: {prediction_filename}")
        plt.close()
        self.show_image(prediction_filename)
        self.plot_prediction_block(y_true, y_pred, modelname)
        self.plot_prediction_boxplot(y_true, y_pred, modelname)



    def plot_prediction_block(self, y_true, y_pred, modelname):
        # Carregar o scaler salvo
        scaler_file = os.path.join(self.base_dir, "TratamentoDeDados", "scaler.pkl")
        scaler = joblib.load(scaler_file)

        # Determinar diretório baseado no tipo de modelo (Old/New)
        model_dir_suffix = self._get_model_directory_suffix(modelname)
        self.output_directory = os.path.join(self.base_dir, model_dir_suffix, "PrevisoesDosModelos(Block)")
        os.makedirs(self.output_directory, exist_ok=True)

        # Desnormalizar os dados
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1))
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

        # Garantir que y_true e y_pred são unidimensionais
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)

        # Calcular a diferença entre previstos e reais
        difference = y_pred - y_true

        plt.figure(figsize=(10, 6))

        # Plotando os blocos de valores reais, previstos e diferença
        bar_width = 0.2
        index = np.arange(len(y_true))

        plt.bar(index, y_true, bar_width, label='Valor Real', color='blue')
        plt.bar(index + bar_width, y_pred, bar_width, label='Previsão', color='red')
        plt.bar(index + 2 * bar_width, difference, bar_width, label='Diferença (Previsto - Real)', color='green')

        plt.xlabel('Índice')
        plt.ylabel('LONGTIME')
        plt.title(f'Previsão de LONGTIME vs. Valor Real {modelname}') 
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Salvando o gráfico
        prediction_filename = os.path.join(self.output_directory, f"prediction_plot_bar{modelname}.jpg")
        plt.savefig(prediction_filename)
        plt.close()  # Fechar a figura para liberar memória
        self.show_image(prediction_filename)
    
    def plot_prediction_boxplot(self, y_true, y_pred, modelname):
        # Carregar o scaler salvo
        scaler_file = os.path.join(self.base_dir, "TratamentoDeDados", "scaler.pkl")
        scaler = joblib.load(scaler_file)

        # Determinar diretório baseado no tipo de modelo (Old/New)
        model_dir_suffix = self._get_model_directory_suffix(modelname)
        self.output_directory = os.path.join(self.base_dir, model_dir_suffix, "PrevisoesDosModelos(BoxPlot)")
        os.makedirs(self.output_directory, exist_ok=True)

        # Desnormalizar os dados
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1))
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

        # Garantir que y_true e y_pred são unidimensionais
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)

        # Calcular a diferença entre previstos e reais
        difference = y_pred - y_true

        plt.figure(figsize=(10, 6))

        # Preparando os dados para o boxplot
        data_to_plot = [y_true, y_pred, difference]
        labels = ['Valor Real', 'Previsão', 'Diferença (Previsto - Real)']

        # Criando o boxplot
        plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                    boxprops=dict(facecolor='lightblue'), 
                    medianprops=dict(color='red'))

        plt.xlabel('Categorias')
        plt.ylabel('LONGTIME')
        plt.title(f'Boxplot de LONGTIME vs. Previsão - {modelname}') 
        plt.grid(True)
        plt.tight_layout()

        # Salvando o gráfico
        prediction_filename = os.path.join(self.output_directory, f"prediction_boxplot_{modelname}.jpg")
        plt.savefig(prediction_filename)
        plt.close()  # Fechar a figura para liberar memória
        self.show_image(prediction_filename)


    def plot_metrics(self, mse, rmse, modelname):
        # Determinar diretório baseado no tipo de modelo (Old/New)
        model_dir_suffix = self._get_model_directory_suffix(modelname)
        output_directory = os.path.join(self.base_dir, model_dir_suffix, "MetricaDosModelos")
        # Create the output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        plt.figure(figsize=(6, 4))
        
        # Plotting the bars with better visibility
        bars = plt.bar(['MSE', 'RMSE'], [mse, rmse], color=['skyblue', 'orange'], edgecolor='black')
        
        plt.xlabel('Métrica de Avaliação')
        plt.ylabel('Valor')
        plt.title(f'Métricas de Avaliação do Modelo {modelname}')
        
        # Adding the values on top of the bars for better visibility
        plt.text(0, mse + mse * 0.05, f'{mse:.2e}', ha='center', va='bottom', color='blue', fontsize=10, weight='bold')
        plt.text(1, rmse + rmse * 0.05, f'{rmse:.2e}', ha='center', va='bottom', color='orange', fontsize=10, weight='bold')
        
        # Ajustar os limites do eixo y para dar mais espaço
        plt.ylim(0, max(mse, rmse) * 1.2)
        
        plt.grid(axis='y')
        plt.tight_layout()

        metrics_filename = os.path.join(output_directory, f"metrics_plot_{modelname}.jpg")
        plt.savefig(metrics_filename)
        self.show_image(metrics_filename)

    def plot_metric_boxplot(self, mse_list, rmse_list, modelname):
        # Determinar diretório baseado no tipo de modelo (Old/New)
        model_dir_suffix = self._get_model_directory_suffix(modelname)
        output_directory = os.path.join(self.base_dir, model_dir_suffix, "MetricaDosModelos")
        # Create the output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        # Convert lists to numpy arrays for consistency
        mse_array = np.array(mse_list)
        rmse_array = np.array(rmse_list)

        # Prepare data for boxplot
        data = [mse_array, rmse_array]

        # Plotting boxplots
        plt.figure(figsize=(12, 6))

        plt.boxplot(data, labels=['MSE', 'RMSE'], patch_artist=True, 
                   boxprops=dict(facecolor='lightgreen', color='darkgreen'),
                    whiskerprops=dict(color='darkgreen'),
                    capprops=dict(color='darkgreen'),
                    medianprops=dict(color='green'))

        plt.xlabel('Métrica de Avaliação')
        plt.ylabel('Valor')
        plt.title(f'Boxplot das Métricas de Avaliação do Modelo {modelname}')
        
        plt.grid(axis='y')
        plt.tight_layout()
        
        metrics_filename = os.path.join(output_directory, f"boxplot_metrics_{modelname}.jpg")
        plt.savefig(metrics_filename)
        self.show_image(metrics_filename)
        plt.close() 
        self.plot_mse_progression(mse_list,modelname)
        self.plot_metrics_comparison_boxplot(modelname=modelname)

    def plot_metrics_comparison_boxplot(self, shared_csv_file="shared_model_metrics_list.csv", model_type="Old", modelname=None):
        """Plot boxplot de métricas - compatível com nova estrutura de pastas"""
        # Se modelname foi fornecido, determinar automaticamente o tipo
        if modelname:
            model_dir_suffix = self._get_model_directory_suffix(modelname)
            output_directory = os.path.join(self.base_dir, model_dir_suffix, "RelatorioDosModelos(CSV)")
            output_directory1 = os.path.join(self.base_dir, model_dir_suffix, "ComparacaoMetricas(BoxPlot)")
        else:
            # Usar model_type fornecido
            if model_type == "Old":
                output_directory = os.path.join(self.base_dir, "DadosDoPostreino/ModelosOlds/RelatorioDosModelos(CSV)")
                output_directory1 = os.path.join(self.base_dir, "DadosDoPostreino/ModelosOlds/ComparacaoMetricas(BoxPlot)")
            else:
                output_directory = os.path.join(self.base_dir, "DadosDoPostreino/ModelosNew/RelatorioDosModelos(CSV)")
                output_directory1 = os.path.join(self.base_dir, "DadosDoPostreino/ModelosNew/ComparacaoMetricas(BoxPlot)")
        
        os.makedirs(output_directory, exist_ok=True)
        os.makedirs(output_directory1, exist_ok=True)
        
        # Caminho completo do arquivo CSV
        csv_file_path = os.path.join(output_directory, shared_csv_file)
        if not os.path.isfile(csv_file_path):
            print(f"Arquivo {shared_csv_file} não encontrado.")
            return

        model_names = []
        mse_values = []
        rmse_values = []

        # Ler dados do arquivo CSV
        with open(csv_file_path, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                model_names.append(row['Model Name'])
                mse_values.append(list(map(float, row['MSE List'].split(','))))
                rmse_values.append(list(map(float, row['RMSE List'].split(','))))

        # Boxplot para MSE
        plt.figure(figsize=(12, 6))
        plt.boxplot(mse_values, labels=model_names, patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', color='darkgreen'),
                    whiskerprops=dict(color='darkgreen'),
                    capprops=dict(color='darkgreen'),
                    medianprops=dict(color='green'))

        plt.xlabel('Modelos')
        plt.ylabel('MSE')
        plt.title('Comparação de MSE entre Modelos')
        plt.grid(axis='y')
        plt.tight_layout()

        mse_filename = os.path.join(output_directory1, "boxplot_mse_comparison.jpg")
        plt.savefig(mse_filename)
        self.show_image(mse_filename)
        plt.close()

        # Boxplot para RMSE
        plt.figure(figsize=(12, 6))
        plt.boxplot(rmse_values, labels=model_names, patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', color='darkgreen'),
                    whiskerprops=dict(color='darkgreen'),
                    capprops=dict(color='darkgreen'),
                    medianprops=dict(color='green'))

        plt.xlabel('Modelos')
        plt.ylabel('RMSE')
        plt.title('Comparação de RMSE entre Modelos')
        plt.grid(axis='y')
        plt.tight_layout()

        rmse_filename = os.path.join(output_directory1, "boxplot_rmse_comparison.jpg")
        plt.savefig(rmse_filename)
        self.show_image(rmse_filename)
        plt.close()

        print(f"Gráficos de comparação de MSE e RMSE salvos com sucesso.")
        # Passar modelname se disponível
        if modelname:
            self.plot_difference_comparison_boxplot(modelname=modelname)
        else:
            self.plot_difference_comparison_boxplot(model_type=model_type)

    def plot_difference_comparison_boxplot(self, shared_csv_file="shared_model_difference_list.csv", model_type="Old", modelname=None):
        """Plot boxplot de diferenças - compatível com nova estrutura de pastas"""
        # Se modelname foi fornecido, determinar automaticamente o tipo
        if modelname:
            model_dir_suffix = self._get_model_directory_suffix(modelname)
            output_directory = os.path.join(self.base_dir, model_dir_suffix, "RelatorioDosModelos(CSV)")
        else:
            # Usar model_type fornecido
            if model_type == "Old":
                output_directory = os.path.join(self.base_dir, "DadosDoPostreino/ModelosOlds/RelatorioDosModelos(CSV)")
            else:
                output_directory = os.path.join(self.base_dir, "DadosDoPostreino/ModelosNew/RelatorioDosModelos(CSV)")
        os.makedirs(output_directory, exist_ok=True)
        
        # Caminho completo do arquivo CSV
        csv_file_path = os.path.join(output_directory, shared_csv_file)
        if not os.path.isfile(csv_file_path):
            print(f"Arquivo {shared_csv_file} não encontrado.")
            return

        model_names = []
        differences = []

        # Ler dados do arquivo CSV
        with open(csv_file_path, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                model_names.append(row['Model Name'])
                
                # Tratar ambos os formatos: antigo '[-0.06992126]' e novo '-0.06992126,0.123,...'
                diff_str = row['Difference']
                
                # Se começar com '[', é o formato antigo (array numpy como string)
                if diff_str.startswith('['):
                    # Remover todos os colchetes (abertura e fechamento)
                    diff_str = diff_str.replace('[', '').replace(']', '')
                    # Substituir espaços por vírgulas
                    diff_str = diff_str.replace(' ', ',')
                    # Remover vírgulas múltiplas
                    while ',,' in diff_str:
                        diff_str = diff_str.replace(',,', ',')
                    # Remover vírgulas no início ou fim
                    diff_str = diff_str.strip(',')
                    diff_values = [float(x.strip()) for x in diff_str.split(',') if x.strip()]
                else:
                    # Formato novo: valores separados por vírgula
                    diff_values = [float(x.strip()) for x in diff_str.split(',') if x.strip()]
                
                differences.append(diff_values)

        # Boxplot para as diferenças
        plt.figure(figsize=(12, 6))
        plt.boxplot(differences, labels=model_names, patch_artist=True,
                   boxprops=dict(facecolor='lightgreen', color='darkgreen'),
                    whiskerprops=dict(color='darkgreen'),
                    capprops=dict(color='darkgreen'),
                    medianprops=dict(color='green'))

        plt.xlabel('Modelos')
        plt.ylabel('Diferença (Previsto - Real)')
        plt.title('Comparação de Diferenças entre Modelos')
        plt.grid(axis='y')
        plt.tight_layout()

        # Usar o mesmo model_dir_suffix para salvar o gráfico
        if modelname:
            output_directory1 = os.path.join(self.base_dir, model_dir_suffix, "PrevisaoDosModelos(Diferenca)")
        else:
            if model_type == "Old":
                output_directory1 = os.path.join(self.base_dir, "DadosDoPostreino/ModelosOlds/PrevisaoDosModelos(Diferenca)")
            else:
                output_directory1 = os.path.join(self.base_dir, "DadosDoPostreino/ModelosNew/PrevisaoDosModelos(Diferenca)")
        os.makedirs(output_directory1, exist_ok=True)

        difference_filename = os.path.join(output_directory1, "boxplot_difference_comparison.jpg")
        plt.savefig(difference_filename)
        self.show_image(difference_filename)
        plt.close()

        print(f"Gráfico de comparação de diferenças salvo com sucesso.")

    @pyqtSlot(str)
    def show_image(self, filename):
        image_label = QLabel()
        pixmap = QPixmap(filename)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)  # Centraliza a imagem dentro do QLabel

        layout = QVBoxLayout()
        layout.addWidget(image_label, alignment=Qt.AlignCenter)  # Centraliza o QLabel dentro do layout
        self.main_layout.addLayout(layout)
    
    def plot_metrics_shared_old(self):
        """Exibe métricas dos modelos ANTIGOS (versão original)"""
        intput_directory = os.path.join(self.base_dir, "DadosDoPostreino/ModelosOlds/RelatorioDosModelos(CSV)")
        metrics_filename = os.path.join(intput_directory, 'shared_model_metrics.csv')
        
        # Verificar se o arquivo existe
        if not os.path.exists(metrics_filename):
            QMessageBox.warning(self, "Aviso", 
                f"Arquivo de métricas não encontrado!\n\n"
                f"Execute os modelos antigos (GRU, LSTM, RNN, TCN) primeiro.\n"
                f"Caminho esperado: {metrics_filename}")
            return
        
        try:
            # Carregar o CSV com as métricas compartilhadas
            metrics_df = pd.read_csv(metrics_filename)
            self._plot_metrics_comparison(metrics_df, "ModelosOlds", "Modelos Antigos")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar métricas dos modelos antigos:\n{str(e)}")
    
    def plot_metrics_shared_new(self):
        """Exibe métricas dos modelos NOVOS (versão corrigida)"""
        intput_directory = os.path.join(self.base_dir, "DadosDoPostreino/ModelosNew/RelatorioDosModelos(CSV)")
        metrics_filename = os.path.join(intput_directory, 'shared_model_metrics.csv')
        
        # Verificar se o arquivo existe
        if not os.path.exists(metrics_filename):
            QMessageBox.warning(self, "Aviso", 
                f"Arquivo de métricas não encontrado!\n\n"
                f"Execute os modelos corrigidos (GRU_corrigido, LSTM_corrigido, etc) primeiro.\n"
                f"Caminho esperado: {metrics_filename}")
            return
        
        try:
            # Carregar o CSV com as métricas compartilhadas
            metrics_df = pd.read_csv(metrics_filename)
            self._plot_metrics_comparison(metrics_df, "ModelosNew", "Modelos Novos (Corrigidos)")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar métricas dos modelos novos:\n{str(e)}")
    
    def _plot_metrics_comparison(self, metrics_df, folder_suffix, title_suffix):
        """Função auxiliar para plotar comparação de métricas"""

        metrics_to_plot = ['MSE', 'RMSE', 'R²']

        for metric in metrics_to_plot:
            plt.figure(figsize=(8, 6))
            
            # Obter os valores da métrica para todos os modelos
            metric_values = metrics_df[metric].values
            model_names = metrics_df['Model Name'].values
            
            # Gerar uma paleta de cores com o mesmo número de cores que o número de modelos
            colors = sns.color_palette("hsv", len(model_names))

            # Plotar o gráfico de barras com cores diferentes para cada barra
            bars = plt.bar(model_names, metric_values, color=colors, edgecolor='black')
            plt.xlabel('Modelos')
            plt.ylabel('Valor')
            plt.title(f'Comparação de {metric} - {title_suffix}')

            # Adicionar os valores no topo das barras
            for i, value in enumerate(metric_values):
                plt.text(i, value + value * 0.05, f'{value:.2e}', ha='center', va='bottom', color='black', fontsize=10, weight='bold')
            
            # Ajustar os limites do eixo y para dar mais espaço
            plt.ylim(0, max(metric_values) * 1.2)

            plt.xticks(rotation=45, ha='right')  # Rotacionar os nomes dos modelos para melhor leitura
            plt.grid(axis='y')
            plt.tight_layout()

            # Salvar em pasta específica para o tipo de modelo dentro de DadosDoPostreino
            output_directory = os.path.join(self.base_dir, f"DadosDoPostreino/{folder_suffix}/MetricaDosModelos")
            os.makedirs(output_directory, exist_ok=True)

            # Salvar a imagem do gráfico
            metrics_filename = os.path.join(output_directory, f"{metric}_comparison_plot.jpg")
            plt.savefig(metrics_filename)
            self.show_image(metrics_filename)
    
    def plot_mse_progression(self,mse_values,model_name):
        # Determinar diretório baseado no tipo de modelo (Old/New)
        model_dir_suffix = self._get_model_directory_suffix(model_name)
        output_directory = os.path.join(self.base_dir, model_dir_suffix, "MetricaDosModelos")
        # Create the output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        plt.figure(figsize=(10, 6))
        
        # Ordenar os valores de MSE para criar uma linha de tendência que tende a zero
        mse_values_sorted = sorted(mse_values, reverse=True)
        epochs = list(range(1, len(mse_values_sorted) + 1))
        
        plt.plot(epochs, mse_values_sorted, marker='o', linestyle='-', color='b', label='MSE')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title(f'Progressão do MSE Durante o Treinamento {model_name}')
        plt.yscale('log')  # Usar escala logarítmica para melhor visualização se o MSE variar em ordens de magnitude
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        metrics_filename = os.path.join(output_directory, f"mse_progression_plot_{model_name}.jpg")
        plt.savefig(metrics_filename)
        self.show_image(metrics_filename)

    @pyqtSlot(float, float)
    def update_metrics_chart(self, mse, rmse):
        print("Atualizando o gráfico de métricas")
        self.plot_metrics(mse, rmse)

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def update_prediction_chartCNN(self, y_true, y_pred, dates):
        print("Atualizando o gráfico")
        self.plot_predictionCNN(y_true, y_pred, dates)

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def update_prediction_chart(self, y_true, y_pred, dates):
        print("Atualizando o gráfico")
        self.plot_prediction(y_true, y_pred, dates)

    @pyqtSlot(np.ndarray, np.ndarray, str)
    def update_prediction_boxplot(self,mse_list, rmse_list, modelname):
        print("Atualizando o gráfico")
        self.plot_metric_boxplot(mse_list, rmse_list, modelname)
