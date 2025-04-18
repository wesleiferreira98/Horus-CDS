from PyQt5.QtWidgets import (
    QWidget, QTableView, QPushButton, QFileDialog, QVBoxLayout, 
    QScrollArea, QLabel, QTableWidgetItem, QMessageBox, QProgressBar,
    QRadioButton, QTableWidget, QHBoxLayout, QSpacerItem, QSizePolicy
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, Qt, QThreadPool, QTimer, pyqtSlot
import numpy as np
import pandas as pd
from Models.PandasModel import PandasModel
from View.CustomMessageBox import CustomMessageBox
from View.LoadDataThread import LoadDataThread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from ThreadTrain import (
    TrainingThreadGRU, TrainingThreadLSTM, TrainingThreadARIMA, 
    TrainingThreadTCN, TrainingThreadKNN, TrainingThreadRNN, 
    TrainingThreadRandomForest
)
import os
import joblib
import seaborn as sns
import csv
from View.CaptureLog import CaptureLog

class SPTI(QWidget):
    
    update_prediction_chart = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, str)
    update_metrics_chart = pyqtSignal(float, float,str)
    update_progress = pyqtSignal(float)
    update_metrics_chart_boxplot = pyqtSignal(np.ndarray, np.ndarray,str)
    update_prediction_chartCNN = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        self.initUI()
        self.processed_data = False
        self.threadpool = QThreadPool()
        self.model_select = None

    def initUI(self):
        self.setWindowTitle('Sistema SPTI')
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

        self.train_button = QPushButton('Selecionar Data SET')
        self.train_model_button = QPushButton('Iniciar Treinamento')
        self.model_select_train = QLabel("Selecione o Modelo de treinamento")
        self.grafic_models_button = QPushButton('Obter dados do ultimo treinamento')
        self.grafic_logs = QPushButton('Capturar dados de Log') 
        self.label_train_test = QLabel("Dados de Treino")
        self.label_progess_test = QLabel("Progresso do Teste")

        # Organizing radio buttons and labels
        rb_train_layoutV1 = QVBoxLayout()
        rb_train_layoutV1.addWidget(self.rb_RandomForest)
        rb_train_layoutV1.addWidget(self.rb_KNN)
        rb_train_layoutV1.addWidget(self.rb_TCN)

        rb_train_layoutV2 = QVBoxLayout()
        rb_train_layoutV2.addWidget(self.rb_ARIMA)
        rb_train_layoutV2.addWidget(self.rb_LSTM)
        rb_train_layoutV2.addWidget(self.rb_RNN)
        rb_train_layoutV2.addWidget(self.rb_GRU)

        rb_train_layout = QHBoxLayout()
        rb_train_layout.addLayout(rb_train_layoutV1)
        rb_train_layout.addLayout(rb_train_layoutV2)

        integration_rb_label = QVBoxLayout()
        integration_rb_label.addWidget(self.model_select_train)
        integration_rb_label.addLayout(rb_train_layout)

        label_train_layout = QHBoxLayout()
        label_train_layout.addWidget(self.label_train_test)
        label_train_layout.setAlignment(Qt.AlignCenter)

        label_progress_layout = QHBoxLayout()
        label_progress_layout.addWidget(self.label_progess_test)
        label_progress_layout.setAlignment(Qt.AlignCenter)

        self.progress_bar = QProgressBar()

       

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
        self.main_layout.addWidget(self.grafic_models_button)
        self.main_layout.addWidget(self.grafic_logs)
        self.main_layout.addLayout(integration_rb_label)
        self.main_layout.addLayout(label_progress_layout)
        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addLayout(label_train_layout)
        self.main_layout.addLayout(table_layout)

        # Connecting signals to slots
        self.train_button.clicked.connect(self.select_train_data)
        self.train_model_button.clicked.connect(self.start_training)
        self.grafic_models_button.clicked.connect(self.plot_metrics_shared)
        self.grafic_logs.clicked.connect(self.cap_log)
        self.rb_RandomForest.toggled.connect(self.on_radio_button_toggled)
        self.rb_KNN.toggled.connect(self.on_radio_button_toggled)
        self.rb_LSTM.toggled.connect(self.on_radio_button_toggled)
        self.rb_TCN.toggled.connect(self.on_radio_button_toggled)
        self.rb_RNN.toggled.connect(self.on_radio_button_toggled)
        self.rb_GRU.toggled.connect(self.on_radio_button_toggled)
        self.rb_ARIMA.toggled.connect(self.on_radio_button_toggled)

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

       
    def cap_log(self):
        # Instancia e exibe a janela CaptureLog
        self.capture = CaptureLog()  # Salva como um atributo da classe para manter a referência
        self.capture.show()
    def select_train_data(self):
        file_dialog = QFileDialog()
        train_data_file, _ = file_dialog.getOpenFileName(self, 'Selecionar Data SET', '', 'CSV Files (*.csv)')
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

        QMessageBox.information(self, "Treinamento", f"Treinamento iniciado com o modelo {selected_model}.")
        self.train_model(selected_model)

    def on_radio_button_toggled(self):
        selected_model = self.get_selected_model()
        if selected_model:
            self.model_select_train.setText(f"Treinar com Modelo {selected_model}")

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
        return None

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
        else:
            QMessageBox.critical(self, "Erro", "Modelo de treinamento inválido.")
            return

        self.train_thread.update_progress.connect(self.update_progress)
        self.train_thread.update_prediction_chart.connect(self.plot_prediction)
        self.train_thread.update_metrics_chart.connect(self.plot_metrics)
        self.train_thread.update_metrics_chart_boxplot.connect(self.update_prediction_boxplot)
        self.train_thread.start()



    def plot_prediction(self, y_true, y_pred, dates, modelname):
        # Carregar o scaler salvo
        scaler_file = "./TratamentoDeDados/scaler.pkl"
        scaler = joblib.load(scaler_file)

        # Convertendo as datas para o formato datetime
        dates = pd.to_datetime(dates)

        self.output_directory = "./PrevisoesDosModelos"
        os.makedirs(self.output_directory, exist_ok=True)
        
        # Usar apenas os 20 primeiros dados
        y_true = y_true[:20]
        y_pred = y_pred[:20]
        dates = dates[:20]

        # Desnormalizar os dados
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1))
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)
        dates = np.ravel(dates)

        df = pd.DataFrame({
            'dates': dates,
            'y_true': y_true,
            'y_pred': y_pred
        })

        df_sorted = df.sort_values(by='y_true', ascending=False)
        dates_sorted = df_sorted['dates']
        y_true_sorted = df_sorted['y_true']
        y_pred_sorted = df_sorted['y_pred']

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
        plt.close()
        self.show_image(prediction_filename)
        self.plot_prediction_block(y_true, y_pred, modelname)
        self.plot_prediction_boxplot(y_true, y_pred, modelname)



    def plot_prediction_block(self, y_true, y_pred, modelname):
        # Carregar o scaler salvo
        scaler_file = "./TratamentoDeDados/scaler.pkl"
        scaler = joblib.load(scaler_file)

        self.output_directory = "./PrevisoesDosModelos(Block)"
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
        scaler_file = os.path.join("./TratamentoDeDados", "scaler.pkl")
        scaler = joblib.load(scaler_file)

        self.output_directory = "./PrevisoesDosModelos(BoxPlot)"
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
        output_directory = "./MetricaDosModelos"
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
        output_directory = "./MetricaDosModelos"
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
        self.plot_metrics_comparison_boxplot()

    def plot_metrics_comparison_boxplot(self, shared_csv_file="shared_model_metrics_list.csv"):
        output_directory = "./RelatorioDosModelos(CSV)"
        os.makedirs(output_directory, exist_ok=True)

        output_directory1 = "./ComparacaoMetricas(BoxPlot)"
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
        self.plot_difference_comparison_boxplot()

    def plot_difference_comparison_boxplot(self, shared_csv_file="shared_model_difference_list.csv"):
        output_directory = "./RelatorioDosModelos(CSV)"
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
                differences.append(list(map(float, row['Difference'].split(','))))

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

        output_directory1 = "./PrevisaoDosModelos(Diferenca)"
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
    
    def plot_metrics_shared(self):
        intput_directory = "./RelatorioDosModelos(CSV)"
        # Create the output directory if it doesn't exist
        metrics_filename = os.path.join(intput_directory, 'shared_model_metrics.csv')
        os.makedirs(intput_directory, exist_ok=True)

        # Carregar o CSV com as métricas compartilhadas
        metrics_df = pd.read_csv(metrics_filename)

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
            plt.title(f'Comparação de {metric} entre Modelos')

            # Adicionar os valores no topo das barras
            for i, value in enumerate(metric_values):
                plt.text(i, value + value * 0.05, f'{value:.2e}', ha='center', va='bottom', color='black', fontsize=10, weight='bold')
            
            # Ajustar os limites do eixo y para dar mais espaço
            plt.ylim(0, max(metric_values) * 1.2)

            plt.xticks(rotation=45, ha='right')  # Rotacionar os nomes dos modelos para melhor leitura
            plt.grid(axis='y')
            plt.tight_layout()

            output_directory = "./MetricaDosModelos"
            os.makedirs(output_directory, exist_ok=True)

            # Salvar a imagem do gráfico
            metrics_filename = os.path.join(output_directory, f"{metric}_comparison_plot.jpg")
            plt.savefig(metrics_filename)
            self.show_image(metrics_filename)
    
    def plot_mse_progression(self,mse_values,model_name):

        output_directory = "./MetricaDosModelos"
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