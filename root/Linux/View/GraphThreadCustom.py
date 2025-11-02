import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal
import joblib
import seaborn as sns
class GraphThreadCustom(QThread):
    # Sinal para transmitir o caminho do arquivo gerado
    show_image = pyqtSignal(str)

    def __init__(self, y_true, y_pred, dates, modelname,graphtype, mse=0, rmse=0,parent=None,mse_list=[], rmse_list=[]):
        super().__init__(parent)
        self.y_true = y_true
        self.y_pred = y_pred
        self.dates = dates
        self.modelname = modelname
        self.graphtype = graphtype
        self.mse = mse
        self.rmse = rmse
        self.mse_list = mse_list
        self.rmse_list=rmse_list
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def _get_model_directory_suffix(self, modelname):
        """Determina o sufixo do diretório baseado no nome do modelo"""
        if "Corrigido" in modelname or "corrigido" in modelname:
            return "DadosDoPostreino/ModelosNew"
        else:
            return "DadosDoPostreino/ModelosOlds"
    
    def run(self):
        try:
            if(self.graphtype == 1):
                self.plot_prediction(self.y_true, self.y_pred, self.dates, self.modelname)
            elif(self.graphtype == 2):
                self.plot_metrics(self.mse, self.rmse, self.modelname)
            elif(self.graphtype == 3):
                self.plot_metric_boxplot(self.mse_list, self.rmse_list, self.modelname)
            elif(self.graphtype == 4):
                self.plot_metrics_shared()
        except Exception as e:
            print(f"An error occurred: {e}")

            
    def plot_prediction(self, y_true, y_pred, dates, modelname):
        # Carregar o scaler salvo
        scaler_file = os.path.join("TratamentoDeDados", "scaler.pkl")
        scaler = joblib.load(scaler_file)

        # Convertendo as datas para o formato datetime
        dates = pd.to_datetime(dates)
        all_y_true = scaler.inverse_transform(y_true.reshape(-1, 1))
        all_y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

        self.output_directory = "PrevisoesDosModelos"
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)
        
        # Usar apenas os 20 primeiros dados
        y_true = y_true[:20]
        y_pred = y_pred[:20]
        dates = dates[:20]

        # Desnormalizar os dados
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1))
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

        print(y_true.shape)
        print(y_pred.shape)
        print(dates.shape)

        # Garantir que y_true, y_pred e dates são unidimensionais
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)
        dates = np.ravel(dates)

        # Criar um DataFrame para facilitar a ordenação
        df = pd.DataFrame({
            'dates': dates,
            'y_true': y_true,
            'y_pred': y_pred
        })

        # Ordenar o DataFrame com base nos valores reais (ou previsões) do maior para o menor
        df_sorted = df.sort_values(by='y_true', ascending=False)

        # Atualizar variáveis com os dados ordenados
        dates_sorted = df_sorted['dates']
        y_true_sorted = df_sorted['y_true']
        y_pred_sorted = df_sorted['y_pred']

        plt.figure(figsize=(10, 6))
        
        # Convertendo as datas para strings apenas com a data (sem hora)
        date_strings_sorted = [str(date.date()) for date in dates_sorted]
        
        # Plotando os pontos reais e previstos
        plt.scatter(date_strings_sorted, y_true_sorted, label='Valor Real', color='blue', s=500)  # Aumentar o tamanho dos pontos
        plt.scatter(date_strings_sorted, y_pred_sorted, label='Previsão', color='red', s=300)

        plt.xlabel('Data')
        plt.ylabel('LONGTIME')
        plt.title(f'Previsão de LONGTIME vs. Valor Real {modelname}') 
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45, ha='right')  # Rodar as etiquetas do eixo x para melhor legibilidade
        plt.tight_layout()

        # Salvando o gráfico
        prediction_filename = os.path.join(self.output_directory, f"prediction_plot_{modelname}.jpg")
        plt.savefig(prediction_filename)
        plt.close()  # Fechar a figura para liberar memória
        self.show_image.emit(prediction_filename)
        self.plot_prediction_block(y_true, y_pred,modelname)
        self.plot_prediction_boxplot(y_true, y_pred,modelname)

    def plot_prediction_block(self, y_true, y_pred, modelname):
        # Carregar o scaler salvo
        scaler_file = os.path.join("TratamentoDeDados", "scaler.pkl")
        scaler = joblib.load(scaler_file)

        self.output_directory = "PrevisoesDosModelos"
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
        self.show_image.emit(prediction_filename)
    
    def plot_prediction_boxplot(self, y_true, y_pred, modelname):
        # Carregar o scaler salvo
        scaler_file = os.path.join("TratamentoDeDados", "scaler.pkl")
        scaler = joblib.load(scaler_file)

        self.output_directory = "PrevisoesDosModelos"
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
        self.show_image.emit(prediction_filename)

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
        self.show_image.emit(metrics_filename)

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
                    boxprops=dict(facecolor='lightblue', color='skyblue'),
                    whiskerprops=dict(color='skyblue'),
                    capprops=dict(color='skyblue'),
                    medianprops=dict(color='blue'))

        plt.xlabel('Métrica de Avaliação')
        plt.ylabel('Valor')
        plt.title(f'Boxplot das Métricas de Avaliação do Modelo {modelname}')
        
        plt.grid(axis='y')
        plt.tight_layout()
        
        metrics_filename = os.path.join(output_directory, f"boxplot_metrics_{modelname}.jpg")
        plt.savefig(metrics_filename)
        self.show_image.emit(metrics_filename)
        plt.close() 
        self.plot_mse_progression(mse_list,modelname)

    def plot_metrics_shared(self):
        intput_directory = "RelatorioDosModelos"
        # Create the output directory if it doesn't exist
        metrics_filename = os.path.join(intput_directory, 'shared_model_metrics.csv')
        os.makedirs(intput_directory, exist_ok=True)

        # Carregar o CSV com as métricas compartilhadas
        metrics_df = pd.read_csv(metrics_filename)

        metrics_to_plot = ['MSE', 'RMSE', 'R²']

        for metric in metrics_to_plot:
            plt.figure(figsize=(8, 8))
            
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

            # Determinar diretório baseado no tipo de modelo (Old/New)
            model_dir_suffix = self._get_model_directory_suffix(self.modelname)
            output_directory = os.path.join(self.base_dir, model_dir_suffix, "MetricaDosModelos")
            os.makedirs(output_directory, exist_ok=True)

            # Salvar a imagem do gráfico
            metrics_filename = os.path.join(output_directory, f"{metric}_comparison_plot.jpg")
            plt.savefig(metrics_filename)
            self.show_image.emit(metrics_filename)
    
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
        self.show_image.emit(metrics_filename)
