import re
import matplotlib.pyplot as plt
import io
import os
import logging
from flask import send_file,request,jsonify # type: ignore

# Definir diretório base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuração do diretório de logs
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Configuração do logging
logging.basicConfig(filename=os.path.join(LOG_DIR, 'packet_logs.txt'),
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s')

class Logger:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.pattern_ataque = re.compile(r"Ataque detectado")
        self.pattern_normal = re.compile(r"Requisição normal")
        self.pattern_inconclusivo = re.compile(r"Pacote não TCP")

    def log(self, message):
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(f"{message}\n")

    def process_logs(self, filtro):
        # Variáveis para contadores de packet_logs.txt
        ataque_detectado = 0
        requisicao_normal = 0
        inconclusivo = 0

        # Variáveis para armazenar dados de predictions_log.txt
        normalized_predictions = []
        desnormalized_predictions = []
        resultados = []
        
        # Variáveis para armazenar logs
        logs = []

        # Processar o arquivo packet_logs.txt
        with open(self.log_file_path, 'r') as file:
            lines = file.readlines()

        # Aplicar o filtro no arquivo packet_logs.txt
        if filtro == 'recentes':
            lines = lines[-20:]  # Pegar as 20 linhas mais recentes
        elif filtro == 'antigos':
            lines = lines[:20]  # Pegar as 20 primeiras linhas
        # No caso de 'todos', não filtramos (padrão)

        # Processar as linhas filtradas
        for line in lines:
            timestamp = line.split(' - ')[0]  # Extrair timestamp
            if self.pattern_ataque.search(line):
                ataque_detectado += 1
                logs.append({
                    'timestamp': timestamp,
                    'source_ip': 'IP de origem não disponível',  # Defina isso se disponível
                    'destination_ip': 'IP de destino não disponível',  # Defina isso se disponível
                    'tipo': 'Ataque'
                })
            elif self.pattern_normal.search(line):
                requisicao_normal += 1
                # Extrair IPs de origem e destino
                match = re.search(r'Pacote: Ether / IP / TCP ([^ ]+) > ([^ ]+) ', line)
                if match:
                    source_ip = match.group(1)
                    destination_ip = match.group(2)
                    logs.append({
                        'timestamp': timestamp,
                        'source_ip': source_ip,
                        'destination_ip': destination_ip,
                        'tipo': 'Permitido'
                    })
            elif self.pattern_inconclusivo.search(line):
                inconclusivo += 1
                logs.append({
                    'timestamp': timestamp,
                    'tipo': 'Inconclusivo'
                })  # Adicione um log para inconclusivos

        # Processar o arquivo predictions_log.txt
        pred_log_path = os.path.join(LOG_DIR, 'predictions_log.txt')
        if not os.path.exists(pred_log_path):
            # Se não existe, retornar listas vazias
            normalized_predictions = []
            desnormalized_predictions = []
            resultados = []
            pred_lines = []
        else:
            with open(pred_log_path, 'r') as file:
                pred_lines = file.readlines()

            # Aplicar o filtro no arquivo predictions_log.txt
            if filtro == 'recentes':
                pred_lines = pred_lines[-10:]  # Pegar as 10 predições mais recentes
            elif filtro == 'antigos':
                pred_lines = pred_lines[:10]  # Pegar as 10 primeiras predições
            # No caso de 'todos', não filtramos (padrão)

            # Processar as predições filtradas
            for line in pred_lines:
                try:
                    normalized_prediction, desnormalized_prediction, resultado = line.strip().split(',')
                    normalized_predictions.append(float(normalized_prediction))
                    desnormalized_predictions.append(float(desnormalized_prediction))
                    resultados.append(resultado)
                except ValueError:
                    continue  # Pular linhas mal formatadas

        # Criar um dicionário com os dados coletados de packet_logs.txt e predictions_log.txt
        data = {
            'packet_logs': {
                'ataques_detectados': ataque_detectado,
                'requisicoes_permitidas': requisicao_normal,
                'inconclusivos': inconclusivo,
                'logs': logs  # Inclua os logs capturados aqui
            },
            'predictions_log': {
                'normalized_predictions': normalized_predictions,
                'desnormalized_predictions': desnormalized_predictions,
                'resultados': resultados
            }
        }

        # Retornar os dados como JSON
        return data

    def generate_prediction_chart(self):
        # Listas para armazenar os dados
        normalized_predictions = []
        desnormalized_predictions = []
        resultados = []

        # Caminho do arquivo de log de predições
        log_file_path = os.path.join(LOG_DIR, 'predictions_log.txt')

        # Verificar se o arquivo existe
        if not os.path.exists(log_file_path):
            # Retornar uma imagem vazia ou mensagem
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Nenhuma predição registrada ainda', 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            return send_file(buf, mimetype='image/png')

        # Ler as últimas 10 linhas do arquivo de log
        with open(log_file_path, 'r') as file:
            # Pegar as últimas 10 linhas do arquivo
            lines = file.readlines()[-10:]

            # Processar as últimas 10 linhas
            for line in lines:
                normalized_prediction, desnormalized_prediction, resultado = line.strip().split(',')
                normalized_predictions.append(float(normalized_prediction))
                desnormalized_predictions.append(float(desnormalized_prediction))
                resultados.append(resultado)

        # Criar o gráfico
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Gráfico da predição normalizada
        ax1.plot(normalized_predictions, label="Predição Normalizada", marker='o', color='b')
        ax1.set_xlabel('Pacote (Últimos 10)')
        ax1.set_ylabel('Predição Normalizada', color='b')

        # Gráfico do tempo desnormalizado
        ax2 = ax1.twinx()  # Criar um eixo y secundário
        ax2.plot(desnormalized_predictions, label="Tempo Desnormalizado", marker='s', color='g')
        ax2.set_ylabel('Tempo Desnormalizado', color='g')

        # Marcadores para resultados de ataque/permitido
        for i, resultado in enumerate(resultados):
            color = 'r' if resultado == 'Ataque' else 'g'
            ax1.scatter(i, normalized_predictions[i], color=color, s=100, label=f'{resultado}' if i == 0 else "")
            ax2.scatter(i, desnormalized_predictions[i], color=color, s=100)

        # Exibir as grades
        ax1.grid(True)
        ax2.grid(False)  # Se você quiser grades no segundo eixo, mude para True

        # Adicionar legendas
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Configurações do gráfico
        plt.title('Predições Normalizadas e Tempos Desnormalizados (Últimas 10 Requisições)')
        fig.tight_layout()

        # Salvar o gráfico em um buffer de memória
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Fechar a figura para liberar memória
        plt.close()

        # Enviar a imagem gerada como resposta
        return send_file(buf, mimetype='image/png')

    def generate_log_chart(self):
        # Contadores para os tipos de requisições
        ataque_detectado = 0
        requisicao_normal = 0
        inconclusivo = 0

        # Padrões de detecção
        pattern_ataque = re.compile(r"Ataque detectado")
        pattern_normal = re.compile(r"Requisição normal")
        pattern_inconclusivo = re.compile(r"Pacote não TCP")

        # Processar o arquivo de log
        log_file_path = self.log_file_path
        
        # Verificar se o arquivo existe
        if not os.path.exists(log_file_path):
            # Retornar gráfico vazio
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Nenhum log registrado ainda', 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            return send_file(buf, mimetype='image/png')
        
        with open(log_file_path, 'r') as file:
            for line in file:
                if pattern_ataque.search(line):
                    ataque_detectado += 1
                elif pattern_normal.search(line):
                    requisicao_normal += 1
                elif pattern_inconclusivo.search(line):
                    inconclusivo += 1

        # Criar o gráfico
        labels = ['Ataques Detectados', 'Requisições Permitidas', 'Inconclusivos']
        values = [ataque_detectado, requisicao_normal, inconclusivo]

        plt.figure(figsize=(8, 6))
        
        # Criar o gráfico de barras
        bars = plt.bar(labels, values, color=['red', 'green', 'gray'])
        
        # Adicionar título e rótulo do eixo Y
        plt.title('Análise de Requisições de Rede')
        plt.ylabel('Quantidade')

        # Exibir as grades verticais
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Adicionar a legenda para explicar as cores
        plt.legend(bars, ['Ataques Detectados', 'Requisições Permitidas', 'Inconclusivos'], loc='upper right')

        # Salvar o gráfico em um buffer de memória
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Fechar a figura para liberar memória
        plt.close()

        # Enviar a imagem gerada como resposta
        return send_file(buf, mimetype='image/png')
