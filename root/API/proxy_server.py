import pandas as pd
from flask import Flask, request, jsonify, send_file # type: ignore
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from scapy.all import sniff, TCP # type: ignore
import threading
import requests
import json
import joblib
from flask_cors import CORS # type: ignore
import logging
import matplotlib.pyplot as plt
import re
import io
app = Flask(__name__)
CORS(app)  # Permitir solicitações de outros domínios

# Configurar o logging
logging.basicConfig(filename='packet_logs.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Carregar o modelo
model = load_model('gru_model.h5')

# Carregar o scaler salvo
scaler = joblib.load('scaler.pkl')

# Verificar a forma de entrada esperada pelo modelo
input_shape = model.input_shape
print(f"Forma de entrada esperada pelo modelo: {input_shape}")

@app.route('/gerar_dados', methods=['GET'])
def gerar_dados():
    # Obter o filtro solicitado do front-end (padrão: "todos")
    filtro = request.args.get('filter', 'todos')

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

    # Padrões de detecção para packet_logs.txt
    pattern_ataque = re.compile(r"Ataque detectado")
    pattern_normal = re.compile(r"Requisição normal")
    pattern_inconclusivo = re.compile(r"Pacote não TCP")

    # Processar o arquivo packet_logs.txt
    log_file_path = 'packet_logs.txt'  # Supondo que esse arquivo exista
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    # Aplicar o filtro no arquivo packet_logs.txt
    if filtro == 'recentes':
        lines = lines[-20:]  # Pegar as 10 linhas mais recentes
    elif filtro == 'antigos':
        lines = lines[:20]  # Pegar as 10 primeiras linhas
    # No caso de 'todos', não filtramos (padrão)

    # Processar as linhas filtradas
    for line in lines:
        timestamp = line.split(' - ')[0]  # Extrair timestamp
        if pattern_ataque.search(line):
            ataque_detectado += 1
            logs.append({
                'timestamp': timestamp,
                'source_ip': 'IP de origem não disponível',  # Defina isso se disponível
                'destination_ip': 'IP de destino não disponível',  # Defina isso se disponível
                'tipo': 'Ataque'
            })
        elif pattern_normal.search(line):
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
        elif pattern_inconclusivo.search(line):
            inconclusivo += 1
            # Adicione um log para inconclusivos se necessário

    # Processar o arquivo predictions_log.txt
    pred_log_path = 'predictions_log.txt'  # Supondo que esse arquivo exista
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
    return jsonify(data)


@app.route('/prediction_chart', methods=['GET'])
def prediction_chart():
    # Listas para armazenar os dados
    normalized_predictions = []
    desnormalized_predictions = []
    resultados = []

    # Caminho do arquivo de log de predições
    log_file_path = 'predictions_log.txt'

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





@app.route('/log_chart', methods=['GET'])
def log_chart():
    # Contadores para os tipos de requisições
    ataque_detectado = 0
    requisicao_normal = 0
    inconclusivo = 0

    # Padrões de detecção
    pattern_ataque = re.compile(r"Ataque detectado")
    pattern_normal = re.compile(r"Requisição normal")
    pattern_inconclusivo = re.compile(r"Pacote não TCP")

    # Processar o arquivo de log
    log_file_path = 'packet_logs.txt'  # O caminho onde o arquivo de log está armazenado no servidor
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

    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Receber dados em formato JSON
    print(f"Dados recebidos: {data}")  # Log para verificar os dados recebidos

    X = np.array(data['features'])  # Extrair recursos

    # Ajustar a forma dos dados de entrada conforme esperado pelo modelo
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=0)
    if len(X.shape) == 2:
        X = np.expand_dims(X, axis=2)

    print(f"Forma dos dados após ajuste: {X.shape}")  # Log para verificar a forma dos dados

    # Normalizar os dados
    #X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # Garantir que a entrada tenha 8 passos de tempo e 1 característica por passo
    if X.shape[1] != 8 or X.shape[2] != 1:
        error_message = f"Dados de entrada com formato incorreto. Esperado: {input_shape}, Recebido: {X.shape}"
        print(error_message)  # Log para erro de formato
        return jsonify({'error': error_message}), 400

    # Fazer a previsão
    prediction = model.predict(X)
    return jsonify({'prediction': prediction.tolist()})



# Função para determinar se um pacote é um ataque
def is_attack(features):
    # Ajustar a forma dos dados de entrada conforme esperado pelo modelo
    X = np.array(features)
    X = np.expand_dims(X, axis=0)  # Adicionar dimensão de batch_size

    # Normalizar os dados com o scaler antes de enviar para a API Flask
    normalized_features = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # Converter o array numpy em uma lista antes de enviar
    normalized_features = normalized_features.tolist()

    # Enviar os recursos para a API Flask para predição
    url = 'http://localhost:5000/predict'
    data = {'features': normalized_features}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    # Adicione logs para depuração
    print(f"Resposta da API: {response.json()}")  # Imprime a resposta da API

    if 'prediction' not in response.json():
        print(f"Erro na resposta: {response.json()}")
        return False

    # Obter a previsão normalizada
    normalized_prediction = response.json()['prediction'][0][0]
    print(f"Predição normalizada: {normalized_prediction}")

    # Desnormalizar a previsão
    desnormalized_prediction = scaler.inverse_transform([[normalized_prediction]])[0][0]
    print("Tempo previsto desnormalizado", desnormalized_prediction)
    
     

    # Definir o threshold desnormalizado
    threshold = 200.0  # Ajuste conforme necessário para sua aplicação
    
    # Salvar as informações em um arquivo
    with open('predictions_log.txt', 'a') as log_file:
        log_file.write(f"{normalized_prediction},{desnormalized_prediction},{'Ataque' if desnormalized_prediction < threshold else 'Permitido'}\n")
    return desnormalized_prediction < threshold




# hping3  

# Função de callback para processar cada pacote capturado
def process_packet(packet):
    print("Pacote capturado")
    if packet.haslayer(TCP):
        features = extract_features(packet)
        if is_attack(features):
            log_message = f"Ataque detectado! Requisição negada. Pacote: {packet.summary()}"
            print(log_message)
            logging.info(log_message)
        else:
            log_message = f"Requisição normal. Permitida. Pacote: {packet.summary()}"
            print(log_message)
            logging.info(log_message)
    else:
        log_message = f"Pacote não TCP: {packet.summary()}"
        print(log_message)
        logging.info(log_message)

# Função para extrair recursos dos pacotes
def extract_features(packet):
    # Capturar o timestamp do pacote (para referência)
    timestamp = packet.time
    
    
    # Gerar características aleatórias variando de 100 a 1000
    features = np.random.randint(30, 100, size=(8,))
    
    # Transformar a lista em um array com a forma (8, 1)
    features = features.reshape((8, 1))
    print(features)
    
    return features

# Função para capturar pacotes em todas as interfaces
def start_sniffing():
    print("Captura iniciada")
    sniff(prn=process_packet, iface='lo')

# Iniciar o servidor Flask em uma thread separada
def start_flask():
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    # Iniciar a captura de pacotes em uma thread separada
    sniffing_thread = threading.Thread(target=start_sniffing)
    sniffing_thread.start()

    # Iniciar o servidor Flask
    start_flask()

