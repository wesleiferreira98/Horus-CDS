from flask import Flask, request, jsonify  # type: ignore
from flask_cors import CORS  # type: ignore
from packet_sniffer import PacketSniffer  
from prediction import Prediction  
from logger import Logger, log_packet  
import threading
import os

app = Flask(__name__)
CORS(app)

# Inicializando o logger
LOG_DIR = './API/logs'
os.makedirs(LOG_DIR, exist_ok=True)
logger = Logger('logs/packet_logs.txt')  

# Instâncias das classes
prediction_model = Prediction('./API/models/gru_model.h5', './API/models/scaler.pkl')
packet_sniffer = PacketSniffer(prediction_model, logger)

@app.route('/gerar_dados', methods=['GET'])
def gerar_dados():
    filtro = request.args.get('filter', 'todos')
    data = packet_sniffer.process_logs(filtro)
    return jsonify(data)

@app.route('/prediction_chart', methods=['GET'])
def prediction_chart():
    return packet_sniffer.generate_prediction_chart()

@app.route('/log_chart', methods=['GET'])
def log_chart():
    return packet_sniffer.generate_log_chart()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    return prediction_model.predict(data['features'])

if __name__ == '__main__':
    # Iniciar a captura de pacotes em uma thread separada
    sniffing_thread = threading.Thread(target=packet_sniffer.start_sniffing)
    sniffing_thread.start()

    # Iniciar o servidor Flask
    app.run(host='0.0.0.0', port=5000)
