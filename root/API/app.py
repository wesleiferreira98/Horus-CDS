from flask import Flask, request, jsonify  # type: ignore
from flask_cors import CORS  # type: ignore
from packet_sniffer import PacketSniffer  
from prediction import Prediction  
from logger import Logger
import threading
import os

app = Flask(__name__)
CORS(app)

# Inicializando o logger
LOG_DIR = './logs'
MODELS_DIR = './models'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
# Inicialize o modelo padrão
active_model_name = "Horus-CDS V4 (TCN)"
model_paths = {
    "Horus-CDS V1 (RNN)": "./models/rnn_model.h5",
    "Horus-CDS V2 (LSTM)": "./models/lstm_model.h5",
    "Horus-CDS V3 (GRU)": "./models/gru_model.h5",
    "Horus-CDS V4 (TCN)": "./models/tcn_model.h5",
}
log_directories = {
    "Horus-CDS V1 (RNN)": "./logs/rnn_logs/",
    "Horus-CDS V2 (LSTM)": "./logs/lstm_logs/",
    "Horus-CDS V3 (GRU)": "./logs/gru_logs/",
    "Horus-CDS V4 (TCN)": "./logs/tcn_logs/",
}

# Inicialize o logger e o modelo ativo
os.makedirs(log_directories[active_model_name], exist_ok=True)
logger = Logger(f"{log_directories[active_model_name]}packet_logs.txt")
prediction_model = Prediction(model_paths[active_model_name], './models/scaler.pkl')

# Instâncias das classes
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
    try:
        result = prediction_model.predict(data['features'])
        if 'error' in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/set_model', methods=['POST'])
def set_model():
    global active_model_name, prediction_model, logger

    data = request.json
    selected_model = data.get("model")

    if not selected_model:
        return jsonify({"error": "Nenhum modelo foi especificado."}), 400

    if selected_model not in model_paths:
        return jsonify({"error": f"Modelo '{selected_model}' não encontrado. Escolha entre: {', '.join(model_paths.keys())}"}), 400

    try:
        # Criar nova instância antes do lock
        new_model = Prediction(model_paths[selected_model], './models/scaler.pkl')
        new_logger = Logger(f"{log_directories[selected_model]}packet_logs.txt")

        with packet_sniffer.lock:
            active_model_name = selected_model
            prediction_model = new_model
            logger = new_logger
            packet_sniffer.prediction_model = new_model
            packet_sniffer.logger = new_logger

        return jsonify({"message": f"Modelo alterado para {selected_model} com sucesso."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    # Iniciar a captura de pacotes em uma thread separada
    sniffing_thread = threading.Thread(target=packet_sniffer.start_sniffing)
    sniffing_thread.start()

    # Iniciar o servidor Flask
    app.run(host='0.0.0.0', port=5000)
