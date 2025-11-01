from flask import Flask, request, jsonify  
from flask_cors import CORS  # type: ignore
from packet_sniffer import PacketSniffer  
from prediction import Prediction  
from logger import Logger
import threading
import os

app = Flask(__name__)
CORS(app)

# Definir diretório base da API (onde app.py está localizado)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Inicializando o logger
LOG_DIR = os.path.join(BASE_DIR, 'logs')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Inicialize o modelo padrão
active_model_name = "Horus-CDS V4 (TCN)"

# Caminhos dos modelos (suporta .h5 e .keras)
model_paths = {
    "Horus-CDS V1 (RNN)": os.path.join(MODELS_DIR, "rnn_model.h5"),
    "Horus-CDS V2 (LSTM)": os.path.join(MODELS_DIR, "lstm_model.h5"),
    "Horus-CDS V3 (GRU)": os.path.join(MODELS_DIR, "gru_model.h5"),
    "Horus-CDS V4 (TCN)": os.path.join(MODELS_DIR, "tcn_model.keras"),  # Usando .keras por padrão
}

# Diretórios de logs por modelo
log_directories = {
    "Horus-CDS V1 (RNN)": os.path.join(LOG_DIR, "rnn_logs"),
    "Horus-CDS V2 (LSTM)": os.path.join(LOG_DIR, "lstm_logs"),
    "Horus-CDS V3 (GRU)": os.path.join(LOG_DIR, "gru_logs"),
    "Horus-CDS V4 (TCN)": os.path.join(LOG_DIR, "tcn_logs"),
}

# Inicialize o logger e o modelo ativo
os.makedirs(log_directories[active_model_name], exist_ok=True)
logger = Logger(os.path.join(log_directories[active_model_name], "packet_logs.txt"))
prediction_model = Prediction(model_paths[active_model_name], SCALER_PATH)

# Instâncias das classes
# simulation_mode=None: Modo real por padrão, fallback para simulação se sem permissões
# simulation_mode=True: Força modo simulação
# simulation_mode=False: Força modo real
packet_sniffer = PacketSniffer(prediction_model, logger, simulation_mode=None)

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
        # Verificar se o arquivo do modelo existe
        model_file = model_paths[selected_model]
        if not os.path.exists(model_file):
            return jsonify({"error": f"Arquivo do modelo não encontrado: {model_file}"}), 404
        
        # Verificar se o scaler existe
        if not os.path.exists(SCALER_PATH):
            return jsonify({"error": f"Arquivo scaler.pkl não encontrado: {SCALER_PATH}"}), 404
        
        # Criar diretório de logs se não existir
        os.makedirs(log_directories[selected_model], exist_ok=True)
        
        # Criar nova instância antes do lock
        new_model = Prediction(model_file, SCALER_PATH)
        new_logger = Logger(os.path.join(log_directories[selected_model], "packet_logs.txt"))

        with packet_sniffer.lock:
            active_model_name = selected_model
            prediction_model = new_model
            logger = new_logger
            packet_sniffer.prediction_model = new_model
            packet_sniffer.logger = new_logger

        return jsonify({"message": f"Modelo alterado para {selected_model} com sucesso."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Retorna o status da API e modo de captura"""
    return jsonify({
        "active_model": active_model_name,
        "simulation_mode": packet_sniffer.simulation_mode,
        "scapy_available": packet_sniffer.simulation_mode == False or os.geteuid() == 0,
        "running_as_root": os.geteuid() == 0,
        "mode_description": "Simulação (sem sudo)" if packet_sniffer.simulation_mode else "Captura Real (com sudo)"
    })

@app.route('/toggle_mode', methods=['POST'])
def toggle_mode():
    """Alterna entre modo simulação e modo real"""
    try:
        new_mode = request.json.get('simulation_mode', True)
        
        if not new_mode and os.geteuid() != 0:
            return jsonify({
                "error": "Modo real requer permissões de root. Execute com sudo ou mantenha modo simulação.",
                "current_mode": "simulation"
            }), 403
        
        # Para a captura atual
        packet_sniffer.stop_sniffing()
        
        # Atualiza o modo
        packet_sniffer.simulation_mode = new_mode
        
        # Reinicia a captura
        sniffing_thread = threading.Thread(target=packet_sniffer.start_sniffing, daemon=True)
        sniffing_thread.start()
        
        mode_desc = "Simulação (sem sudo)" if new_mode else "Captura Real (com sudo)"
        return jsonify({
            "message": f"Modo alterado para: {mode_desc}",
            "simulation_mode": new_mode
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Iniciar a captura de pacotes em uma thread separada
    sniffing_thread = threading.Thread(target=packet_sniffer.start_sniffing, daemon=True)
    sniffing_thread.start()

    # Iniciar o servidor Flask
    app.run(host='0.0.0.0', port=5000)
