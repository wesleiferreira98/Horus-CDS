from flask import Flask, send_from_directory, jsonify, request
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(BASE_DIR, 'dist')

app = Flask(__name__, static_folder=DIST_DIR, static_url_path='')

@app.route('/api-config')
def api_config():
    """
    Retorna a URL base da API para o frontend.

    Mesma máquina (padrão): deriva o host da própria requisição e usa porta 5000.
    Máquinas distintas: defina a variável de ambiente HORUS_API_URL antes de subir.

    Exemplo:
        HORUS_API_URL=http://192.168.1.20:5000 python run_web.py
    """
    api_url = os.environ.get('HORUS_API_URL')
    if not api_url:
        host = request.host.split(':')[0]
        api_url = f'http://{host}:5000'
    return jsonify({'apiUrl': api_url.rstrip('/')})

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    target = os.path.join(DIST_DIR, path)
    if path and os.path.exists(target):
        return send_from_directory(DIST_DIR, path)
    return send_from_directory(DIST_DIR, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
