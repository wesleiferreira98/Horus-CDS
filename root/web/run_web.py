from flask import Flask, render_template

app = Flask(__name__)

# Rota para servir a página principal do dashboard
@app.route('/')
def dashboard():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
