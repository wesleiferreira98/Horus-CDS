# Horus-CDS API - Sistema de Detecção de Intrusão

## Descrição

API REST para detecção de ataques em tempo real usando modelos de Deep Learning (RNN, LSTM, GRU, TCN).

## Pré-requisitos

### Arquivos Necessários (na pasta `models/`):

```
root/API/models/
├── rnn_model.h5         # Modelo RNN
├── lstm_model.h5        # Modelo LSTM  
├── gru_model.h5         # Modelo GRU
├── tcn_model.keras      # Modelo TCN (padrão)
└── scaler.pkl           # Normalizador (OBRIGATÓRIO)
```

 **IMPORTANTE:** O arquivo `scaler.pkl` é **obrigatório** e deve estar na pasta `models/`.

### Dependências Python:

```bash
pip install flask flask-cors tensorflow scapy joblib matplotlib numpy keras-tcn
```

**Nota:** O pacote `keras-tcn` é necessário para carregar modelos TCN personalizados.

## Como Executar

### 1. Preparar os Modelos

Certifique-se de que os arquivos dos modelos e o `scaler.pkl` estejam na pasta `models/`:

```bash
cd root/API
ls models/
# Deve mostrar: gru_model.h5, lstm_model.h5, rnn_model.h5, tcn_model.keras, scaler.pkl
```

### 2. Iniciar a API

#### Modo Captura Real (Padrão - COM sudo):

```bash
sudo python app.py
```

A API tentará capturar pacotes de rede reais da interface `lo`. Este é o modo padrão e requer permissões de root.

#### Modo Simulação (Fallback Automático - SEM sudo):

```bash
python app.py
```

Se executado sem `sudo`, a API detectará automaticamente a falta de permissões e alternará para o modo simulação, gerando pacotes sintéticos para teste.

A API estará disponível em: `http://0.0.0.0:5000`

### 3. Iniciar o Frontend (Opcional)

Em outro terminal:

```bash
cd root/web
python run_web.py
```

O dashboard estará disponível em: `http://localhost:5001`

## Endpoints da API

### 1. Obter Dados dos Logs

```http
GET /gerar_dados?filter=<filtro>
```

**Parâmetros:**

- `filter` (opcional): `todos`, `recentes`, `antigos`

**Resposta:**

```json
{
  "packet_logs": {
    "ataques_detectados": 10,
    "requisicoes_permitidas": 50,
    "inconclusivos": 5,
    "logs": [...]
  },
  "predictions_log": {
    "normalized_predictions": [],
    "desnormalized_predictions": [],
    "resultados": []
  }
}
```

### 2. Gráfico de Predições

```http
GET /prediction_chart
```

Retorna: Imagem PNG

### 3. Gráfico de Análise de Logs

```http
GET /log_chart
```

Retorna: Imagem PNG

### 4. Fazer Predição Manual

```http
POST /predict
Content-Type: application/json

{
  "features": [[30, 40, 50, 60, 70, 80, 90, 100]]
}
```

**Resposta:**

```json
{
  "prediction": 180.5,
  "status": "Permitido"
}
```

### 5. Trocar Modelo em Runtime

```http
POST /set_model
Content-Type: application/json

{
  "model": "Horus-CDS V2 (LSTM)"
}
```

**Modelos Disponíveis:**

- `Horus-CDS V1 (RNN)`
- `Horus-CDS V2 (LSTM)`
- `Horus-CDS V3 (GRU)`
- `Horus-CDS V4 (TCN)` (padrão)

**Resposta:**

```json
{
  "message": "Modelo alterado para Horus-CDS V2 (LSTM) com sucesso."
}
```

### 6. Verificar Status da API

```http
GET /status
```

**Resposta:**

```json
{
  "active_model": "Horus-CDS V4 (TCN)",
  "simulation_mode": true,
  "scapy_available": true,
  "running_as_root": false,
  "mode_description": "Simulação (sem sudo)"
}
```

### 7. Alternar Modo de Captura

```http
POST /toggle_mode
Content-Type: application/json

{
  "simulation_mode": true
}
```

**Parâmetros:**

- `simulation_mode: true` → Modo simulação (sem sudo)
- `simulation_mode: false` → Modo captura real (requer sudo)

**Resposta:**

```json
{
  "message": "Modo alterado para: Simulação (sem sudo)",
  "simulation_mode": true
}
```

## Modos de Operação

### Modo Captura Real (Padrão)

**Características:**

- Captura pacotes TCP reais da interface de rede (interface: lo)
- Requer permissões de root (sudo)
- Usado para análise de tráfego real em produção
- Modo recomendado para uso operacional

**Como usar:**

```bash
sudo python app.py
```

**Alternativas ao sudo (Avançado):**

1. **Dar permissões CAP_NET_RAW ao Python:**

```bash
sudo setcap cap_net_raw+ep $(which python3)
```

2. **Adicionar usuário ao grupo com permissões:**

```bash
sudo usermod -aG wireshark $USER
# Reiniciar sessão após este comando
```

**Nota de Segurança:** Estas alternativas podem criar vulnerabilidades de segurança.

### Modo Simulação (Fallback Automático)

**Características:**

- Ativado automaticamente quando sem permissões root
- Não requer sudo
- Gera pacotes sintéticos automaticamente (a cada 2-5 segundos)
- Ideal para desenvolvimento, testes e demonstrações
- Não interfere com a rede real

**Como usar:**

```bash
python app.py
# Se executado sem sudo, alternará automaticamente para modo simulação
```

## Estrutura de Diretórios

```
root/API/
├── app.py                    # Servidor Flask principal
├── packet_sniffer.py         # Capturador de pacotes
├── prediction.py             # Motor de predição
├── logger.py                 # Sistema de logging
├── README_API.md            # Esta documentação
├── models/                   # Modelos treinados
│   ├── rnn_model.h5
│   ├── lstm_model.h5
│   ├── gru_model.h5
│   ├── tcn_model.keras
│   └── scaler.pkl           # OBRIGATÓRIO
└── logs/                     # Logs gerados automaticamente
    ├── rnn_logs/
    │   └── packet_logs.txt
    ├── lstm_logs/
    │   └── packet_logs.txt
    ├── gru_logs/
    │   └── packet_logs.txt
    ├── tcn_logs/
    │   └── packet_logs.txt
    └── predictions_log.txt
```

## Threshold de Detecção

O sistema usa um threshold de **200.0**:

- **< 200**: Classificado como **Ataque**
- **≥ 200**: Classificado como **Permitido**

## Troubleshooting

### Erro: "Could not locate class 'TCN'"

**Causa:** A biblioteca `keras-tcn` não está instalada ou importada.

**Solução:**

```bash
pip install keras-tcn
```

Reinicie a API após a instalação.

### Erro: "Arquivo do modelo não encontrado"

**Solução:** Verifique se o arquivo do modelo existe na pasta `models/`:

```bash
ls -la models/
```

### Erro: "Arquivo scaler.pkl não encontrado"

**Solução:** O arquivo `scaler.pkl` é obrigatório. Copie-o para a pasta `models/`:

```bash
cp /caminho/do/scaler.pkl models/
```

### Erro: ModuleNotFoundError

**Solução:** Instale as dependências:

```bash
pip install flask flask-cors tensorflow scapy joblib matplotlib numpy
```

### API não responde

**Solução:** Verifique se a porta 5000 está disponível:

```bash
lsof -i :5000
# Se estiver em uso, mate o processo ou mude a porta em app.py
```

## Formatos Suportados

A API suporta modelos nos formatos:

- `.h5` (HDF5 - TensorFlow/Keras antigo)
- `.keras` (Formato nativo do Keras 3.0+)

## Troca de Modelos

A troca de modelos é feita **em tempo de execução** sem necessidade de reiniciar o servidor:

1. Pelo Dashboard Web (Interface gráfica)
2. Via API REST usando o endpoint `/set_model`

**Nota:** Os logs são salvos em diretórios separados para cada modelo.

## Configurações

### Porta do Servidor

Para mudar a porta, edite `app.py`:

```python
app.run(host='0.0.0.0', port=5000)  # Mude 5000 para sua porta
```

### Interface de Captura

Para mudar a interface de rede, edite `packet_sniffer.py`:

```python
sniff(prn=self.process_packet, iface='lo')  # Mude 'lo' para 'eth0', 'wlan0', etc.
```

## Logs

Os logs são salvos automaticamente em:

- `logs/<modelo>_logs/packet_logs.txt` - Logs de pacotes capturados
- `logs/predictions_log.txt` - Logs de predições realizadas

## Segurança

**Aviso de Segurança:**

- A API está configurada para aceitar conexões de qualquer origem (`host='0.0.0.0'`)
- **Não exponha diretamente para a internet sem autenticação**
- Use apenas em redes confiáveis ou adicione autenticação

## Suporte

Para problemas ou dúvidas, consulte a documentação completa ou abra uma issue no repositório.

---

**Versão:** 4.0
**Última atualização:** 2025-11-01
