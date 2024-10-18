### Documentação da Ferramenta SPTI

---

#### 1. **Introdução**
   - **Nome da ferramenta**: SPTI (Sistema de Previsão e Tratamento de Incidentes)
   - **Descrição geral**: A SPTI é uma ferramenta desenvolvida para detectar ataques em pacotes de rede, com foco em ataques de botnet, utilizando redes neurais (modelo TCN). A ferramenta também oferece uma interface gráfica em Python (Qt5) para visualização de dados, além de uma API que permite integração com outros sistemas de segurança.

#### 2. **Requisitos do Sistema**
   - **Sistema Operacional**: Linux, Windows, macOS
   - **Dependências**:
     - Python 3.x
     - Bibliotecas:
       - `PyQt5`: Interface gráfica
       - `TensorFlow/Keras`: Para o modelo TCN
       - `Pandas, NumPy, Scikit-learn`: Manipulação de dados
       - `Flask/FastAPI`: Para a API
       - **Outras dependências**: Mencionar Docker, caso queira usar para facilitar a instalação.
   
#### 3. **Instalação**
   - **Passo 1**: Clone o repositório no GitHub:
     ```bash
     git clone https://github.com/usuario/spti.git
     cd spti
     ```
   - **Passo 2**: Instale as dependências:
     ```bash
     pip install -r requirements.txt
     ```
   - **Passo 3**: (Opcional) Usar Docker para configurar o ambiente:
     - **Instruções Dockerfile**:
       - Passos para construir e rodar a imagem Docker, incluindo as dependências e configurações necessárias.
   
#### 4. **Exemplo de Execução da API**
   - Para rodar a API:
     ```bash
     python api_spti.py
     ```
   - O sistema abrirá um endpoint onde você pode enviar pacotes de rede para detecção de ataques.
   - **Endpoint de exemplo**:
     ```bash
     POST /detect
     Body: { "packet_data": "dados do pacote" }
     ```
   - **Resposta esperada**: A API retorna se o pacote foi classificado como "permitido" ou "ataque", juntamente com informações adicionais.

#### 5. **Interface Gráfica (Python Qt5)**
   - **Descrição**: A interface gráfica foi desenvolvida com PyQt5 para exibir os dados em tempo real.
   - **Principais Funcionalidades**:
     - Exibição de gráficos que mostram o tráfego de rede, previsões do modelo, e detecções de ataque.
     - Um painel de controle para visualizar as requisições com data, IP de origem/destino, e status (ataque/permitido).
   - **Como Executar**:
     ```bash
     python gui.py
     ```
   - **Tela principal**: 
     - Mostra gráficos de predições normalizadas e a interface interativa para monitorar ataques em tempo real.
     - A classe `GraphThread` é responsável por gerar gráficos em uma thread separada e passar os dados para a `STARAYERApp`.

#### 6. **Exemplo de Execução com a Interface Gráfica**
   - **Passo 1**: Execute o script que inicia a API e o front-end:
     ```bash
     python start_app.py
     ```
   - **Passo 2**: Abra o navegador e acesse o dashboard da ferramenta.
     - Você verá os gráficos de previsões e as requisições sendo atualizadas em tempo real.

#### 7. **Testes e Exemplos de Uso**
   - **Exemplo mínimo** de detecção de ataque:
     - Enviar um pacote de rede malicioso pela API e visualizar o resultado no dashboard.
   - **Exemplo com múltiplas requisições**:
     - Testar o sistema com diferentes pacotes para ver o comportamento em diferentes cenários (DoS, injeção de comandos, etc.).

#### 8. **Contribuição e Suporte**
   - **Como contribuir**:
     - Instruções para forks, pull requests, e contribuição para o projeto.
   - **Suporte**:
     - Onde encontrar ajuda para problemas com a ferramenta.

---
