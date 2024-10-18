# Documentação da Ferramenta SPTI

---

### 1. **Introdução**

- **Nome da ferramenta**: SPTI (Sistema de Previsão e Tratamento de Incidentes)
- **Descrição geral**: A SPTI é uma solução para detectar e tratar incidentes de segurança em redes, com foco em ataques de botnet e outras ameaças cibernéticas. Utilizando redes neurais, como **TCN (Temporal Convolutional Network)**, a ferramenta analisa pacotes de rede para identificar atividades maliciosas. A SPTI inclui uma API REST para integração com outras soluções e uma interface gráfica desenvolvida em **PyQt5**. Além disso, conta com um **dashboard web** para monitoramento e visualização em tempo real.

---

### 2. **Requisitos do Sistema**

- **Sistema Operacional**: Linux, Windows, macOS
- **Dependências**:
  - **Python 3.x**
  - **Bibliotecas Python**:
    - `PyQt5`: Interface gráfica.
    - `TensorFlow/Keras`: Para o modelo TCN.
    - `Pandas`, `NumPy`: Manipulação e análise de dados.
    - `Scikit-learn`: Normalização e processamento de dados.
    - **Frameworks Web**:
      - `Flask` ou `FastAPI` para a API REST.
      - `HTML`, `CSS`, `JavaScript` para o desenvolvimento da parte web.
    - **Outras dependências**:
      - Docker (opcional) para isolar e facilitar a instalação.

---

### 3. **Instalação**

- **Passo 1**: Clone o repositório do projeto:
  ```bash
  git clone https://github.com/usuario/spti.git
  cd spti
  ```
- **Passo 2**: Instale as dependências:
  ```bash
  pip install -r requirements.txt
  ```
- **Passo 3**: (Opcional) Utilize Docker para configurar o ambiente rapidamente:
  - **Dockerfile**: O projeto inclui um `Dockerfile` que pode ser utilizado para construir e rodar a aplicação em um contêiner.

  ```bash
  docker build -t spti .
  docker run -p 5000:5000 spti
  ```

---

### 4. **Execução da API**

- **Passo 1**: Execute o arquivo da API:
  ```bash
  python app.py
  ```
- **Passo 2**: O sistema abrirá um endpoint em `http://localhost:5000`, permitindo o envio de pacotes de rede para detecção de ataques.
- **Exemplo de Endpoint**:
  - **POST /predict**
    - Enviar pacotes para o endpoint `/predict` para análise:
      ```bash
      curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"features": [dados_do_pacote]}'
      ```
  - **Resposta**:
    ```json
    {
      "prediction": [valor_da_predição],
      "status": "Ataque" ou "Permitido"
    }
    ```

---

### 5. **Interface Gráfica (Python Qt5)**

- **Descrição**: A interface gráfica desenvolvida em **PyQt5** oferece uma visualização em tempo real das predições de pacotes de rede, exibindo os dados normalizados e desnormalizados.
- **Principais Funcionalidades**:
  - Exibição de gráficos com predições de ataques e status de tráfego de rede.
  - Painel de controle para visualização de requisições, mostrando IP de origem/destino, data/hora e status (Ataque/Permitido).
  - Geração de logs e gráficos exportáveis.
- **Execução**:
  ```bash
  python SPTI.py
  ```

---

### 6. **Parte Web do SPTI**

- **Descrição**: O **dashboard web** do SPTI foi desenvolvido utilizando as tecnologias **HTML**, **CSS** e **JavaScript**, em conjunto com o **Flask** ou **FastAPI** para a comunicação entre o front-end e o back-end. A interface web permite monitorar as requisições e predições em tempo real, com gráficos e tabelas interativas para uma visão completa do status da rede.
- **Estrutura**:

  - **HTML**: A estrutura básica do dashboard é montada em HTML, com elementos que incluem gráficos, tabelas de requisições e botões de controle.
  - **CSS**: Responsável pelo design e layout da página, garantindo que os componentes sejam apresentados de maneira clara e organizada.
  - **JavaScript**: Utilizado para manipular os dados em tempo real, atualizando os gráficos e a tabela de requisições conforme os pacotes são processados.
  - **API**: A comunicação entre o front-end (dashboard web) e o back-end (modelo de predição) é feita por meio de chamadas à API REST, que retorna os resultados da análise de pacotes.
- **Funcionalidades**:

  - **Exibição em tempo real**: O dashboard exibe gráficos que mostram as predições de pacotes de rede em tempo real, classificando-os como "Ataque" ou "Permitido".
  - **Tabela de requisições**: Mostra os detalhes de cada requisição, incluindo o IP de origem e destino, data/hora e status de segurança.
  - **Gráficos interativos**: O gráfico de predições é atualizado dinamicamente, fornecendo uma visão clara das tendências e anomalias detectadas pelo modelo.
- **Arquitetura do Front-End**:

  - **HTML** (em `templates/`): Define a estrutura da página.
  - **CSS** (em `static/css/`): Controla o estilo e layout do dashboard.
  - **JavaScript** (em `static/js/`): Atualiza os dados em tempo real, conecta-se à API e renderiza os gráficos.
  - **Flask/FastAPI**: Serve o conteúdo da aplicação e fornece os dados necessários para o front-end.
- **Como Executar**:

  ```bash
  python run_web.py
  ```

  - **Acesso**: Após rodar o comando, abra um navegador e acesse `http://localhost:5000` para visualizar o dashboard.

---

### 7. **Exemplo de Execução Completa**

- **Passo 1**: Execute o script principal da API e a interface web:
  ```bash
  python start_app.py
  ```
- **Passo 2**: Acesse o **dashboard** da ferramenta através de `http://localhost:5000`.
  - O dashboard exibirá as predições do modelo e as requisições de pacotes sendo atualizadas em tempo real.
  - Os gráficos mostram os resultados das predições e se os pacotes foram classificados como "Ataque" ou "Permitido".

---

### 8. **Testes e Exemplos de Uso**

- **Exemplo de Teste Mínimo**:
  - Enviar um pacote de rede malicioso pela API e verificar o resultado no dashboard web. A API deverá identificar a atividade como um **ataque** e exibir a informação no gráfico e na tabela de requisições.
- **Testes com múltiplos cenários**:
  - Teste o sistema enviando pacotes de diferentes tipos de ataque, como **DoS**, **injeção de comandos** ou **ataques de replay**, e veja o comportamento do modelo no dashboard web.

---

### 9. **Contribuição e Suporte**

- **Como contribuir**:
  - Para contribuir com o desenvolvimento da SPTI:
    - Faça um **fork** do projeto.
    - Crie uma nova **branch** para as suas alterações.
    - Envie um **pull request** com a descrição detalhada das modificações.
- **Suporte**:
  - Em caso de dúvidas ou problemas, consulte a seção de **issues** no repositório ou entre em contato com os desenvolvedores.

---

### 10. **Futuras Melhorias**

- **Otimizações no Modelo**: Melhorar a precisão do modelo TCN ou explorar outros algoritmos de machine learning.
- **Dashboard aprimorado**: Adicionar mais funcionalidades ao dashboard web, como filtros avançados e relatórios exportáveis.
- **Integração com Ferramentas de Segurança**: Expansão da API para se integrar com sistemas como **Wazuh**, aumentando as capacidades de monitoramento e resposta a incidentes.

---
