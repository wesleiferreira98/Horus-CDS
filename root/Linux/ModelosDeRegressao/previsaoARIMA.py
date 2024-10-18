import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Carregar os dados
data = pd.read_csv('dados_normalizados_smartgrid.csv')

# Converter 'TXTDATE' para o formato de data
data['TXTDATE'] = pd.to_datetime(data['TXTDATE'])

# Extrair características de data e hora
data['Dia_da_Semana'] = data['TXTDATE'].dt.dayofweek
data['Mês'] = data['TXTDATE'].dt.month
data['Hora'] = pd.to_datetime(data['TXTTIME'], format='%H:%M:%S').dt.hour

# Selecionar características e alvo
features = ['TXTDATE', 'Dia_da_Semana', 'Mês', 'Hora']
target = 'LONGTIME'

X = data[features]
y = data[target]

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test = X[:int(0.8*len(X))], X[int(0.8*len(X)):]
y_train, y_test = y[:int(0.8*len(y))], y[int(0.8*len(y)):]

# Ajustar o modelo ARIMA
model = ARIMA(y_train, order=(5,1,0))  # Exemplo de ARIMA(5,1,0)
arima_model = model.fit()

# Fazer previsões
y_pred = arima_model.forecast(steps=len(X_test))

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

# Plotar o gráfico das previsões com ARIMA
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test.values, label='Dados de Teste', marker='o')
plt.plot(y_test.index, y_pred, label='Previsões ARIMA', marker='o')
plt.xlabel('Índice')
plt.ylabel('LONGTIME')
plt.title('Previsões de LONGTIME com ARIMA')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calcular o RMSE
rmse = np.sqrt(mse)

# Plotar as métricas de avaliação do modelo
plt.figure(figsize=(8, 6))
plt.bar(['MSE', 'RMSE'], [mse, rmse], color=['skyblue', 'orange'])
plt.xlabel('Métrica de Avaliação')
plt.ylabel('Valor')
plt.title('Métricas de Avaliação do Modelo (ARIMA)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
