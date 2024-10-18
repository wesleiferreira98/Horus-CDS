import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
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
features = ['Dia_da_Semana', 'Mês', 'Hora']
target = 'LONGTIME'

X = data[features]
y = data[target]

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir pipeline de pré-processamento
numeric_features = ['Dia_da_Semana', 'Mês', 'Hora']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Definir o modelo de rede neural
model = Sequential([
    Dense(64, activation='relu', input_shape=(3,)),
    Dense(32, activation='relu'),
    Dense(1)  # Camada de saída
])

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Criar o pipeline completo
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Treinar o modelo
pipeline.fit(X_train, y_train, model__epochs=30, model__batch_size=32)

# Fazer previsões
y_pred = pipeline.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

# Prever para os próximos 10 dias no mesmo horário
data_test_recente = X_test.iloc[-1]  # Obter a última linha do conjunto de dados de teste
data_recente = data_test_recente['Dia_da_Semana'], data_test_recente['Mês'], data_test_recente['Hora']
#data_final = X_test.index[-1] + pd.DateOffset(days=10)  # Adicionar 10 dias à última data disponível
#proximos_10_dias = pd.date_range(data_final, periods=10, freq='D')

# Montar o DataFrame de previsões para os próximos 10 dias
previsoes_10_dias = pd.DataFrame({'Dia_da_Semana': [data_recente[0]] * 10,
                                  'Mês': [data_recente[1]] * 10,
                                  'Hora': [data_recente[2]] * 10})
previsoes_10_dias['Previsao_LONGTIME'] = pipeline.predict(previsoes_10_dias)

# Plotar o gráfico
plt.figure(figsize=(10, 6))
plt.scatter(pd.to_datetime(X_test['Dia_da_Semana']), y_test, label='Valor Real', marker='o')
plt.scatter(pd.to_datetime(X_test['Dia_da_Semana']), y_pred, label='Previsão', marker='o')
#plt.plot(proximos_10_dias, previsoes_10_dias['Previsao_LONGTIME'], label='Previsão para 10 dias', marker='o')
plt.xlabel('Data')
plt.ylabel('LONGTIME')
plt.title('Previsão de LONGTIME vs. Valor Real')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Calcular o RMSE
rmse = np.sqrt(mse)

# Plotar as métricas de avaliação do modelo
plt.figure(figsize=(8, 6))
plt.bar(['MSE', 'RMSE'], [mse, rmse], color=['skyblue', 'orange'])
plt.xlabel('Métrica de Avaliação')
plt.ylabel('Valor')
plt.title('Métricas de Avaliação do Modelo')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
