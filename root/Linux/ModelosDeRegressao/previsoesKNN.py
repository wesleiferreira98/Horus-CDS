import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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

# Definir o modelo KNN
model = KNeighborsRegressor(n_neighbors=5)

# Criar o pipeline completo
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Treinar o modelo
pipeline.fit(X_train.drop(columns=['TXTDATE']), y_train)

# Fazer previsões
y_pred = pipeline.predict(X_test.drop(columns=['TXTDATE']))

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

# Prever para os próximos 10 dias no mesmo horário
horario_atual = X_test['Hora'].iloc[0]
proximos_10_dias = pd.date_range(X_test['TXTDATE'].iloc[-1], periods=10, freq='D')
previsoes_10_dias = pd.DataFrame({'TXTDATE': proximos_10_dias, 'Hora': horario_atual})
previsoes_10_dias['Mês'] = proximos_10_dias.month
previsoes_10_dias['Dia_da_Semana'] = proximos_10_dias.dayofweek
previsoes_10_dias['Previsao_LONGTIME'] = pipeline.predict(previsoes_10_dias.drop(columns=['TXTDATE']))

# Plotar o gráfico
plt.figure(figsize=(10, 6))
plt.scatter(pd.to_datetime(X_test['TXTDATE']), y_test, label='Valor Real', marker='o')
plt.scatter(pd.to_datetime(X_test['TXTDATE']), y_pred, label='Previsão', marker='o')
plt.scatter(previsoes_10_dias['TXTDATE'], previsoes_10_dias['Previsao_LONGTIME'], label='Previsão para 10 dias', marker='o')
plt.xlabel('Data')
plt.ylabel('LONGTIME')
plt.title('Previsão de LONGTIME vs. Valor Real (KNN)')
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
plt.title('Métricas de Avaliação do Modelo (KNN)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
