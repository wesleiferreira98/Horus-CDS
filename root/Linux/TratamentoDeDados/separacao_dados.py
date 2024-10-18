from sklearn.model_selection import train_test_split
import pandas as pd

# Carregar o arquivo CSV com os dados filtrados
df = pd.read_csv('dados_normalizados_smartgrid.csv')

# Dividir os dados em conjuntos de treinamento e teste
# Por padrão, a função train_test_split divide os dados em 75% para treinamento e 25% para teste
# Para obter metade para treinamento e metade para teste, definimos test_size=0.5
# random_state é usado para garantir a reprodutibilidade da divisão
train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)

# Salvar os conjuntos de treinamento e teste em arquivos CSV
train_df.to_csv('dados_treinamento.csv', index=False)
test_df.to_csv('dados_teste.csv', index=False)

print("Os dados foram divididos em conjuntos de treinamento e teste.")
print("Os dados de treinamento foram salvos em 'dados_treinamento.csv'.")
print("Os dados de teste foram salvos em 'dados_teste.csv'.")
