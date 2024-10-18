import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os 


def normalizacao(output_file):
    # Carregar o arquivo CSV com os dados filtrados
    df = pd.read_csv(output_file)

    # Selecionar apenas as colunas numéricas para normalização
    numeric_columns = ['LONGTIME']

    # Criar um objeto StandardScaler
    scaler = StandardScaler()

    # Normalizar os dados
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Salvar o novo conjunto de dados normalizado em um novo arquivo CSV
    output_normalized_file = 'dados_normalizados_smartgrid.csv'
    df.to_csv(output_normalized_file, index=False)

    print("Dados normalizados foram salvos em", output_normalized_file)


def treinar_e_salvar_scaler(input_file, scaler_file):
    # Carregar o arquivo CSV com os dados originais
    df = pd.read_csv(input_file)

    # Selecionar apenas a coluna 'LONGTIME' para normalização
    numeric_columns = ['LONGTIME']

    # Criar um objeto StandardScaler
    scaler = StandardScaler()

    # Treinar o scaler com os dados originais
    scaler.fit(df[numeric_columns])

    # Salvar o scaler treinado para uso posterior
    joblib.dump(scaler, scaler_file)

    print(f"Scaler treinado foi salvo em {scaler_file}")

# Treinar e salvar o scaler
def save_model():
        output_directory = "TratamentoDeDados"
        os.makedirs(output_directory, exist_ok=True)
        pkl = os.path.join(output_directory,"scaler.pkl")
        csv = os.path.join(output_directory,"dados_filtrados_smartgrid.csv")
        treinar_e_salvar_scaler(csv,pkl)


save_model()

#normalizacao("dados_filtrados_smartgrid.csv")