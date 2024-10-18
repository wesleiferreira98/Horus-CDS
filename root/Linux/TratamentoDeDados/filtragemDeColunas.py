import pandas as pd
from sklearn.preprocessing import StandardScaler

def filtrar_colunas(input_file, output_file):
    # Carregar o arquivo CSV
    df = pd.read_csv(input_file)

    # Definir a função para categorizar os acessos com base no tempo (LONGTIME)
    def categorize_access(longtime):
        # Se a coluna LONGTIME tiver apenas um dígito, retornar 'não categorizado'
        if len(str(longtime)) == 1:
            return 'não categorizado'

        # Caso contrário, considerar apenas os três últimos dígitos
        last_three_digits = str(longtime)[-3:]

        # Converter para inteiro
        last_three_digits = int(last_three_digits)

        if last_three_digits > 400:
            return 'válido'
        elif 300 <= last_three_digits <= 400:
            return 'suspeito'
        else:
            return 'ilegal'

    # Aplicar a função categorize_access à coluna 'LONGTIME' para criar a nova coluna 'CATEGORY'
    df['CATEGORY'] = df['LONGTIME'].apply(categorize_access)

    # Ajustar a coluna 'LONGTIME' para conter apenas os três últimos dígitos
    df['LONGTIME'] = df['LONGTIME'].astype(str).str[-3:].astype(int)

    # Excluir todas as linhas onde 'CATEGORY' é 'não categorizado'
    df = df[df['CATEGORY'] != 'não categorizado']

    # Selecionar apenas as colunas especificadas
    colunas_filtradas = df[['id', 'TXTDATE', 'TXTTIME', 'LONGTIME', 'INTACTIVE', 'INTMANUAL', 'CATEGORY']]
    
    # Substituir os valores de 'INTACTIVE' e 'INTMANUAL' com base na categoria
    colunas_filtradas['INTACTIVE'] = colunas_filtradas.apply(lambda row: 1 if row['CATEGORY'] == 'válido' else 0, axis=1)
    colunas_filtradas['INTMANUAL'] = colunas_filtradas.apply(lambda row: 1 if row['CATEGORY'] == 'válido' else 0, axis=1)

    # Salvar o novo dataframe em um novo arquivo CSV
    colunas_filtradas.to_csv(output_file, index=False)

    print("Filtragem concluída. Os dados filtrados foram salvos em", output_file)

# Nome do arquivo de entrada e saída
input_file = 'dataset_smartgrid2.csv'
output_file = 'dados_filtrados_smartgrid.csv'

# Chamar a função para filtrar as colunas e salvar o resultado
filtrar_colunas(input_file, output_file)
