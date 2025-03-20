import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Diretório dos arquivos
data_folder = "./data/"

# Diretório dos modelos
models_folder = "./models/"

# Nome do arquivo da nuvem sem classificação
new_dataset = "3DML_validation.xyz"

# Carregar modelo
print("Carregando modelo treinado")
with open(models_folder + 'model_trained.pkl', 'rb') as f:
    rf_classifier = pickle.load(f)

# Carregar a nova nuvem de pontos
print("Carregando nova nuvem de pontos")
new_pcd = pd.read_csv(data_folder + new_dataset, delimiter=' ')
new_pcd.dropna(inplace=True)

# Selecionar as mesmas features usadas no treinamento
print("Selecionando features")
features_columns = ['Z', 'R', 'G', 'B', 'omnivariance_2', 'normal_cr_2', 'NumberOfReturns', 'planarity_2', 'omnivariance_1', 'verticality_1']
new_features = new_pcd[features_columns]

# Normalizar as features com MinMaxScaler
print("Normalizando features")
scaler = MinMaxScaler()
new_features_scaled = scaler.fit_transform(new_features)

# Fazer previsões
print("Fazendo previsões")
predictions = rf_classifier.predict(new_features_scaled)

# Adicionar a coluna de classificação ao dataframe
new_pcd['Classification'] = predictions

# Salvar a nuvem classificada
print("Salvando nuvem classificada")
output_file = data_folder + "nova_nuvem_classificada.xyz"
new_pcd.to_csv(output_file, sep=' ', index=False, header=True)

print(f"Nuvem classificada salva em: {output_file}")
