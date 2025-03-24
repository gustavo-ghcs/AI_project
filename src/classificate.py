import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Diretórios dos arquivos
data_folder = "./data/"
models_folder = "./models/"
output_folder = "./images/"

# Nome do arquivo da nuvem sem classificação
new_dataset = "nuvem_pontos_1.xyz"

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
output_file = data_folder + "nuvem_classificada.xyz"
new_pcd.to_csv(output_file, sep=' ', index=False, header=True)

print(f"Nuvem classificada salva em: {output_file}")

# Mapeando cores para as classes
color_map = {
    1: "#0009DE",  # Terreno (ground)
    2: "#7EFF0C",  # Vegetação (vegetation)
    3: "#FB0304"   # Edifícios (buildings)
}

# Converter previsões para cores
colors = np.array([color_map[class_id] for class_id in predictions])

# Gerar imagem da Nuvem de Pontos Classificada
print("Gerando imagem da Nuvem de Pontos Classificada")
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(new_pcd['X'], new_pcd['Y'], c=colors, s=0.05)
ax.set_title('3D Point Cloud Predictions')
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Salvar imagem
plt.savefig(output_folder + "nuvem_classificada.jpg", dpi=600)
print(f"Nuvem de Pontos Classificada salva em: {output_folder}nuvem_classificada.jpg")

print("Processo concluído com sucesso!")