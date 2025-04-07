import pandas as pd
import time
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

# Diretório dos arquivos
data_folder = "./data/"

# Nome dos arquivos
dataset = "3DML_urban_point_cloud.xyz"
val_dataset = "3DML_validation.xyz"

# Configuração de log
logging.basicConfig(filename='logs.log', level=logging.INFO)

def memory_usage():
    try:
        with open('/proc/self/status') as f:
            lines = f.readlines()
        for line in lines:
            if 'VmRSS' in line:
                memory_usage = line.split()[1]
                return f"{int(memory_usage) // 1024} MB"
    except FileNotFoundError:
        return "Informação de memória não disponível."

def print_status(message):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    status_message = f"{current_time} [STATUS]: {message} - MEM: {memory_usage()}"
    print(status_message)
    logging.info(status_message)

start_time = time.time()
print_status("Início do processamento")

try:
    # Carregamento dos datasets
    print_status("Carregando datasets")
    pcd = pd.read_csv(data_folder + dataset, delimiter=' ')
    pcd.dropna(inplace=True)
    val_pcd = pd.read_csv(data_folder + val_dataset, delimiter=' ')
    val_pcd.dropna(inplace=True)

    # Processamento de features e rótulos
    print_status("Processando features e rótulos")
    val_labels = val_pcd['Classification']
    val_features = val_pcd[['Z', 'R', 'G', 'B', 'omnivariance_2', 'normal_cr_2',
                            'NumberOfReturns', 'planarity_2', 'omnivariance_1', 'verticality_1']]
    val_features_sampled, _, val_labels_sampled, _ = train_test_split(val_features, val_labels, test_size=0.9)
    val_features_scaled_sample = MinMaxScaler().fit_transform(val_features_sampled)

    labels = pd.concat([pcd['Classification'], val_labels_sampled])
    features = pd.concat([pcd[['Z', 'R', 'G', 'B', 'omnivariance_2', 'normal_cr_2',
                               'NumberOfReturns', 'planarity_2', 'omnivariance_1', 'verticality_1']], val_features_sampled])
    features_scaled = MinMaxScaler().fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.4)

    # Função auxiliar para treinar e avaliar modelos
    def train_and_evaluate_model(model, model_name):
        print_status(f"Treinando o modelo {model_name}")
        start_train = time.time()
        model.fit(X_train, y_train)
        end_train = time.time()
        print_status(f"{model_name} treinado em {end_train - start_train:.2f} segundos")

        predictions = model.predict(X_test)
        print_status(f"Métricas do modelo {model_name}:")
        print_status(f"Matriz de Confusão {model_name}:\n{confusion_matrix(y_test, predictions)}")
        print_status(f"Relatório de Classificação {model_name}:\n{classification_report(y_test, predictions)}")

        return model

    # Modelos
    rf_classifier = train_and_evaluate_model(RandomForestClassifier(n_estimators=10), "Random Forest")
    knn_classifier = train_and_evaluate_model(KNeighborsClassifier(n_neighbors=5), "KNN")
    mlp_classifier = train_and_evaluate_model(MLPClassifier(hidden_layer_sizes=(100,), max_iter=300), "MLP")

except Exception as e:
    error_message = f"[ERRO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {str(e)}"
    print(error_message)
    logging.error(error_message)

finally:
    total_time_seconds = time.time() - start_time
    total_time_hours = total_time_seconds / 3600
    print_status(f"Tempo total de execução: {total_time_hours:.2f} horas")
