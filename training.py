import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# Configuração de log
logging.basicConfig(filename='log_poux.log', level=logging.INFO)

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

data_folder = "C:/Users/gustavo/Desktop/Projeto IA/Nuvens/"

dataset = "3DML_urban_point_cloud.xyz"
val_dataset = "3DML_validation.xyz"

try:
    print_status("Carregando datasets")
    pcd = pd.read_csv(data_folder + dataset, delimiter=' ')
    pcd.dropna(inplace=True)
    val_pcd = pd.read_csv(data_folder + val_dataset, delimiter=' ')
    val_pcd.dropna(inplace=True)
    
    print_status("Processando features e rótulos")
    val_labels = val_pcd['Classification']
    val_features = val_pcd[['Z', 'R', 'G', 'B', 'omnivariance_2', 'normal_cr_2', 'NumberOfReturns', 'planarity_2', 'omnivariance_1', 'verticality_1']]
    val_features_sampled, val_features_test, val_labels_sampled, val_labels_test = train_test_split(val_features, val_labels, test_size=0.9)
    val_features_scaled_sample = MinMaxScaler().fit_transform(val_features_test)
    
    labels = pd.concat([pcd['Classification'], val_labels_sampled])
    features = pd.concat([pcd[['Z', 'R', 'G', 'B', 'omnivariance_2', 'normal_cr_2', 'NumberOfReturns', 'planarity_2', 'omnivariance_1', 'verticality_1']], val_features_sampled])
    features_scaled = MinMaxScaler().fit_transform(features)
    
    print_status("Treinando o modelo")
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.4)
    start_train = time.time()
    
    rf_classifier = RandomForestClassifier(n_estimators=10)
    rf_classifier.fit(X_train, y_train)
    end_train = time.time()
    
    print_status(f"Treinamento concluído em {end_train - start_train:.2f} segundos")
    
    rf_predictions = rf_classifier.predict(X_test)
    print_status("Calculando métricas")
    print_status(f"Matriz de Confusão:\n{confusion_matrix(y_test, rf_predictions)}")
    print_status(f"Relatório de Classificação:\n{classification_report(y_test, rf_predictions)}")
    
    with open('model_trained.pkl', 'wb') as f:
        pickle.dump(rf_classifier, f)
    print_status("Modelo treinado salvo")
    
    print_status("Classificando dados de validação")
    val_features_scaled = MinMaxScaler().fit_transform(val_features)
    val_rf_predictions = rf_classifier.predict(val_features_scaled)
    
    val_pcd['Classification'] = val_rf_predictions
    val_pcd.to_csv('classified_cloud_point.xyz', sep=' ', index=False, header=True)
    print_status("Dados classificados salvos")
    
except Exception as e:
    error_message = f"[ERRO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {str(e)}"
    print(error_message)
    logging.error(error_message)

finally:
    total_time_seconds = time.time() - start_time
    total_time_hours = total_time_seconds / 3600
    print_status(f"Tempo total de execução: {total_time_hours:.2f} horas")