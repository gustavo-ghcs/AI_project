import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time
import logging

# Configuração de log
logging.basicConfig(filename='log_apuana.log', level=logging.INFO)

# Função para monitorar o uso de memória
def memory_usage():
    try:
        with open('/proc/self/status') as f:
            lines = f.readlines()
        for line in lines:
            if 'VmRSS' in line:  # VmRSS is the Resident Set Size (memory usage)
                memory_usage = line.split()[1]  # The value is in kilobytes
                return f"{int(memory_usage) // 1024} MB"  # Convert to MB
    except FileNotFoundError:
        return "Informação de memória não disponível no sistema."

# Função para imprimir status e salvar no log
def print_status(message):
    # Obtém a data e hora atuais
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    
    # Formatação aprimorada para visualização
    status_message = (
        f"{current_date} {current_time} [STATUS]: {message} - MEM: {memory_usage()}"
    )
    
    print(status_message)
    logging.info(status_message)


print_status(f"[INÍCIO DO PROCESSAMENTO]\n")
start_time = time.time()  # Iniciar o contador de tempo total

try:
    # Carregar dados de treino
    print_status("Etapa 1 - Carregar dados de treino")
    df = pd.read_csv('', sep=" ", header=0) # Inserir dataset de treino

    # Carregar a nuvem de pontos não classificada
    print_status("Etapa 2 - Carregar nuvem de pontos não classificada")
    df_unclassified = pd.read_csv('', sep=" ", header=0) # Inserir dataset de teste

    # Mapear a coluna 'Classification' para os valores desejados
    print_status("Etapa 3 - Mapear a coluna Classification")
    mapping = {2: 'ground', 6: 'buildings', 5: 'vegetation'} # testando não usar a camada do ground original
    df['Scalar_field'] = df['Scalar_field'].map(mapping)
    df['Scalar_field'].fillna('others', inplace=True)

    # Dividir os dados em conjuntos de treinamento e teste
    print_status("Etapa 4 - Separar features e variável de saída")
    X_train, X_test, y_train, y_test = train_test_split(
        df[['X', 'Y', 'Z', 'R', 'G', 'B']], 
        df["Scalar_field"], test_size=0.2, random_state=42
    )

    # Normalizar os dados de treinamento e teste
    print_status("Etapa 5 - Verificar balanceamento das classes")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Criar e treinar o modelo nos dados normalizados
    print_status("Etapa 6 - Treinar modelos")
    start_train = time.time()
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train_scaled, y_train)
    end_train = time.time()

    print(f"Tempo de execução: {end_train - start_train:.2f} segundos")
    logging.info(f"Tempo de execução: {end_train - start_train:.2f} segundos")
    print_status("Treinamento dos modelos concluído!")

    # Fazer previsões no conjunto de teste normalizado
    y_pred = model.predict(X_test_scaled)

    # Calcular a matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)
    print_status("Matriz de Confusão:")
    print_status(conf_matrix)

    # Imprimir um relatório de classificação
    print_status("\nRelatório de Classificação:")
    print_status(classification_report(y_test, y_pred))

    # Salvar o modelo treinado - caso queira salvar, tirar o comentário das linhas abaixo
    #with open('model.pkl', 'wb') as f:
    #    pickle.dump(model, f)

    # Normalizar os dados não classificados
    print_status("Etapa 8 - Normalizar dados não classificados")
    df_unclassified_scaled = scaler.transform(df_unclassified[['X', 'Y', 'Z', 'R', 'G', 'B']])

    # Usar o modelo para fazer previsões nos dados não classificados
    print_status("Etapa 9 - Fazer previsões nos dados não classificados")
    predictions = model.predict(df_unclassified_scaled)

    # Mapear as previsões de volta para os números correspondentes
    inverse_mapping = {'buildings': 6, 'vegetation': 5, 'ground':2}
    predictions_mapped = [inverse_mapping[pred] for pred in predictions]

    # Adicionar as previsões como uma nova coluna aos dados não classificados
    df_unclassified['Scalar_field'] = predictions_mapped

    # Salvar os dados não classificados agora classificados em um novo arquivo .xyz
    print_status("Etapa 10 - Salvar dados classificados")
    df_unclassified.to_csv('classified_RF.xyz', sep=" ", index=False, header=True)
    
    # Imprimir o tempo de execução
    print("Tempo de execução: %s segundos" % (time.time() - start_time))


except Exception as e:
    
    # Obtém a data e hora atuais
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    
    # Formatação aprimorada para visualização
    status_message = (
        f"==============================================\n"
        f"[ERRO]: {current_date} {current_time}\n"
        f"MENSAGEM: {str(e)}\n"
        f"=============================================="
    )
    
    print(status_message)
    logging.error(status_message)

finally:
    # Calcular o tempo total de execução
    total_time_seconds = time.time() - start_time
    total_time_hours = total_time_seconds / 3600
    print(f"Tempo total de execução: {total_time_hours:.2f} horas")
