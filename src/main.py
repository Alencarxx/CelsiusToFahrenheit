# -*- coding: utf-8 -*-
"""
Projeto: Conversão Celsius para Fahrenheit com Rede Neural
Descrição:
    Este projeto utiliza uma rede neural simples em TensorFlow para aprender a equação de conversão de temperaturas.
    Equação: T(°F) = T(°C) × 9/5 + 32
    
Autor: Alencar Porto
Data: 03/01/2025
"""

# Importação das bibliotecas necessárias
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Função para carregar e explorar os dados
def load_and_explore_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Dados carregados com sucesso do arquivo: {file_path}")
        logging.info(f"Resumo dos dados: \n{data.info()}")
        return data
    except Exception as e:
        logging.error(f"Erro ao carregar os dados: {e}")
        raise

# Função para visualizar os dados
def visualize_data(data):
    sns.scatterplot(x=data['Celsius'], y=data['Fahrenheit'])
    plt.title('Celsius vs Fahrenheit')
    plt.xlabel('Celsius')
    plt.ylabel('Fahrenheit')
    plt.show()

# Função para criar e treinar o modelo
def train_model(X, y, learning_rate=0.1, epochs=500):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mean_squared_error')
    history = model.fit(X, y, epochs=epochs, verbose=0)
    logging.info("Treinamento concluído")
    return model, history

# Função para avaliar o modelo
def evaluate_model(history):
    plt.plot(history.history['loss'])
    plt.title('Progresso da Perda do Modelo Durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend(['Perda de Treinamento'])
    plt.show()

# Função para realizar previsões
def predict_temperature(model, temp_c):
    temp_f = model.predict(np.array([temp_c]))
    return temp_f[0][0]

# Execução do projeto
if __name__ == "__main__":
    # Caminho para o arquivo CSV
    file_path = '../data/Celsius-to-Fahrenheit.csv'

    # Etapa 1: Carregar e explorar os dados
    temperature_data = load_and_explore_data(file_path)
    visualize_data(temperature_data)

    # Etapa 2: Configuração dos dados
    X_train = temperature_data['Celsius']
    y_train = temperature_data['Fahrenheit']

    # Etapa 3: Treinamento do modelo
    model, training_history = train_model(X_train, y_train)

    # Etapa 4: Avaliação do modelo
    evaluate_model(training_history)

    # Etapa 5: Previsão
    test_temp_c = 10  # exemplo de teste
    predicted_temp_f = predict_temperature(model, test_temp_c)
    logging.info(f"Temperatura prevista para {test_temp_c}°C: {predicted_temp_f:.2f}°F")
