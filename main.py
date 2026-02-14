"""
Script principal para optimización de hiperparámetros en modelos de deep learning
para la predicción de series temporales en el problema 'Favorita Grocery Sales Forecasting'

Este script:
- Carga y preprocesa datos
- Crea ventanas temporales
- Entrena modelos básicos (RNN, CNN, TCN)
- Aplica algoritmos de optimización (Bayesian, Random Search, GA, PSO)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime

from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman


from proyecto_final import *


# =========================================
# CONFIGURACIÓN BÁSICA
# =========================================

# Semilla para reproducibilidad
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# =========================================
# CARGA Y PREPROCESADO DE DATOS
# =========================================

print("="*80)
print("CARGA Y PREPROCESADO DE DATOS")
print("="*80)

# Cargar datos
series_df = load_and_prepare_data_global('series_tienda47_seleccionados.csv', max_rows=None)

# Calcular estadísticas de cada serie
print("\n=== ESTADÍSTICAS DE CADA SERIE (MEDIA Y STD DE 'y' EN LOG SCALE) ===")
grouped_stats = series_df.groupby(['store_nbr', 'item_nbr'])['y'].agg(['mean', 'std', 'count'])
print(grouped_stats.head(10))
print(f"Total de series: {len(grouped_stats)}")

# Crear ventanas temporales
print("\n=== CREACIÓN DE VENTANAS TEMPORALES ===")
X_seq, X_store, X_item, y, dates = create_windows_per_series(series_df, window_size=30)

# Crear mapeos para embeddings
print("\n=== MAPEOS PARA EMBEDDINGS ===")
n_stores, n_items, X_store, X_item = create_embedding_mappings(X_store, X_item)

# Escalar y dividir en train/val/test
print("\n=== ESCALADO Y DIVISIÓN TRAIN/VAL/TEST ===")
(X_train, y_train, X_store_train, X_item_train, X_val,
 y_val, X_store_val, X_item_val, X_test, y_test,
 X_store_test, X_item_test, scaler) = scale_and_split(
     X_seq, y, dates, X_store, X_item,
     train_size=0.6, test_size=0.2, val_size=0.2
 )


# =========================================
# ENTRENAR MODELOS BÁSICOS (RNN, CNN, TCN)
# =========================================

print("\n" + "="*80)
print("ENTRENANDO MODELOS BÁSICOS (RNN, CNN, TCN)")
print("="*80)

input_shape = X_train.shape[1:]

# RNN con hiperparámetros fijos
rnn_hparams = {
    "lstm_units": 64,
    "dense_units": 32,
    "dropout": 0.2,
    "lr": 1e-3
}
rnn_model = build_rnn_model(input_shape, rnn_hparams, n_stores, n_items)
rnn_results = train_and_evaluate(
    rnn_model, "RNN (LSTM)",
    [X_train, X_store_train, X_item_train], y_train,
    [X_val, X_store_val, X_item_val], y_val,
    [X_test, X_store_test, X_item_test], y_test
)

# CNN con hiperparámetros fijos
cnn_hparams = {
    "filters": 32,
    "kernel_size": 3,
    "dense_units": 32,
    "lr": 1e-3
}
cnn_model = build_cnn_model(input_shape, cnn_hparams, n_stores, n_items)
cnn_results = train_and_evaluate(
    cnn_model, "CNN 1D",
    [X_train, X_store_train, X_item_train], y_train,
    [X_val, X_store_val, X_item_val], y_val,
    [X_test, X_store_test, X_item_test], y_test
)

# TCN con hiperparámetros fijos
tcn_hparams = {
    "filters": 32,
    "kernel_size": 3,
    "num_blocks": 3,
    "dropout": 0.1,
    "dense_units": 32,
    "lr": 1e-3
}
tcn_fixed_model = build_tcn_model(input_shape, tcn_hparams, n_stores, n_items)
tcn_fixed_results = train_and_evaluate(
    tcn_fixed_model, "TCN ",
    [X_train, X_store_train, X_item_train], y_train,
    [X_val, X_store_val, X_item_val], y_val,
    [X_test, X_store_test, X_item_test], y_test
)

print("\nResumen modelos básicos (MSLE, RMSLE, RMSE) en test:")
print(f"  RNN (LSTM): {rnn_results}")
print(f"  CNN 1D:     {cnn_results}")
print(f"  TCN :   {tcn_fixed_results}")

# =========================================
# PREDICCIÓN NAIVE 
# =========================================

print("\n === PREDICCIÓN NAIVE ===")


# Predicción naive: usar el  valor del día anterior como predicción
y_naive_pred = X_test[:, -1, 0]  # Último valor de 'y' en cada ventana de test

# Calcular métricas para naive
msle_diario_naive = tf.keras.metrics.MeanSquaredError()(y_test, y_naive_pred)
rmsle_diario_naive = tf.keras.metrics.RootMeanSquaredError()(y_test, y_naive_pred)
rmse_diario_naive = rmse(y_test, y_naive_pred)

print(f"Predicción Naive usando como predicción el valor del día anterior - MSLE: {msle_diario_naive:.4f}, RMSLE: {rmsle_diario_naive:.4f}, RMSE: {rmse_diario_naive:.4f}")


# Predicción naive semanal: usar el valor de la semana anterior (7 días atrás) como predicción
y_seasonal_pred = X_test[:, -8, 0]  # Valor de 'y' 7 días atrás

# Calcular métricas para naive semanal
msle_semanal_naive = tf.keras.metrics.MeanSquaredError()(y_test, y_seasonal_pred)
rmsle_semanal_naive = tf.keras.metrics.RootMeanSquaredError()(y_test, y_seasonal_pred)
rmse_semanal_naive   = rmse(y_test, y_seasonal_pred)

print(f"Predicción Naive usando como predicción el valor de la semana anterior - MSLE: {msle_semanal_naive:.4f}, RMSLE: {rmsle_semanal_naive:.4f}, RMSE: {rmse_semanal_naive:.4f}\n")


# ============================================
# OPTIMIZACIÓN DE HIPERPARÁMETROS
# ============================================
# Todas las llamadas de optimización están aquí al final del archivo.
# Descomentar para ejecutar.

modelos = ["cnn", "lstm", "tcn"]  


# Abrir archivo para guardar resultados
resultados_file = open('resultados_optimizacion.txt', 'w', encoding='utf-8')
resultados_file.write("="*80 + "\n")
resultados_file.write("RESULTADOS DE OPTIMIZACIÓN DE HIPERPARÁMETROS\n")
resultados_file.write(f"Fecha de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
resultados_file.write("="*80 + "\n\n")

# # ============================================
# # 1. OPTIMIZACIÓN BAYESIANA (OPTUNA)
# # ============================================

resultados_file.write("\n" + "="*80 + "\n")
resultados_file.write("1. OPTIMIZACIÓN BAYESIANA (OPTUNA)\n")
resultados_file.write("="*80 + "\n\n")

for modelo in modelos:
    resultados_file.write(f"Optimizacion Bayesiana - {modelo}\n")
    tiempo_inicio = datetime.now()
    bo_hparams, bo_val_loss, bo_val_rmse, bo_val_rmse_real, bo_model = bayesian_optimization(
        modelo, input_shape, n_stores, n_items,
        X_train, y_train, X_val, y_val,
        X_store_train, X_item_train, X_store_val, X_item_val,
        n_trials=40, max_epochs=15, batch_size=32
    )
    tiempo_fin = datetime.now()
    tiempo_transcurrido = tiempo_fin - tiempo_inicio
    registrar_resultado(resultados_file, "BO", modelo, bo_val_rmse, bo_val_rmse_real,
                        bo_hparams, tiempo_transcurrido, tiempo_fin)
    resultados_file.write("\n")


# # ============================================
# # 2. RANDOM SEARCH
# # ============================================

resultados_file.write("\n" + "="*80 + "\n")
resultados_file.write("2. RANDOM SEARCH\n")
resultados_file.write("="*80 + "\n\n")

for modelo in modelos:
    resultados_file.write(f"Random Search - {modelo}\n")
    tiempo_inicio = datetime.now()
    rs_hparams, rss_val_loss, rs_val_rmse, rs_val_rmse_real, rs_model = random_search(
        modelo, input_shape, n_stores, n_items,
        X_train, y_train, X_val, y_val,
        X_store_train, X_item_train, X_store_val, X_item_val,
        n_trials=40, max_epochs=15, batch_size=32
    )
    tiempo_fin = datetime.now()
    tiempo_transcurrido = tiempo_fin - tiempo_inicio
    registrar_resultado(resultados_file, "RS", modelo, rs_val_rmse, rs_val_rmse_real, 
                        rs_hparams, tiempo_transcurrido, tiempo_fin)
    resultados_file.write("\n")


# # ============================================
# # 3. ALGORITMO GENÉTICO (GA)
# # ============================================
# # Utilizamos los parametros del GA obtenidos en optimization_GA_PSO.py

resultados_file.write("\n" + "="*80 + "\n")
resultados_file.write("3. ALGORITMO GENÉTICO (GA)\n")
resultados_file.write("="*80 + "\n\n")

for modelo in modelos:
    resultados_file.write(f"Genetic Algorithm - {modelo}\n")
    tiempo_inicio = datetime.now()
    tiempo_inicio = datetime.now()
    hparams, val_msle, val_rmsle, val_rmse, model = genetic_search(
            modelo, input_shape, n_stores, n_items,
            X_train, y_train, X_val, y_val,
            X_store_train, X_item_train, X_store_val, X_item_val,
            pop_size=12,
            n_generations=10,
            crossover_rate=0.7,
            mutation_rate=0.05,
            tournament_k=3, max_epochs=15, batch_size=32
        )
    tiempo_fin = datetime.now()
    tiempo_transcurrido = tiempo_fin - tiempo_inicio
    registrar_resultado(resultados_file, "GA", modelo, val_rmsle, val_rmse, 
                      hparams, tiempo_transcurrido, tiempo_fin)
    resultados_file.write("\n")

# # ============================================
# # 4. PSO (Particle Swarm Optimization)
# # ============================================
# # Utilizamos los parametros del PSO obtenidos en optimization_GA_PSO.py

resultados_file.write("\n" + "="*80 + "\n")
resultados_file.write("4. PARTICLE SWARM OPTIMIZATION (PSO)\n")
resultados_file.write("="*80 + "\n\n")

for modelo in modelos:
    resultados_file.write(f"Particle Swarm Optimization - {modelo}\n")
    tiempo_inicio = datetime.now()
    tiempo_inicio = datetime.now()
    hparams, val_loss, val_rmsle, val_rmse, model = pso_search(
            modelo, input_shape, n_stores, n_items,
            X_train, y_train, X_val, y_val,
            X_store_train, X_item_train, X_store_val, X_item_val,
            n_particles=15,
            n_iterations=10,
            max_epochs=15,
            batch_size=32,
            c1=1.5,
            c2=1.5,
            w=0.7,
        )
    tiempo_fin = datetime.now()
    tiempo_transcurrido = tiempo_fin - tiempo_inicio
    registrar_resultado(resultados_file, "PSO", modelo, val_rmsle, val_rmse, 
                      hparams, tiempo_transcurrido, tiempo_fin)
    resultados_file.write("\n")

# Cerrar archivo de resultados
resultados_file.write("\n" + "="*80 + "\n")
resultados_file.write(f"Fecha de finalización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
resultados_file.write("="*80 + "\n")
resultados_file.close()
print("\n✓ Resultados guardados en 'resultados_optimizacion.txt'")


cnn_hparams = {

    "BO": {
        "filters": 64,
        "kernel_size": 7,
        "dense_units": 32,
        "lr": 0.0045397
    },

    "RS": {
        "filters": 16,
        "kernel_size": 5,
        "dense_units": 32,
        "lr": 0.0084325
    },

    "GA": {
        "filters": 128,
        "kernel_size": 7,
        "dense_units": 32,
        "lr": 0.0041380
    },

    "PSO": {
        "filters": 128,
        "kernel_size": 7,
        "dense_units": 32,
        "lr": 0.0009070
    }
}

lstm_hparams = {

    "BO": {
        "lstm_units": 96,
        "dropout": 0.5,
        "dense_units": 32,
        "lr": 0.0041919
    },

    "RS": {
        "lstm_units": 80,
        "dropout": 0.4,
        "dense_units": 32,
        "lr": 0.0045413
    },

    "GA": {
        "lstm_units": 192,
        "dropout": 0.1,
        "dense_units": 32,
        "lr": 0.0095761
    },

    "PSO": {
        "lstm_units": 160,
        "dropout": 0.0,
        "dense_units": 32,
        "lr": 0.0100
    }
}

tcn_hparams = {

    "BO": {
        "filters": 64,
        "kernel_size": 3,
        "num_blocks": 4,
        "dropout": 0.1,
        "dense_units": 32,
        "lr": 0.0038562
    },

    "RS": {
        "filters": 32,
        "kernel_size": 5,
        "num_blocks": 4,
        "dropout": 0.1,
        "dense_units": 32,
        "lr": 0.0016997
    },

    "GA": {
        "filters": 16,
        "kernel_size": 5,
        "num_blocks": 2,
        "dropout": 0.1,
        "dense_units": 32,
        "lr": 0.0062938
    },

    "PSO": {
        "filters": 128,
        "kernel_size": 3,
        "num_blocks": 4,
        "dropout": 0.1,
        "dense_units": 32,
        "lr": 0.0005509
    }
}

# # ============================================
# # EVALUACIÓN DE MODELOS OPTIMIZADOS EN TEST
# # ============================================

print("\n" + "="*80)
print("EVALUACIÓN DE MODELOS OPTIMIZADOS EN TEST")
print("="*80)

# Almacenar resultados
resultados_optimizados = []

# Métodos de optimización
metodos = ["BO", "RS", "GA", "PSO"]

# Evaluar CNN con cada método
print("\n--- Evaluando CNN ---")
for metodo in metodos:
    print(f"  CNN - {metodo}...")
    hparams = cnn_hparams[metodo]
    model = build_cnn_model(input_shape, hparams, n_stores, n_items)
    test_msle, test_rmsle, test_rmse = train_and_evaluate(
        model, f"CNN - {metodo}",
        [X_train, X_store_train, X_item_train], y_train,
        [X_val, X_store_val, X_item_val], y_val,
        [X_test, X_store_test, X_item_test], y_test
    )
    resultados_optimizados.append({
        'modelo': 'CNN',
        'metodo': metodo,
        'msle': test_msle,
        'rmsle': test_rmsle,
        'rmse': test_rmse
    })

# Evaluar LSTM con cada método
print("\n--- Evaluando LSTM ---")
for metodo in metodos:
    print(f"  LSTM - {metodo}...")
    hparams = lstm_hparams[metodo]
    model = build_rnn_model(input_shape, hparams, n_stores, n_items)
    test_msle, test_rmsle, test_rmse = train_and_evaluate(
        model, f"LSTM - {metodo}",
        [X_train, X_store_train, X_item_train], y_train,
        [X_val, X_store_val, X_item_val], y_val,
        [X_test, X_store_test, X_item_test], y_test
    )
    resultados_optimizados.append({
        'modelo': 'LSTM',
        'metodo': metodo,
        'msle': test_msle,
        'rmsle': test_rmsle,
        'rmse': test_rmse

    })

# Evaluar TCN con cada método
print("\n--- Evaluando TCN ---")
for metodo in metodos:
    print(f"  TCN - {metodo}...")
    hparams = tcn_hparams[metodo]
    model = build_tcn_model(input_shape, hparams, n_stores, n_items)
    test_msle, test_rmsle, test_rmse = train_and_evaluate(
        model, f"TCN - {metodo}",
        [X_train, X_store_train, X_item_train], y_train,
        [X_val, X_store_val, X_item_val], y_val,
        [X_test, X_store_test, X_item_test], y_test
    )
    resultados_optimizados.append({
        'modelo': 'TCN',
        'metodo': metodo,
        'msle': test_msle,
        'rmsle': test_rmsle,
        'rmse': test_rmse
    })

# Convertir a DataFrame y guardar en Excel
df_resultados_optimizados = pd.DataFrame(resultados_optimizados)
output_file = 'resultados_modelos_optimizados.xlsx'
df_resultados_optimizados.to_excel(output_file, index=False)
print(f"\n Resultados guardados en '{output_file}'")



# ============================================
# EVALUACIÓN ESTADÍSTICA CON MÚLTIPLES SEMILLAS
# ============================================
# Ejecutar cada modelo con 20 semillas diferentes para tests estadísticos (Friedman y Nemenyi)

print("\n" + "="*80)
print("EVALUACIÓN ESTADÍSTICA CON 20 SEMILLAS")
print("="*80)

# Mejores hiperparámetros obtenidos de las optimizaciones
best_hparams = {
    "cnn": {
        "filters": 128,
        "kernel_size": 7,
        "dense_units": 32,
        "lr": 0.00413804
    },
    "lstm": {
        "lstm_units": 192,
        "dropout": 0.1,
        "dense_units": 32,
        "lr": 0.0095761
    },
        
    "tcn": {
        "filters": 32,
        "kernel_size": 5,
        "num_blocks": 4,
        "dropout": 0.1,
        "dense_units": 32,
        "lr": 0.0016997
    }
}

# Semillas para reproducibilidad
RANDOM_SEEDS = list(range(1, 31))  # [1, 2, ..., 20]

# Almacenar resultados
resultados_estadisticos = []

print(f"\nEjecutando {len(modelos)} modelos con {len(RANDOM_SEEDS)} semillas cada uno...")
print(f"Total de ejecuciones: {len(modelos) * len(RANDOM_SEEDS)}")

for seed_idx, seed in enumerate(RANDOM_SEEDS, 1):
    print(f"\n--- Semilla {seed_idx}/{len(RANDOM_SEEDS)} (seed={seed}) ---")
    
    # Establecer semilla
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    
    for modelo in modelos:
        print(f"  Entrenando {modelo}..." )
        
        # Construir modelo con los mejores hiperparámetros
        if modelo == "cnn":
            model = build_cnn_model(input_shape, best_hparams[modelo], n_stores, n_items)
        elif modelo == "lstm":
            model = build_rnn_model(input_shape, best_hparams[modelo], n_stores, n_items)
        elif modelo == "tcn":
            model = build_tcn_model(input_shape, best_hparams[modelo], n_stores, n_items)
        
        # Entrenar y evaluar modelo
        test_msle, test_rmsle, test_rmse = train_and_evaluate(
            model, 
            modelo.upper(),
            [X_train, X_store_train, X_item_train], y_train,
            [X_val, X_store_val, X_item_val], y_val,
            [X_test, X_store_test, X_item_test], y_test
        )
        
        # Guardar resultados
        resultados = {
            'modelo': modelo,
            'semilla': seed,
            'rmsle': test_rmsle,
            'rmse': test_rmse,
            'msle': test_msle
        }
        resultados_estadisticos.append(resultados)
        print(f" RMSLE: {test_rmsle:.4f}, RMSE: {test_rmse:.4f}")
              

# Convertir a DataFrame
df_resultados = pd.DataFrame(resultados_estadisticos)

# Guardar resultados en CSV
output_file = 'resultados_estadisticos_30_semillas.xlsx'
df_resultados.to_excel(output_file, index=False)
print(f"\n Resultados guardados en '{output_file}'")


