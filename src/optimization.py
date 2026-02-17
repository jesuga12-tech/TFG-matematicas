"""
Módulo de optimización de hiperparámetros.

Contiene:
- Optimización Bayesiana (Optuna)
- Random Search
- Algoritmo Genético (GA)
- Particle Swarm Optimization (PSO)
- Funciones auxiliares para logging
"""

import random
import numpy as np
from datetime import datetime
import optuna
import tensorflow as tf

from models import build_tcn_model, build_cnn_model, build_lstm_model

__all__ = [
    "train_model_quick",
    "bayesian_optimization",
    "random_search",
    "genetic_search",
    "pso_search",
    "formatear_tiempo",
    "registrar_resultado"
]

# ============================================
# OPTIMIZACIÓN BAYESIANA CON OPTUNA
# ============================================

def train_model_quick(model_type, hparams, input_shape, n_stores, n_items,
                      X_train, y_train, X_val, y_val,
                      X_store_train, X_item_train, X_store_val, X_item_val,
                      max_epochs=15, batch_size=32):
    """
    Entrena un modelo rápidamente para optimización.
    """
    # Construir modelo
    if model_type == "tcn":
        model = build_tcn_model(input_shape, hparams, n_stores, n_items)
    elif model_type == "cnn":
        model = build_cnn_model(input_shape, hparams, n_stores, n_items)
    elif model_type == "lstm":
        model = build_lstm_model(input_shape, hparams, n_stores, n_items)
    else:
        raise ValueError("model_type debe ser 'tcn', 'cnn' o 'lstm'")
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", 
        patience=3, 
        restore_best_weights=True, 
        verbose=0
    )
    model.fit(
        [X_train, X_store_train, X_item_train], y_train,
        validation_data=([X_val, X_store_val, X_item_val], y_val),
        epochs=max_epochs, 
        batch_size=batch_size, 
        callbacks=[early_stop], 
        verbose=0
    )
    
    # Evaluar (val_msle, val_rmsle, val_rmse)
    val_msle, val_rmsle, val_rmse = model.evaluate(
        [X_val, X_store_val, X_item_val], y_val, verbose=0
    )
    return val_msle, val_rmsle, val_rmse, model


def bayesian_optimization(model_type, input_shape, n_stores, n_items,
                          X_train, y_train, X_val, y_val,
                          X_store_train, X_item_train, X_store_val, X_item_val,
                          n_trials=20, max_epochs=15, batch_size=32):
    """Optimización bayesiana para TCN, CNN o LSTM. Optimiza por RMSLE."""
    best_bo_val_rmsle = np.inf
    best_bo_model = None
    best_bo_hparams = None
    best_bo_val_rmse = None
    
    def objective(trial):
        nonlocal best_bo_val_rmsle, best_bo_model, best_bo_hparams, best_bo_val_rmse
        
        # Definir hiperparámetros según modelo
        if model_type == "tcn":
            hparams = {
                "filters": trial.suggest_categorical("filters", [16, 32, 64, 128]),
                "kernel_size": trial.suggest_categorical("kernel_size", [2, 3, 4, 5]),
                "num_blocks": trial.suggest_int("num_blocks", 2, 5),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            }
        elif model_type == "cnn":
            hparams = {
                "filters": trial.suggest_categorical("filters", [16, 32, 64, 128]),
                "kernel_size": trial.suggest_categorical("kernel_size", [2, 3, 4, 5, 6, 7]),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            }
        elif model_type == "lstm":
            hparams = {
                "lstm_units": trial.suggest_int("lstm_units", 32, 256, step=32),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            }
        else:
            raise ValueError("model_type debe ser 'tcn', 'cnn' o 'lstm'")
        
        # Entrenar y evaluar
        val_msle, val_rmsle, val_rmse, model = train_model_quick(
            model_type, hparams, input_shape, n_stores, n_items,
            X_train, y_train, X_val, y_val,
            X_store_train, X_item_train, X_store_val, X_item_val,
            max_epochs=max_epochs, batch_size=batch_size
        )
        
        # Guardar mejor modelo (optimizando por RMSLE)
        if val_rmsle < best_bo_val_rmsle:
            best_bo_val_rmsle = val_rmsle
            best_bo_val_rmse = val_rmse
            best_bo_model = model
            best_bo_hparams = hparams
        
        return val_rmsle
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n===== FIN OPTIMIZACIÓN {model_type.upper()} =====")
    print(f"Mejor val_rmsle: {best_bo_val_rmsle:.4f}")
    print("Mejores hiperparámetros:")
    for k, v in best_bo_hparams.items():
        print(f"  {k}: {v}")
    print("="*60)
    
    return best_bo_hparams, best_bo_val_rmsle, best_bo_val_rmsle, best_bo_val_rmse, best_bo_model


# ============================================
# RANDOM SEARCH
# ============================================

def random_search(model_type, input_shape, n_stores, n_items,
                  X_train, y_train, X_val, y_val,
                  X_store_train, X_item_train, X_store_val, X_item_val,
                  n_trials=20, max_epochs=15, batch_size=32):
    """
    Aplica Random Search sobre los hiperparámetros de TCN, CNN o LSTM.
    Optimiza por RMSLE.
    """
    best_hparams = None
    best_val_rmsle = np.inf
    best_val_rmse = None
    best_model = None
    
    for trial in range(1, n_trials + 1):
        print(f"\n========== RANDOM SEARCH {model_type.upper()} - TRIAL {trial}/{n_trials} ==========")
        
        # Generar hiperparámetros aleatorios según modelo
        if model_type == "tcn":
            hparams = {
                "filters": random.choice([16, 32, 64, 128]),
                "kernel_size": random.choice([2, 3, 4, 5]),
                "num_blocks": random.randint(2, 5),
                "lr" : 10 ** random.uniform(-4, -2)  
            }
        elif model_type == "cnn":
            hparams = {
                "filters": random.choice([16, 32, 64, 128]),
                "kernel_size": random.choice([2, 3, 4, 5, 6, 7]),
                "lr" : 10 ** random.uniform(-4, -2)  
            }
        elif model_type == "lstm":
            hparams = {
                "lstm_units": random.randint(32, 256),
                "dropout": random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
                "lr" : 10 ** random.uniform(-4, -2)  
            }
        else:
            raise ValueError("model_type debe ser 'tcn', 'cnn' o 'lstm'")
        
        print(f"Hiperparámetros: {hparams}")
        
        # Entrenar y evaluar
        val_msle, val_rmsle, val_rmse, model = train_model_quick(
            model_type, hparams, input_shape, n_stores, n_items,
            X_train, y_train, X_val, y_val,
            X_store_train, X_item_train, X_store_val, X_item_val,
            max_epochs=max_epochs, batch_size=batch_size
        )
        
        print(f"Val MSLE: {val_msle:.4f}, Val RMSLE: {val_rmsle:.4f}, Val RMSE: {val_rmse:.4f}")
        
        # Optimizar por RMSLE
        if val_rmsle < best_val_rmsle:
            print(">>> Nueva mejor configuración (Random Search).")
            best_val_rmsle = val_rmsle
            best_val_rmse = val_rmse
            best_hparams = hparams
            best_model = model
    
    print(f"\n===== FIN RANDOM SEARCH {model_type.upper()} =====")
    print(f"Mejor val_rmsle: {best_val_rmsle:.4f}")
    print("Mejores hiperparámetros:")
    for k, v in best_hparams.items():
        print(f"  {k}: {v}")
    print("="*60)
    
    return best_hparams, best_val_rmsle, best_val_rmsle, best_val_rmse, best_model


# ============================================
# ALGORITMO GENÉTICO (GA)
# ============================================

def decode_individual(ind, model_type):
    """
    Decodifica un individuo (vector en [0,1]) a hiperparámetros según el modelo.
    """
    hparams = {}
    
    if model_type == "tcn":
        
        filters_choices = [16, 32, 64, 128]
        idx_filters = int(np.floor(ind[0] * len(filters_choices)))
        idx_filters = max(0, min(idx_filters, len(filters_choices) - 1))
        hparams["filters"] = filters_choices[idx_filters]
        
        kernel_choices = [2, 3, 4, 5]
        idx_kernel = int(np.floor(ind[1] * len(kernel_choices)))
        idx_kernel = max(0, min(idx_kernel, len(kernel_choices) - 1))
        hparams["kernel_size"] = kernel_choices[idx_kernel]
        
        hparams["num_blocks"] = int(2 + np.floor(ind[2] * 4))
        hparams["num_blocks"] = max(2, min(hparams["num_blocks"], 5))
        
        log_lr_min, log_lr_max = -4, -2
        log_lr = log_lr_min + (log_lr_max - log_lr_min) * ind[3]
        hparams["lr"] = float(10 ** log_lr)
        
    elif model_type == "cnn":
        filters_choices = [16, 32, 64, 128]
        idx_filters = int(np.floor(ind[0] * len(filters_choices)))
        idx_filters = max(0, min(idx_filters, len(filters_choices) - 1))
        hparams["filters"] = filters_choices[idx_filters]
        
        kernel_choices = [2, 3, 4, 5, 6, 7]
        idx_kernel = int(np.floor(ind[1] * len(kernel_choices)))
        idx_kernel = max(0, min(idx_kernel, len(kernel_choices) - 1))
        hparams["kernel_size"] = kernel_choices[idx_kernel]
        
        log_lr_min, log_lr_max = -4, -2
        log_lr = log_lr_min + (log_lr_max - log_lr_min) * ind[2]
        hparams["lr"] = float(10 ** log_lr)
        
    elif model_type == "lstm":
        lstm_units_min, lstm_units_max = 32, 256
        lstm_units = int(lstm_units_min + (lstm_units_max - lstm_units_min) * ind[0])
        # Redondear al múltiplo de 32 más cercano
        lstm_units = int(32 * round(lstm_units / 32))
        lstm_units = max(32, min(lstm_units, 256))
        hparams["lstm_units"] = lstm_units
        
        dropout_choices = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        idx_dropout = int(np.floor(ind[1] * len(dropout_choices)))
        idx_dropout = max(0, min(idx_dropout, len(dropout_choices) - 1))
        hparams["dropout"] = dropout_choices[idx_dropout]
        
        log_lr_min, log_lr_max = -4, -2
        log_lr = log_lr_min + (log_lr_max - log_lr_min) * ind[2]
        hparams["lr"] = float(10 ** log_lr)
        
    else:
        raise ValueError("model_type debe ser 'tcn', 'cnn' o 'lstm'")
    
    return hparams


def init_population(pop_size, model_type):
    """Crea una población inicial de 'pop_size' individuos."""
    if model_type == "tcn":
        dim = 4
    elif model_type == "cnn":
        dim = 3
    elif model_type == "lstm":
        dim = 3
    else:
        raise ValueError("model_type debe ser 'tcn', 'cnn' o 'lstm'")
    
    return [np.random.rand(dim) for _ in range(pop_size)]


def tournament_selection(population, fitnesses, k=3):
    """Selección por torneo: elige k individuos al azar, devuelve el mejor."""
    idxs = np.random.choice(len(population), size=k, replace=False)
    best_idx = idxs[0]
    for i in idxs[1:]:
        if fitnesses[i] < fitnesses[best_idx]:
            best_idx = i
    return population[best_idx].copy()


def crossover(parent1, parent2, crossover_rate=0.8):
    """Cruce de un punto."""
    dim = len(parent1)
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, dim)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
    else:
        child1 = parent1.copy()
        child2 = parent2.copy()
    return child1, child2


def mutate(ind, mutation_rate=0.2, sigma=0.1):
    """Mutación gaussiana."""
    for i in range(len(ind)):
        if np.random.rand() < mutation_rate:
            ind[i] += np.random.normal(0, sigma)
    ind = np.clip(ind, 0.0, 1.0)
    return ind


def evaluate_individual(ind, model_type, input_shape, n_stores, n_items,
                        X_train, y_train, X_val, y_val,
                        X_store_train, X_item_train, X_store_val, X_item_val,
                        max_epochs=10, batch_size=32):
    """Decodifica el individuo, entrena el modelo y devuelve su fitness."""
    hparams = decode_individual(ind, model_type)
    val_msle, val_rmsle, val_rmse, model = train_model_quick(
        model_type, hparams, input_shape, n_stores, n_items,
        X_train, y_train, X_val, y_val,
        X_store_train, X_item_train, X_store_val, X_item_val,
        max_epochs=max_epochs, batch_size=batch_size
    )
    fitness = val_rmsle  # Optimizar por RMSLE
    return fitness, val_rmsle, val_rmse, model, hparams


def genetic_search(model_type, input_shape, n_stores, n_items,
                   X_train, y_train, X_val, y_val,
                   X_store_train, X_item_train, X_store_val, X_item_val,
                   pop_size=6, n_generations=3, mutation_rate=0.1,
                   crossover_rate=0.8, tournament_k=3, 
                   max_epochs=15, batch_size=32):
    """Aplica un algoritmo genético para optimizar TCN, CNN o LSTM."""
    population = init_population(pop_size, model_type)
    fitnesses = [None] * pop_size

    best_ind = None
    best_fitness = np.inf
    best_val_rmsle = None
    best_val_rmse = None
    best_hparams = None
    best_model = None

    for gen in range(n_generations):
        print(f"\n===== GENERACIÓN {gen+1}/{n_generations} (GA {model_type.upper()}) =====")

        # Evaluar población
        for i in range(pop_size):
            if fitnesses[i] is None:
                print(f"\nEvaluando individuo {i+1}/{pop_size} de la generación {gen+1}")
                fitness, val_rmsle, val_rmse, model, hparams = evaluate_individual(
                    population[i], model_type, input_shape, n_stores, n_items,
                    X_train, y_train, X_val, y_val,
                    X_store_train, X_item_train, X_store_val, X_item_val,
                    max_epochs=max_epochs, batch_size=batch_size
                )
                fitnesses[i] = fitness

                if fitness < best_fitness:
                    print(">>> Nuevo mejor individuo (GA).")
                    best_fitness = fitness
                    best_val_rmsle = val_rmsle
                    best_val_rmse = val_rmse
                    best_ind = population[i].copy()
                    best_hparams = hparams
                    best_model = model

        # Crear nueva población (elitismo + cruce + mutación)
        new_population = []
        new_fitnesses = []

        # Elitismo: conservar el mejor
        new_population.append(best_ind.copy())
        new_fitnesses.append(best_fitness)

        # Rellenar el resto
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, k=tournament_k)
            parent2 = tournament_selection(population, fitnesses, k=tournament_k)

            child1, child2 = crossover(parent1, parent2, crossover_rate)
            child1 = mutate(child1, mutation_rate=mutation_rate)
            child2 = mutate(child2, mutation_rate=mutation_rate)

            new_population.append(child1)
            new_fitnesses.append(None)

            if len(new_population) < pop_size:
                new_population.append(child2)
                new_fitnesses.append(None)

        population = new_population
        fitnesses = new_fitnesses

    print(f"\n===== FIN DEL GA {model_type.upper()} =====")
    print(f"Mejor val_rmsle: {best_fitness:.4f}")
    print("Mejores hiperparámetros:")
    for k, v in best_hparams.items():
        print(f"  {k}: {v}")
    print("="*60)

    return best_hparams, best_fitness, best_val_rmsle, best_val_rmse, best_model


# ============================================
# PSO (PARTICLE SWARM OPTIMIZATION)
# ============================================

def pso_search(model_type, input_shape, n_stores, n_items,
               X_train, y_train, X_val, y_val,
               X_store_train, X_item_train, X_store_val, X_item_val,
               n_particles=6, n_iterations=5, max_epochs=15,
               batch_size=32, w=0.7, c1=1.5, c2=1.5):
    """PSO sencillo para optimizar TCN, CNN o LSTM."""
    # Determinar dimensión según modelo
    if model_type == "tcn":
        dim = 4
    elif model_type == "cnn":
        dim = 3
    elif model_type == "lstm":
        dim = 3
    else:
        raise ValueError("model_type debe ser 'tcn', 'cnn' o 'lstm'")
    
    # Inicializar posiciones y velocidades
    pos = np.random.rand(n_particles, dim)
    vel = np.zeros((n_particles, dim))

    # Mejor posición personal y global
    p_best_pos = pos.copy()
    p_best_fit = np.array([np.inf] * n_particles)

    g_best_pos = None
    g_best_fit = np.inf
    g_best_val_rmsle = None
    g_best_val_rmse = None
    g_best_hparams = None
    g_best_model = None

    for it in range(n_iterations):
        print(f"\n===== ITERACIÓN PSO {model_type.upper()} {it+1}/{n_iterations} =====")

        for i in range(n_particles):
            # Evaluar partícula i
            print(f"\nEvaluando partícula {i+1}/{n_particles} en iteración {it+1}")
            fitness, val_rmsle, val_rmse, model, hparams = evaluate_individual(
                pos[i], model_type, input_shape, n_stores, n_items,
                X_train, y_train, X_val, y_val,
                X_store_train, X_item_train, X_store_val, X_item_val,
                max_epochs=max_epochs, batch_size=batch_size
            )

            # Actualizar mejor personal
            if fitness < p_best_fit[i]:
                p_best_fit[i] = fitness
                p_best_pos[i] = pos[i].copy()

            # Actualizar mejor global
            if fitness < g_best_fit:
                print(">>> Nueva mejor partícula (PSO).")
                g_best_fit = fitness
                g_best_val_rmsle = val_rmsle
                g_best_val_rmse = val_rmse
                g_best_pos = pos[i].copy()
                g_best_hparams = hparams
                g_best_model = model

        # Actualizar velocidades y posiciones
        for i in range(n_particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            vel[i] = (
                w * vel[i]
                + c1 * r1 * (p_best_pos[i] - pos[i])
                + c2 * r2 * (g_best_pos - pos[i]))
            pos[i] = pos[i] + vel[i]
            pos[i] = np.clip(pos[i], 0.0, 1.0)

    print(f"\n===== FIN DEL PSO {model_type.upper()} =====")
    print(f"Mejor val_rmsle: {g_best_fit:.4f}")
    print("Mejores hiperparámetros:")
    for k, v in g_best_hparams.items():
        print(f"  {k}: {v}")
    print("="*60)

    return g_best_hparams, g_best_fit, g_best_val_rmsle, g_best_val_rmse, g_best_model


# ============================================
# FUNCIONES AUXILIARES PARA LOGGING
# ============================================

def formatear_tiempo(tiempo_delta):
    """Convierte un timedelta a formato legible (minutos y segundos)."""
    total_segundos = int(tiempo_delta.total_seconds())
    minutos = total_segundos // 60
    segundos = total_segundos % 60
    if minutos > 0:
        return f"{minutos}' {segundos}''"
    else:
        return f"{segundos}''"


def registrar_resultado(resultados_file, algoritmo, modelo, rmsle, rmse, hparams, tiempo_transcurrido, hora_fin):
    """Registra un resultado de optimización con el formato solicitado."""
    tiempo_str = formatear_tiempo(tiempo_transcurrido)
    hora_str = hora_fin.strftime('%H:%M')
    
    resultados_file.write(f"{algoritmo} {modelo.upper()} - RMSLE: {rmsle:.4f}, RMSE: {rmse:.4f}, "
                         f"hiperparámetros: {hparams}, tiempo: {tiempo_str}, hora: {hora_str}\n")

