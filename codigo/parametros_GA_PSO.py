import os
from datetime import datetime
from optimization import genetic_search, pso_search, registrar_resultado
from preprocessing import load_and_prepare_data_global, create_windows_per_series, create_embedding_mappings, scale_and_split




series_df = load_and_prepare_data_global('series_tienda47_seleccionados.csv', max_rows=None)
X_seq, X_store, X_item, y, dates = create_windows_per_series(series_df, window_size=30)
n_stores, n_items, X_store, X_item = create_embedding_mappings(X_store, X_item)
(X_train, y_train, X_store_train, X_item_train, X_val,
 y_val, X_store_val, X_item_val, X_test, y_test,
 X_store_test, X_item_test, scaler) = scale_and_split(
     X_seq, y, dates, X_store, X_item,
     train_size=0.6, test_size=0.2, val_size=0.2
 )

input_shape = X_train.shape[1:]

modelos = ["cnn", "lstm", "tcn"]  # Deben estar en minúsculas para las funciones
# ============================================
# OPTIMIZACIÓN DE PARARÁMETROS GA
# ============================================

# Abrir archivo para guardar resultados
resultados_file = open('parametros_GA_PSO.txt', 'w', encoding='utf-8')
resultados_file.write("="*80 + "\n")
resultados_file.write("RESULTADOS DE OPTIMIZACIÓN DE HIPERPARÁMETROS\n")
resultados_file.write(f"Fecha de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
resultados_file.write("="*80 + "\n\n")


# Valores por defecto
GA_DEFAULT_POP_SIZE = 16
GA_DEFAULT_N_GENERATIONS = 10
GA_DEFAULT_CROSSOVER_RATE = 0.8
GA_DEFAULT_MUTATION_RATE = 0.1

# Valores a recorrer
pop_sizes = [8, 12, 16, 20]
n_generations = [5, 8, 10, 15, 20]
crossover_rates = [0.6, 0.7, 0.8, 0.9]
mutation_rates = [0, 0.05, 0.1, 0.15, 0.2]

# Modelos a optimizar


resultados_file.write("\n" + "="*80 + "\n")
resultados_file.write("3. ALGORITMO GENÉTICO (GA)\n")
resultados_file.write("="*80 + "\n\n")

# ============================================
# BUCLE 1: Variar pop_size (fijar n_generations, crossover_rate, mutation_rate)
# ============================================
resultados_file.write("\n" + "-"*80 + "\n")
resultados_file.write("BUCLE 1: Variando pop_size (n_generations=10, crossover_rate=0.8, mutation_rate=0.1)\n")
resultados_file.write("-"*80 + "\n\n")

for modelo in modelos:
    for pop_size in pop_sizes:
        resultados_file.write(f"\n--- GA {modelo} - pop_size={pop_size} ---\n")
        tiempo_inicio = datetime.now()
        hparams, val_msle, val_rmsle, val_rmse, model = genetic_search(
            modelo, input_shape, n_stores, n_items,
            X_train, y_train, X_val, y_val,
            X_store_train, X_item_train, X_store_val, X_item_val,
            pop_size=pop_size,
            n_generations=GA_DEFAULT_N_GENERATIONS,
            crossover_rate=GA_DEFAULT_CROSSOVER_RATE,
            mutation_rate=GA_DEFAULT_MUTATION_RATE,
            tournament_k=3, max_epochs=15, batch_size=32
        )
        tiempo_fin = datetime.now()
        tiempo_transcurrido = tiempo_fin - tiempo_inicio
        registrar_resultado(resultados_file, f"GA_pop{pop_size}", modelo, val_rmsle, val_rmse, 
                          hparams, tiempo_transcurrido, tiempo_fin)
        resultados_file.write("\n")



# ============================================
# BUCLE 2: Variar n_generations (fijar pop_size, crossover_rate, mutation_rate)
# ============================================
GA_DEFAULT_POP_SIZE = 12


resultados_file.write("\n" + "-"*80 + "\n")
resultados_file.write("BUCLE 2: Variando n_generations (pop_size=12, crossover_rate=0.8, mutation_rate=0.1)\n")
resultados_file.write("-"*80 + "\n\n")


for modelo in modelos:
    for n_gen in n_generations:
        resultados_file.write(f"\n--- GA {modelo} - n_generations={n_gen} ---\n")
        tiempo_inicio = datetime.now()
        hparams, val_msle, val_rmsle, val_rmse, model = genetic_search(
            modelo, input_shape, n_stores, n_items,
            X_train, y_train, X_val, y_val,
            X_store_train, X_item_train, X_store_val, X_item_val,
            pop_size=GA_DEFAULT_POP_SIZE,
            n_generations=n_gen,
            crossover_rate=GA_DEFAULT_CROSSOVER_RATE,
            mutation_rate=GA_DEFAULT_MUTATION_RATE,
            tournament_k=3, max_epochs=15, batch_size=32
        )
        tiempo_fin = datetime.now()
        tiempo_transcurrido = tiempo_fin - tiempo_inicio
        registrar_resultado(resultados_file, f"GA_gen{n_gen}", modelo, val_rmsle, val_rmse, 
                          hparams, tiempo_transcurrido, tiempo_fin)
        resultados_file.write("\n")




# ============================================
# BUCLE 3: Variar crossover_rate (fijar pop_size, n_generations, mutation_rate)
# ============================================
GAGA_DEFAULT_N_GENERATIONS = 10


resultados_file.write("\n" + "-"*80 + "\n")
resultados_file.write("BUCLE 3: Variando crossover_rate (pop_size=12, n_generations=10, mutation_rate=0.1)\n")
resultados_file.write("-"*80 + "\n\n")

for modelo in modelos:
    for crossover_rate in crossover_rates:
        resultados_file.write(f"\n--- GA {modelo} - crossover_rate={crossover_rate} ---\n")
        tiempo_inicio = datetime.now()
        hparams, val_msle, val_rmsle, val_rmse, model = genetic_search(
            modelo, input_shape, n_stores, n_items,
            X_train, y_train, X_val, y_val,
            X_store_train, X_item_train, X_store_val, X_item_val,
            pop_size=GA_DEFAULT_POP_SIZE,
            n_generations=GA_DEFAULT_N_GENERATIONS,
            crossover_rate=crossover_rate,
            mutation_rate=GA_DEFAULT_MUTATION_RATE,
            tournament_k=3, max_epochs=15, batch_size=32
        )
        tiempo_fin = datetime.now()
        tiempo_transcurrido = tiempo_fin - tiempo_inicio
        registrar_resultado(resultados_file, f"GA_cr{crossover_rate}", modelo, val_rmsle, val_rmse, 
                          hparams, tiempo_transcurrido, tiempo_fin)
        resultados_file.write("\n")



# ============================================
# BUCLE 4: Variar mutation_rate (fijar pop_size, n_generations, crossover_rate)
# ============================================
GA_DEFAULT_CROSSOVER_RATE = 0.7

resultados_file.write("\n" + "-"*80 + "\n")
resultados_file.write("BUCLE 4: Variando mutation_rate (pop_size=12, n_generations=10, crossover_rate=0.7)\n")
resultados_file.write("-"*80 + "\n\n")

for modelo in modelos:
    for mutation_rate in mutation_rates:
        resultados_file.write(f"\n--- GA {modelo} - mutation_rate={mutation_rate} ---\n")
        tiempo_inicio = datetime.now()
        hparams, val_msle, val_rmsle, val_rmse, model = genetic_search(
            modelo, input_shape, n_stores, n_items,
            X_train, y_train, X_val, y_val,
            X_store_train, X_item_train, X_store_val, X_item_val,
            pop_size=GA_DEFAULT_POP_SIZE,
            n_generations=GA_DEFAULT_N_GENERATIONS,
            crossover_rate=GA_DEFAULT_CROSSOVER_RATE,
            mutation_rate=mutation_rate,
            tournament_k=3, max_epochs=15, batch_size=32
        )
        tiempo_fin = datetime.now()
        tiempo_transcurrido = tiempo_fin - tiempo_inicio
        registrar_resultado(resultados_file, f"GA_mr{mutation_rate}", modelo, val_rmsle, val_rmse, 
                          hparams, tiempo_transcurrido, tiempo_fin)
        resultados_file.write("\n")

GA_DEFAULT_MUTATION_RATE = 0.05

print("Los resultados de los parámetros por defecto para GA son: ")
print(f"pop_size={GA_DEFAULT_POP_SIZE}, n_generations={GA_DEFAULT_N_GENERATIONS}, crossover_rate={GA_DEFAULT_CROSSOVER_RATE}, mutation_rate={GA_DEFAULT_MUTATION_RATE}")

# ============================================
# OPTIMIZACION PARÁMETROS PSO 
# ============================================

# Valores por defecto
PSO_DEFAULT_N_PARTICLES = 20
PSO_DEFAULT_N_ITERATIONS = 20
PSO_DEFAULT_C1 = 1.5
PSO_DEFAULT_C2 = 1.5
PSO_DEFAULT_W = 0.7

# Valores a recorrer
particles = [5, 10, 15, 20, 25]
iterations = [5, 10, 15, 20, 30]
c1_c2 = [(1.2, 1.2), (1.5, 1.5), (2.0, 2.0), (1.5, 2.0), (2.0, 1.5)]
w = [0.4, 0.6, 0.7, 0.8, 0.9]



resultados_file.write("\n" + "="*80 + "\n")
resultados_file.write("4. PSO (PARTICLE SWARM OPTIMIZATION)\n")
resultados_file.write("="*80 + "\n\n")

# ============================================
# BUCLE 1: Variar n_particles (fijar n_iterations, c1, c2, w)
# ============================================
resultados_file.write("\n" + "-"*80 + "\n")
resultados_file.write("BUCLE 1: Variando n_particles (n_iterations=30, c1=1.5, c2=1.5, w=0.7)\n")
resultados_file.write("-"*80 + "\n\n")

for modelo in modelos:
    for n_parts in particles:
        resultados_file.write(f"\n--- PSO {modelo} - n_particles={n_parts} ---\n")
        tiempo_inicio = datetime.now()
        hparams, val_loss, val_rmsle, val_rmse, model = pso_search(
            modelo, input_shape, n_stores, n_items,
            X_train, y_train, X_val, y_val,
            X_store_train, X_item_train, X_store_val, X_item_val,
            n_particles=n_parts,
            n_iterations=PSO_DEFAULT_N_ITERATIONS,
            max_epochs=15,
            batch_size=32,
            w=PSO_DEFAULT_W,
            c1=PSO_DEFAULT_C1,
            c2=PSO_DEFAULT_C2
        )
        tiempo_fin = datetime.now()
        tiempo_transcurrido = tiempo_fin - tiempo_inicio
        registrar_resultado(resultados_file, f"PSO_part{n_parts}", modelo, val_rmsle, val_rmse, 
                          hparams, tiempo_transcurrido, tiempo_fin)
        resultados_file.write("\n")


# ============================================
# BUCLE 2: Variar n_iterations (fijar n_particles, c1, c2, w)
# ============================================
PSO_DEFAULT_N_PARTICLES = 15

resultados_file.write("\n" + "-"*80 + "\n")
resultados_file.write("BUCLE 2: Variando n_iterations (n_particles=15, c1=1.5, c2=1.5, w=0.7)\n")
resultados_file.write("-"*80 + "\n\n")
for modelo in modelos:
    for n_iter in iterations:
        resultados_file.write(f"\n--- PSO {modelo} - n_iterations={n_iter} ---\n")
        tiempo_inicio = datetime.now()
        hparams, val_loss, val_rmsle, val_rmse, model = pso_search(
            modelo, input_shape, n_stores, n_items,
            X_train, y_train, X_val, y_val,
            X_store_train, X_item_train, X_store_val, X_item_val,
            n_particles=PSO_DEFAULT_N_PARTICLES,
            n_iterations=n_iter,
            max_epochs=15,
            batch_size=32,
            w=PSO_DEFAULT_W,
            c1=PSO_DEFAULT_C1,
            c2=PSO_DEFAULT_C2
        )
        tiempo_fin = datetime.now()
        tiempo_transcurrido = tiempo_fin - tiempo_inicio
        registrar_resultado(resultados_file, f"PSO_iter{n_iter}", modelo, val_rmsle, val_rmse, 
                          hparams, tiempo_transcurrido, tiempo_fin)
        resultados_file.write("\n")

# ============================================
# BUCLE 3: Variar c1 y c2 (fijar n_particles, n_iterations, w)
# ============================================
PSO_DEFAULT_N_ITERATIONS = 10

resultados_file.write("\n" + "-"*80 + "\n")
resultados_file.write("BUCLE 3: Variando c1 y c2 (n_particles=15, n_iterations=10, w=0.7)\n")
resultados_file.write("-"*80 + "\n\n")

for modelo in modelos:
    for c1_val, c2_val in c1_c2:
        resultados_file.write(f"\n--- PSO {modelo} - c1={c1_val}, c2={c2_val} ---\n")
        tiempo_inicio = datetime.now()
        hparams, val_loss, val_rmsle, val_rmse, model = pso_search(
            modelo, input_shape, n_stores, n_items,
            X_train, y_train, X_val, y_val,
            X_store_train, X_item_train, X_store_val, X_item_val,
            n_particles=PSO_DEFAULT_N_PARTICLES,
            n_iterations=PSO_DEFAULT_N_ITERATIONS,
            max_epochs=15,
            batch_size=32,
            w=PSO_DEFAULT_W,
            c1=c1_val,
            c2=c2_val
        )
        tiempo_fin = datetime.now()
        tiempo_transcurrido = tiempo_fin - tiempo_inicio
        registrar_resultado(resultados_file, f"PSO_c1{c1_val}_c2{c2_val}", modelo, val_rmsle, val_rmse, 
                          hparams, tiempo_transcurrido, tiempo_fin)
        resultados_file.write("\n")

# ============================================
# BUCLE 4: Variar w (fijar n_particles, n_iterations, c1, c2)
# ============================================
PSO_DEFAULT_C1 = 1.5
PSO_DEFAULT_C2 = 1.5

resultados_file.write("\n" + "-"*80 + "\n")
resultados_file.write("BUCLE 4: Variando w (n_particles=15, n_iterations=10, c1=1.5, c2=1.5)\n")
resultados_file.write("-"*80 + "\n\n")

for modelo in modelos:
    for w_val in w:
        resultados_file.write(f"\n--- PSO {modelo} - w={w_val} ---\n")
        tiempo_inicio = datetime.now()
        hparams, val_loss, val_rmsle, val_rmse, model = pso_search(
            modelo, input_shape, n_stores, n_items,
            X_train, y_train, X_val, y_val,
            X_store_train, X_item_train, X_store_val, X_item_val,
            n_particles=PSO_DEFAULT_N_PARTICLES,
            n_iterations=PSO_DEFAULT_N_ITERATIONS,
            max_epochs=15,
            batch_size=32,
            w=w_val,
            c1=PSO_DEFAULT_C1,
            c2=PSO_DEFAULT_C2
        )
        tiempo_fin = datetime.now()
        tiempo_transcurrido = tiempo_fin - tiempo_inicio
        registrar_resultado(resultados_file, f"PSO_w{w_val}", modelo, val_rmsle, val_rmse, 
                          hparams, tiempo_transcurrido, tiempo_fin)
        resultados_file.write("\n")

# Cerrar archivo de resultados
resultados_file.write("\n" + "="*80 + "\n")
resultados_file.write(f"Fecha de finalización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
resultados_file.write("="*80 + "\n")
resultados_file.close()

# Restaurar valores finales por defecto para PSO
PSO_DEFAULT_W = 0.7
print("Los resultados de los parámetros por defecto para PSO son: ")
print(f"n_particles={PSO_DEFAULT_N_PARTICLES}, n_iterations={PSO_DEFAULT_N_ITERATIONS}, c1={PSO_DEFAULT_C1}, c2={PSO_DEFAULT_C2}, w={PSO_DEFAULT_W}")
print("\n✓ Resultados guardados en 'parametros_GA_PSO.txt'")

print("\n" + "="*80)
print("FIN DEL SCRIPT")
print("="*80)

# __________________________________________________________________________________________