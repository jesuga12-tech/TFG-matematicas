# Optimización de Hiperparámetros para Series Temporales

Este proyecto implementa un sistema completo para la optimización de hiperparámetros en modelos de deep learning (LSTM, CNN 1D, TCN) aplicados a la predicción de series temporales de ventas. El sistema utiliza cuatro algoritmos de optimización: Optimización Bayesiana (Optuna), Random Search, Algoritmo Genético (GA) y Particle Swarm Optimization (PSO).

##  Workflow del Proyecto

### 1. Análisis Exploratorio de Datos

**Archivo**: `notebooks/exploratory_analysis.ipynb`

El proceso comienza con un análisis exploratorio de los datos que incluye:

- **Selección de tienda**: Análisis estadístico de todas las tiendas para identificar la tienda 47 como representativa
- **Análisis de productos**: Filtrado de productos con alta cobertura temporal y baja variabilidad
- **Visualizaciones**: Gráficos de series temporales, distribución de ventas, relación cobertura-variabilidad, y análisis de transformación logarítmica

**Resultados generados**:
- **Figuras**: Guardadas en `results/figures/`
  - `grafico_01_top_tiendas_ventas.png`
  - `grafico_03_series_temporales_3_items.png`
  - `grafico_05_cobertura_vs_variabilidad.png`
  - `grafico_08_boxplot_ventas_productos.png`
  - `grafico_09_transformacion_logaritmica.png`
- **Tablas Excel**: Guardadas en `results/metrics/`
  - `store_stats.xlsx`: Información estadística sobre todas las tiendas
  - `product_range.xlsx`: Información detallada sobre los productos de la tienda 47

**Datos generados**:
- `data/processed/series_tienda_47.csv`: Datos filtrados de la tienda 47
- `data/processed/series_tienda47_seleccionados.csv`: Productos seleccionados de la tienda 47

### 2. Preprocesamiento de Series Temporales

**Archivo**: `src/preprocessing.py`

Este módulo gestiona toda la preparación de las series temporales:

- **`load_and_prepare_data_global()`**: Carga datos, aplica transformación logarítmica, rellena valores faltantes y añade características temporales (features calendario, lags, estadísticas rolling)
- **`create_windows_per_series()`**: Crea ventanas temporales de tamaño fijo para cada serie (store, item)
- **`create_embedding_mappings()`**: Genera mapeos de tiendas e items a índices numéricos consecutivos para los embeddings
- **`scale_and_split()`**: Escala las características usando StandardScaler y divide los datos en conjuntos de entrenamiento, validación y test respetando el orden temporal

### 3. Modelos de Deep Learning

**Archivo**: `src/models.py`

Contiene la implementación de todos los modelos de deep learning:

- **`build_lstm_model()`**: Modelo LSTM con embeddings para store/item
  - Hiperparámetros optimizables: `lstm_units`, `dropout`, `lr`
- **`build_cnn_model()`**: Modelo CNN 1D con embeddings para store/item
  - Hiperparámetros optimizables: `filters`, `kernel_size`, `lr`
- **`build_tcn_model()`**: Modelo TCN (Temporal Convolutional Network) con embeddings
  - Hiperparámetros optimizables: `filters`, `kernel_size`, `num_blocks`, `lr`
- **`train_and_evaluate()`**: Función para entrenar y evaluar modelos con EarlyStopping
- **`rmse()`**: Métrica personalizada para calcular RMSE en escala real

### 4. Selección de Parámetros de GA y PSO

**Archivo**: `src/parametros_GA_PSO.py`

Este script selecciona secuencialmente los parámetros internos de los algoritmos genético y PSO:

- **Algoritmo Genético (GA)**: Selecciona `pop_size`, `n_generations`, `crossover_rate`, `mutation_rate`
- **Particle Swarm Optimization (PSO)**: Selecciona `n_particles`, `n_iterations`, `c1`, `c2`, `w`

**Resultados generados**:
- `results/metrics/parametros_GA.txt`: Resultados del ajuste secuencial de parámetros del GA
- `results/metrics/parametros_PSO.txt`: Resultados del ajuste secuencial de parámetros del PSO

### 5. Optimización de Hiperparámetros

**Archivo**: `src/optimization.py`

Contiene las funciones para optimizar los hiperparámetros de los modelos:

- **`bayesian_optimization()`**: Optimización Bayesiana usando Optuna
- **`random_search()`**: Búsqueda aleatoria de hiperparámetros
- **`genetic_search()`**: Algoritmo Genético para optimización
- **`pso_search()`**: Particle Swarm Optimization para optimización
- **`train_model_quick()`**: Entrenamiento rápido para optimización (con menos epochs)

Todos los algoritmos exploran el learning rate en **escala logarítmica** para una búsqueda más eficiente.

### 6. Script Principal

**Archivo**: `src/main.py`

Este script ejecuta todo el pipeline completo:

#### 6.1. Procesamiento de la Serie Temporal
- Carga de datos desde `data/processed/series_tienda47_seleccionados.csv`
- Creación de ventanas temporales (window_size=30)
- Escalado y división en train/val/test (60%/20%/20%)

#### 6.2. Entrenamiento de Modelos con Parámetros Fijos
- Entrenamiento de LSTM, CNN y TCN con hiperparámetros por defecto
- Cálculo de métricas en el conjunto de test
- Comparación con modelos baseline (naive y seasonal naive)

#### 6.3. Optimización de Hiperparámetros
- Optimización de cada modelo (CNN, LSTM, TCN) usando los 4 algoritmos (BO, RS, GA, PSO)
- Evaluación en el conjunto de validación
- **Resultados guardados en**: `results/metrics/resultados_optimizacion_hiperparametros_val.txt`

#### 6.4. Evaluación de Mejores Configuraciones en Test
- Evaluación de las mejores configuraciones obtenidas en validación sobre el conjunto de test
- **Resultados guardados en**: `results/metrics/resultados_modelos_optimizados.xlsx`

#### 6.5. Evaluación Estadística con Múltiples Semillas
- Selección de los mejores hiperparámetros de cada modelo
- Entrenamiento de cada modelo con 30 semillas aleatorias distintas
- Los resultados se utilizan para tests estadísticos (Friedman y Nemenyi)
- **Resultados guardados en**: `results/metrics/resultados_estadisticos_30_semillas.xlsx`

##  Estructura del Proyecto

```
.
├── data/
│   ├── raw/
│   │   └── train.csv ( disponible en Kaggle https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting )
[Dataset Store Sales - Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

│   └── processed/
│       ├── series_tienda_47.csv ( disponible en https://drive.google.com/drive/folders/1AGcDz5r77UsFA0V-XrCepc-sbd6kGqAP?usp=sharing )
│       └── series_tienda47_seleccionados.csv 
├── notebooks/
│   └── exploratory_analysis.ipynb
├── results/
│   ├── figures/
│   │   ├── grafico_01_top_tiendas_ventas.png
│   │   ├── grafico_03_series_temporales_3_items.png
│   │   ├── grafico_05_cobertura_vs_variabilidad.png
│   │   ├── grafico_08_boxplot_ventas_productos.png
│   │   └── grafico_09_transformacion_logaritmica.png
│   └── metrics/
│       ├── store_stats.xlsx
│       ├── product_range.xlsx
│       ├── parametros_GA.txt
│       ├── parametros_PSO.txt
│       ├── resultados_optimizacion_hiperparametros_val.txt
│       ├── resultados_modelos_optimizados.xlsx
│       └── resultados_estadisticos_30_semillas.xlsx
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── models.py
    ├── optimization.py
    ├── parametros_GA_PSO.py
    ├── main.py
    └── friedman_nemenyi.py
```

##  Rangos de Hiperparámetros Optimizables

### LSTM
- `lstm_units`: [32, 256] (múltiplos de 32)
- `dropout`: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
- `lr`: [1e-4, 1e-2] (escala logarítmica)

### CNN 1D
- `filters`: [16, 32, 64, 128]
- `kernel_size`: [2, 3, 4, 5, 6, 7]
- `lr`: [1e-4, 1e-2] (escala logarítmica)

### TCN
- `filters`: [16, 32, 64, 128]
- `kernel_size`: [2, 3, 4, 5]
- `num_blocks`: [2, 3, 4, 5]
- `lr`: [1e-4, 1e-2] (escala logarítmica)

##  Características Técnicas

### Transformación de Datos
- Los datos objetivo (`y`) están en escala logarítmica: `y = log1p(unit_sales)`
- Las métricas MSLE y RMSLE se calculan directamente en esta escala
- RMSE se calcula después de transformar a escala real: `expm1(y)`

### Learning Rate
Todos los algoritmos exploran el learning rate en **escala logarítmica**:
- Bayesian: `trial.suggest_float("lr", 1e-4, 1e-2, log=True)`
- Random Search: `10 ** random.uniform(-4, -2)`
- GA y PSO: Decodificación en escala logarítmica


### Embeddings
- Los embeddings de tienda e item se concatenan y se añaden a cada paso temporal
- Dimensión de embedding: 1 para store, 5 para item (total: 6)


##  Dependencias

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Optuna
- Scipy
- scikit-posthocs
- Matplotlib
- Seaborn
- Jupyter

## Licencia

Este proyecto es parte de un Trabajo de Fin de Grado (TFG).


