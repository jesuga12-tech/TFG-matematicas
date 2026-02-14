"""
Módulo de preprocesamiento de datos para series temporales.

Contiene funciones para:
- Carga y preparación de datos
- Creación de ventanas temporales
- Mapeo de embeddings
- Escalado y división train/val/test
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

__all__ = [
    "load_and_prepare_data_global",
    "create_windows_per_series",
    "create_embedding_mappings",
    "scale_and_split"
]



def load_and_prepare_data_global(train_csv, max_rows=None):
    """
    Carga múltiples series de train.csv y las concatena en un solo DataFrame.
    Limita filas para velocidad.
    """
    print("Leyendo datos globales...")
    df = pd.read_csv(train_csv, parse_dates=["date"], nrows=max_rows)
    df = df.sort_values(["store_nbr", "item_nbr", "date"])
    
    # Evitar negativos
    df["unit_sales"] = df["unit_sales"].clip(lower=0)
    df["y"] = np.log1p(df["unit_sales"])
    
    # Convertir a str para consistencia
    df["store_nbr"] = df["store_nbr"].astype(str)
    df["item_nbr"] = df["item_nbr"].astype(str)
    
    # Procesar cada serie para rellenar los NA, añadir lags y rolling statistics
    series_list = []
    for (store, item), group in df.groupby(["store_nbr", "item_nbr"]):
        group = group.sort_values("date").copy()

        start = group['date'].min()
        end = group['date'].max()
        # full_range = pd.date_range(start, end, freq="D")
        full_range = pd.date_range("02/01/2013", "15/08/2017", freq="D")

        # Crear fechas faltantes
        group = (
        group
        .set_index("date")
        .reindex(full_range)
        .rename_axis("date")
        .reset_index()
        )

        # Features calendario
        group["dayofweek"] = group["date"].dt.dayofweek
        group["month"] = group["date"].dt.month
        group["day"] = group["date"].dt.day
        group["week"] = group["date"].dt.isocalendar().week

        # Codificación cíclica
        group["dow_sin"] = np.sin(2 * np.pi * group["dayofweek"] / 7)
        group["dow_cos"] = np.cos(2 * np.pi * group["dayofweek"] / 7)
        group["month_sin"] = np.sin(2 * np.pi * group["month"] / 12)
        group["month_cos"] = np.cos(2 * np.pi * group["month"] / 12)
        group["day_sin"] = np.sin(2 * np.pi * group["day"] / 31)
        group["day_cos"] = np.cos(2 * np.pi * group["day"] / 31)
        group["week_sin"] = np.sin(2 * np.pi * group["week"] / 52)
        group["week_cos"] = np.cos(2 * np.pi * group["week"] / 52)
        
        # Rellenamos los NA de y con 0
        group["y"] = group["y"].fillna(0)

        # LAG FEATURES 
        lags = [1, 7, 14, 28, 56, 84]
        for lag in lags:
            group[f'lag_{lag}'] = group['y'].shift(lag)

        # ROLLING STATISTICS 
        roll_windows = [3, 7, 14, 28]
        for window in roll_windows:
            # Usar shift(1) para evitar data leakage (no incluir día t)
            group[f'roll_mean_{window}'] = group['y'].shift(1).rolling(window, min_periods=1).mean()
            group[f'roll_std_{window}'] = group['y'].shift(1).rolling(window, min_periods=1).std()
            group[f'roll_median_{window}'] = group['y'].shift(1).rolling(window, min_periods=1).median()
            group[f'roll_min_{window}'] = group['y'].shift(1).rolling(window, min_periods=1).min()
            group[f'roll_max_{window}'] = group['y'].shift(1).rolling(window, min_periods=1).max()
            
        # Rellenar NAs de lags y rolling 
        for col in group.columns:
            if col.startswith('lag_') or col.startswith('promo_lag_'):
                group[col] = group[col].fillna(0)
            elif col.startswith('roll_') or col.startswith('promo_roll_'):
                group[col] = group[col].ffill().fillna(0)

        group["store_nbr"] = store
        group["item_nbr"] = item

        series_list.append(group)
        
    all_series = pd.concat(series_list, ignore_index=True)
    print(f"Datos preparados: {len(all_series)} filas de {len(series_list)} series.")
    print(f"Features disponibles: {[col for col in all_series.columns if col not in ['date', 'unit_sales', 'store_nbr', 'item_nbr']]}")
    return all_series


def create_windows_per_series(series_df, window_size=30):
    """
    Crea ventanas temporales por cada serie (store, item)
    y concatena todo para entrenar un modelo global.
    """
    # Features temporales secuenciales (las que van en la ventana)
    temporal_features = [
        'y',  # Ventas (target pasado)
        'dow_sin', 'dow_cos',  # Día de la semana (cíclico)
        'month_sin', 'month_cos',  # Mes (cíclico)
        'day_sin', 'day_cos',  # Día del mes (cíclico)
        'week_sin', 'week_cos',  # Semana del año (cíclico)
        # Lag features
        'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56', 'lag_84',
        # Rolling statistics
        'roll_mean_3', 'roll_std_3', 'roll_median_3', 'roll_min_3', 'roll_max_3',
        'roll_mean_7', 'roll_std_7', 'roll_median_7', 'roll_min_7', 'roll_max_7',
        'roll_mean_14', 'roll_std_14', 'roll_median_14', 'roll_min_14', 'roll_max_14',
        'roll_mean_28', 'roll_std_28', 'roll_median_28', 'roll_min_28', 'roll_max_28'
    ]
    
    # Filtrar solo las features que existen en el DataFrame
    available_features = [f for f in temporal_features if f in series_df.columns]
    print(f"Usando {len(available_features)} features temporales: {available_features[:10]}...")

    X_seq_list = []
    X_store_list = []
    X_item_list = []
    y_list = []
    dates_list = []

    for (store, item), group in series_df.groupby(['store_nbr', 'item_nbr']):
        store = int(store)
        item = int(item)
        group = group.sort_values('date').reset_index(drop=True)

        if len(group) <= window_size:
            continue

        # Asegurar que todas las features estén presentes y sin NaN
        data = group[available_features].fillna(0).values

        for i in range(window_size, len(group)):
            X_seq_list.append(data[i-window_size:i])
            X_store_list.append(store)
            X_item_list.append(item)
            y_list.append(data[i, 0])  # y del día siguiente (primera columna es 'y')
            dates_list.append(group['date'].iloc[i])

    X_seq = np.array(X_seq_list)
    X_store = np.array(X_store_list).reshape(-1, 1)
    X_item = np.array(X_item_list).reshape(-1, 1)
    y = np.array(y_list)
    dates = np.array(dates_list)

    print(f"Ventanas creadas:")
    print(f"  X_seq: {X_seq.shape} (window_size={window_size}, features={X_seq.shape[2]})")
    print(f"  X_store: {X_store.shape}")
    print(f"  y: {y.shape}")
    print(f"  Series únicas: {len(set(zip(X_store.flatten(), X_item.flatten())))}")

    return X_seq, X_store, X_item, y, dates


def create_embedding_mappings(X_store, X_item):
    """
    Crea mapeos de valores únicos de store e item a índices consecutivos (0, 1, 2, ...)
    necesarios para los embeddings de Keras.
    
    Returns:
        n_stores: número de stores únicos
        n_items: número de items únicos
        X_store_mapped: array con stores mapeados a índices
        X_item_mapped: array con items mapeados a índices
    """
    unique_stores = np.unique(X_store.flatten())
    unique_items = np.unique(X_item.flatten())
    
    # Crear mapeos: valor original -> índice (0, 1, 2, ...)
    store_to_idx = {store: idx for idx, store in enumerate(unique_stores)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    
    n_stores = len(unique_stores)
    n_items = len(unique_items)
    
    # Aplicar mapeos y asegurar que sean enteros
    X_store_mapped = np.array([store_to_idx[store] for store in X_store.flatten()], dtype=np.int32).reshape(X_store.shape)
    X_item_mapped = np.array([item_to_idx[item] for item in X_item.flatten()], dtype=np.int32).reshape(X_item.shape)
    
    print(f"Stores únicos: {n_stores} (rango original: {unique_stores.min()} - {unique_stores.max()})")
    print(f"Items únicos: {n_items} (rango original: {unique_items.min()} - {unique_items.max()})")
    print(f"Stores mapeados a índices: 0 - {n_stores-1}")
    print(f"Items mapeados a índices: 0 - {n_items-1}")
    
    return n_stores, n_items, X_store_mapped, X_item_mapped


def scale_and_split(X, y, dates, X_store, X_item, train_size=0.6, test_size=0.2, val_size=0.2):
    """
    Escala las features y divide en train/valid/test usando un split temporal puro.
    """
    num_samples, window_size, num_features = X.shape

    # Reorganizar X a 2D para aplicar StandardScaler
    X_2d = X.reshape(-1, num_features)

    # Escalar TODAS las features
    scaler = StandardScaler()
    X_2d_scaled = scaler.fit_transform(X_2d)

    # Volver a 3D
    X_scaled = X_2d_scaled.reshape(num_samples, window_size, num_features)

    # Calcular índices para split temporal
    train_end = int(num_samples * train_size)
    val_end = train_end + int(num_samples * val_size)

    # Train
    X_train = X_scaled[:train_end]
    y_train = y[:train_end]
    dates_train = dates[:train_end]
    X_store_train = X_store[:train_end]
    X_item_train = X_item[:train_end]

    # Validation
    X_val = X_scaled[train_end:val_end]
    y_val = y[train_end:val_end]
    dates_val = dates[train_end:val_end]
    X_store_val = X_store[train_end:val_end]
    X_item_val = X_item[train_end:val_end]

    # Test
    X_test = X_scaled[val_end:]
    y_test = y[val_end:]
    dates_test = dates[val_end:]
    X_store_test = X_store[val_end:]
    X_item_test = X_item[val_end:]

    print("Tamaños:")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Val:   {X_val.shape}, {y_val.shape}")
    print(f"  Test:  {X_test.shape}, {y_test.shape}")

    return (X_train, y_train, X_store_train, X_item_train, X_val,
            y_val, X_store_val, X_item_val, X_test, y_test,
            X_store_test, X_item_test, scaler)

