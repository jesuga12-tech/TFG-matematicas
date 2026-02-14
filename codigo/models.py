"""
Módulo de modelos de deep learning para series temporales.

Contiene:
- Métricas personalizadas (RMSE)
- Funciones de embeddings
- Construcción de modelos (RNN/LSTM, CNN, TCN)
- Funciones de entrenamiento
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

__all__ = [
    "build_rnn_model",
    "build_cnn_model",
    "build_tcn_model",
    "train_and_evaluate",
    "rmse",
    
]



def rmse(y_true, y_pred):
    """
    Calcular el RMSE en escala real (después de deshacer el log).
    
    :param y_true: Valores reales en log scale.
    :param y_pred: Valores predichos en log scale.
    :return: El RMSE en escala real.
    """
    # Deshacer el logaritmo
    y_true_real = tf.math.expm1(y_true)
    y_pred_real = tf.math.expm1(y_pred)
    
    # Calcular el error cuadrado
    error = tf.math.square(y_true_real - y_pred_real)
    
    # Calcular el RMSE sin pesos
    rmse_value = tf.math.sqrt(tf.reduce_mean(error))

    return rmse_value


def create_embeddings(store_input, item_input, n_stores, n_items):
    """
    Crea embeddings separados para store e item.
    Los embeddings se concatenan directamente sin interacción explícita.
    
    Args:
        store_input: Input layer para store
        item_input: Input layer para item
        n_stores: número de stores únicos
        n_items: número de items únicos
    
    Returns:
        combined_embed: embedding combinado que incluye store e item concatenados
    """
    store_emb = layers.Embedding(n_stores, 1)(store_input)
    item_emb = layers.Embedding(n_items, 5)(item_input)

    print(f"  store_emb: {store_emb.shape}")
    print(f"  item_emb: {item_emb.shape}")

    emb = layers.Concatenate()([
        layers.Flatten()(store_emb),
        layers.Flatten()(item_emb)
    ])
    return emb


def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate):
    """
    Bloque básico de TCN con convolución dilatada y conexión residual.
    """
    conv1 = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu"
    )(x)
    conv1 = layers.Dropout(dropout_rate)(conv1)

    conv2 = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu"
    )(conv1)
    conv2 = layers.Dropout(dropout_rate)(conv2)

    # Residual
    if x.shape[-1] != filters:
        res = layers.Conv1D(filters=filters, kernel_size=1, padding="same")(x)
    else:
        res = x
    out = layers.Add()([res, conv2])

    out = layers.LayerNormalization()(out)
    return out


def build_rnn_model(input_shape, hparams, n_stores, n_items):
    """
    Construye un modelo RNN con LSTM y embeddings para store/item.
    
    input_shape: (window_size, num_features_temporal)
    hparams: diccionario con hiperparámetros
    n_stores: número de stores únicos (para dimensión del embedding)
    n_items: número de items únicos (para dimensión del embedding)
    """
    # Inputs
    seq_input = layers.Input(shape=input_shape, name='sequence')
    store_input = layers.Input(shape=(1,), name='store')
    item_input = layers.Input(shape=(1,), name='item')
    
    # Embeddings separados 
    combined_embed = create_embeddings(store_input, item_input, n_stores, n_items)
    
    # Repetir embeddings para cada timestep y concatenar con secuencia temporal
    window_size = input_shape[0]
    combined_embed_expanded = layers.RepeatVector(window_size)(combined_embed)
    
    # Concatenar embeddings con secuencia temporal
    x = layers.Concatenate(axis=-1)([seq_input, combined_embed_expanded])
    
    # RNN on sequence
    x = layers.LSTM(hparams["lstm_units"], return_sequences=False, dropout=hparams["dropout"])(x)
    x = layers.Dense(hparams["dense_units"], activation="relu")(x)
    outputs = layers.Dense(1)(x)
    
    model = models.Model([seq_input, store_input, item_input], outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hparams["lr"]),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmsle"), rmse]
    )
    return model


def build_cnn_model(input_shape, hparams, n_stores, n_items):
    """
    Construye un modelo CNN 1D con embeddings para store/item.
    
    input_shape: (window_size, num_features_temporal)
    hparams: diccionario con hiperparámetros
    n_stores: número de stores únicos (para dimensión del embedding)
    n_items: número de items únicos (para dimensión del embedding)
    """
    # Inputs
    seq_input = layers.Input(shape=input_shape, name='sequence')
    store_input = layers.Input(shape=(1,), name='store')
    item_input = layers.Input(shape=(1,), name='item')
    
    # Embeddings separados 
    combined_embed = create_embeddings(store_input, item_input, n_stores, n_items)
    
    # Repetir embeddings para cada timestep y concatenar con secuencia temporal
    window_size = input_shape[0]
    combined_embed_expanded = layers.RepeatVector(window_size)(combined_embed)
    
    # Concatenar embeddings con secuencia temporal
    x = layers.Concatenate(axis=-1)([seq_input, combined_embed_expanded])
    
    # CNN on sequence (fijado a 2 capas convolucionales)
    for _ in range(2):
        x = layers.Conv1D(
            filters=hparams["filters"],
            kernel_size=hparams["kernel_size"],
            padding="causal",
            activation="relu"
        )(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(hparams["dense_units"], activation="relu")(x)
    outputs = layers.Dense(1)(x)
    
    model = models.Model([seq_input, store_input, item_input], outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hparams["lr"]),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmsle"), rmse]
    )
    return model


def build_tcn_model(input_shape, hparams, n_stores, n_items):
    """
    Construye un modelo TCN con embeddings para store/item.
    
    input_shape: (window_size, num_features_temporal)
    hparams: diccionario con hiperparámetros
    n_stores: número de stores únicos (para dimensión del embedding)
    n_items: número de items únicos (para dimensión del embedding)
    """
    # Inputs
    seq_input = layers.Input(shape=input_shape, name='sequence')
    store_input = layers.Input(shape=(1,), name='store')
    item_input = layers.Input(shape=(1,), name='item')
    
    # Embeddings separados
    combined_embed = create_embeddings(store_input, item_input, n_stores, n_items)
    
    # Repetir embeddings para cada timestep y concatenar con secuencia temporal
    window_size = input_shape[0]
    combined_embed_expanded = layers.RepeatVector(window_size)(combined_embed)
    
    # Concatenar embeddings con secuencia temporal
    x = layers.Concatenate(axis=-1)([seq_input, combined_embed_expanded])
    
    # TCN on sequence
    for i in range(hparams["num_blocks"]):
        dilation = 2 ** i
        x = tcn_block(
            x,
            filters=hparams["filters"],
            kernel_size=hparams["kernel_size"],
            dilation_rate=dilation,
            dropout_rate=hparams["dropout"]
        )

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(hparams["dense_units"], activation="relu")(x)
    outputs = layers.Dense(1)(x)
    
    model = models.Model([seq_input, store_input, item_input], outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hparams["lr"]),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmsle"), rmse]
    )
    return model


def train_and_evaluate(model, model_name,
                       X_train, y_train, X_val, y_val,
                       X_test, y_test,
                       epochs=20, batch_size=32):
    """
    Entrena un modelo y muestra resultados en validación y test.
    """
    
    print(f"\nEntrenando modelo: {model_name}")
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluar modelo
    test_msle, test_rmsle, test_rmse = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test MSLE: {test_msle:.4f}, Test RMSLE: {test_rmsle:.4f}, Test RMSE: {test_rmse:.4f}")

    return test_msle, test_rmsle, test_rmse
