import os
import pickle
import sqlite3
import threading
from itertools import islice

import numpy as np
import pandas as pd

import modelos_surprise


# Modelos de redes neuronales de los notebooks clase13_Dos_torres_*.ipynb y
# clase13_Una_gran_torre.ipynb. Se entrenan con Keras (backend JAX), pero lo que
# se guarda con pickle son sólo los pesos como arrays de NumPy, y la predicción
# se calcula con NumPy: para recomendar no hace falta tener Keras ni JAX
# instalados, sólo para entrenar.
#
# - dos_torres: como en clase13_Dos_torres_2. Una torre de recuperación (producto
#   escalar entre el embedding del lector y el del libro) elige los 100 mejores
#   candidatos, y el modelo de ranking (embeddings + capas densas) los ordena.
# - gran_torre: como en clase13_Una_gran_torre. Un único modelo de ranking que
#   además de lector y libro usa autor, género, editorial y año de edición del
#   libro, y género y año de nacimiento del lector. Ordena todos los libros.
#
# A diferencia de los notebooks, se entrena sin separar un conjunto de test, y
# con los mismos libros y lectores que los modelos de Surprise (los que tienen más
# de modelos_surprise.MIN_VALORACIONES valoraciones).
NOMBRES = ('dos_torres', 'gran_torre')
CANDIDATOS_DOS_TORRES = 100
RATING_MIN, RATING_MAX = 1, 10

QUERY_INTERACCIONES = modelos_surprise.QUERY_INTERACCIONES
QUERY_LIBROS = "SELECT id_libro, autor, genero, editorial, anio_edicion FROM libros"
QUERY_LECTORES = "SELECT id_lector, genero, nacimiento FROM lectores"

_modelos = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Predicción (sólo NumPy)
# ---------------------------------------------------------------------------

def _puntuar(modelo, u, libros):
    # Las capas densas del modelo de ranking, calculadas con NumPy. La primera
    # capa es lineal sobre la concatenación de los rasgos del lector y del libro,
    # así que su resultado es la suma de una parte que depende sólo del lector y
    # otra que depende sólo del libro, ya calculadas al entrenar
    h = np.maximum(modelo['parte_lector'][u] + modelo['parte_libro'][libros], 0)
    (w2, b2), (w3, b3) = modelo['capas']
    h = np.maximum(h @ w2 + b2, 0)
    rating = 1 / (1 + np.exp(-(h @ w3 + b3)[:, 0]))  # sigmoide
    return rating * (RATING_MAX - RATING_MIN) + RATING_MIN


def ranking(nombre, database_file, id_lector, es_candidato):
    # recorre los libros recomendados de mayor a menor rating estimado.
    # es_candidato(id_libro) dice si el libro se puede recomendar; dos_torres lo
    # necesita para elegir sus 100 candidatos entre libros que no se hayan leído
    m = modelo(nombre, database_file)

    # los lectores que no estaban al entrenar no tienen embedding: se usa Baseline
    u = m['indice_lector'].get(id_lector)
    if u is None:
        yield from modelos_surprise.ranking('baseline', database_file, id_lector)
        return

    if nombre == 'dos_torres':
        afinidad = m['recuperacion_libro'] @ m['recuperacion_lector'][u]
        candidatos = (i for i in np.argsort(-afinidad, kind='stable') if es_candidato(m['ids_libros'][i]))
        libros = np.array(list(islice(candidatos, CANDIDATOS_DOS_TORRES)), dtype=np.int64)
    else:
        libros = np.arange(len(m['ids_libros']))

    if len(libros) == 0:
        return
    est = _puntuar(m, u, libros)
    for j in np.argsort(-est, kind='stable'):
        yield m['ids_libros'][libros[j]], est[j]


# ---------------------------------------------------------------------------
# Carga y entrenamiento
# ---------------------------------------------------------------------------

def modelo(nombre, database_file):
    # se carga (o entrena) la primera vez que se usa y queda en memoria
    with _lock:
        if nombre not in _modelos:
            _modelos[nombre] = _cargar_o_entrenar(nombre, database_file)
        return _modelos[nombre]


def _cargar_o_entrenar(nombre, database_file):
    # si ya hay un modelo guardado se usa ese; si no, se entrena y se guarda.
    # Para reentrenar hay que borrar el .pkl
    archivo = os.path.join(modelos_surprise.CARPETA_MODELOS, f'{nombre}.pkl')
    if os.path.exists(archivo):
        with open(archivo, 'rb') as f:
            return pickle.load(f)

    m = _entrenar(nombre, database_file)
    os.makedirs(modelos_surprise.CARPETA_MODELOS, exist_ok=True)
    with open(archivo + '.tmp', 'wb') as f:
        pickle.dump(m, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(archivo + '.tmp', archivo)
    return m


def _keras():
    # Keras sólo se importa para entrenar
    os.environ.setdefault('KERAS_BACKEND', 'jax')
    import keras
    keras.utils.set_random_seed(42)
    return keras


def _codigos(serie):
    # categoría -> número, como .astype('category').cat.codes en el notebook; los
    # nulos van a un código propio al final en lugar de -1
    codigos = serie.astype('category').cat.codes.to_numpy().astype(np.int64)
    n = int(codigos.max()) + 2
    codigos[codigos < 0] = n - 1
    return codigos, n


def _normalizar_anio(serie):
    # como en el notebook: a número, nulos en 0, y escalado entre 0 y 1
    x = pd.to_numeric(serie, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
    return (x - x.min()) / max(x.max() - x.min(), 1.0)


def _entrenar_ranking(keras, rasgos_lector, rasgos_libro, u, i, y, epochs, batch_size, callbacks=()):
    # Red de ranking: un embedding por cada rasgo categórico (los numéricos van
    # tal cual), todo concatenado y luego Dense(256) -> Dense(64) -> Dense(1).
    # rasgos_*: {nombre: (valores por lector/libro, tamaño del embedding o None si es numérico)}
    entradas, vectores, x = [], [], {}
    for rasgos, filas in ((rasgos_lector, u), (rasgos_libro, i)):
        for nombre, (valores, dim) in rasgos.items():
            entrada = keras.layers.Input(shape=(1,), name=nombre)
            entradas.append(entrada)
            if dim is None:
                vectores.append(entrada)
            else:
                emb = keras.layers.Embedding(int(valores.max()) + 1, dim, name=f'emb_{nombre}')(entrada)
                vectores.append(keras.layers.Flatten()(emb))
            x[nombre] = valores[filas].astype(np.float32)

    h = keras.layers.Concatenate()(vectores)
    h = keras.layers.Dense(256, activation='relu', name='densa_1')(h)
    h = keras.layers.Dense(64, activation='relu', name='densa_2')(h)
    salida = keras.layers.Dense(1, activation='sigmoid', name='densa_3')(h)

    red = keras.Model(inputs=entradas, outputs=salida)
    red.compile(
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.RootMeanSquaredError()],
        optimizer=keras.optimizers.Adagrad(learning_rate=0.1),
    )
    red.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=list(callbacks), verbose=2)

    # la matriz de la primera capa se parte en las filas que corresponden a los
    # rasgos del lector y las que corresponden a los del libro
    w1, b1 = red.get_layer('densa_1').get_weights()
    partes, fila = [], 0
    for rasgos in (rasgos_lector, rasgos_libro):
        bloques = []
        for nombre, (valores, dim) in rasgos.items():
            if dim is None:
                bloques.append(valores[:, None].astype(np.float32))
            else:
                bloques.append(red.get_layer(f'emb_{nombre}').get_weights()[0][valores])
        X = np.concatenate(bloques, axis=1)
        partes.append(X @ w1[fila:fila + X.shape[1]])
        fila += X.shape[1]

    return {
        'parte_lector': partes[0] + b1,
        'parte_libro': partes[1],
        'capas': [tuple(red.get_layer(n).get_weights()) for n in ('densa_2', 'densa_3')],
    }


def _entrenar(nombre, database_file):
    keras = _keras()
    conn = sqlite3.connect(database_file)
    df = pd.read_sql(QUERY_INTERACCIONES, conn)

    # ids de lectores y libros -> posiciones 0..n-1 (los diccionarios de traducción del notebook)
    u, ids_lectores = pd.factorize(df['id_lector'])
    i, ids_libros = pd.factorize(df['id_libro'])
    y = ((df['rating'].to_numpy() - RATING_MIN) / (RATING_MAX - RATING_MIN)).astype(np.float32)
    lectores = np.arange(len(ids_lectores))
    libros = np.arange(len(ids_libros))

    m = {
        'ids_libros': list(ids_libros),
        'indice_lector': {id_lector: n for n, id_lector in enumerate(ids_lectores)},
    }

    if nombre == 'dos_torres':
        # torre de recuperación (clase13_Dos_torres_2): el score es el producto
        # escalar de los embeddings y se entrena para aproximar el rating
        recuperacion = _modelo_recuperacion(keras, len(ids_lectores), len(ids_libros), 256)
        recuperacion.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adagrad(learning_rate=0.1))
        recuperacion.fit({'lector': u, 'libro': i}, y, epochs=10, batch_size=1024, verbose=2)
        m['recuperacion_lector'] = recuperacion.emb_lector.get_weights()[0]
        m['recuperacion_libro'] = recuperacion.emb_libro.get_weights()[0]

        # modelo de ranking: sólo los embeddings de lector y libro (dimensión 32)
        m.update(_entrenar_ranking(keras, {'lector': (lectores, 32)}, {'libro': (libros, 32)}, u, i, y, epochs=10, batch_size=256))
    else:
        libros_df = pd.read_sql(QUERY_LIBROS, conn).set_index('id_libro').reindex(ids_libros)
        lectores_df = pd.read_sql(QUERY_LECTORES, conn).set_index('id_lector').reindex(ids_lectores)

        def categorico(serie):
            codigos, n = _codigos(serie)
            return codigos, min(n // 2 + 1, 50)

        rasgos_lector = {
            'lector': (lectores, min(len(lectores) // 2 + 1, 50)),
            'genero_lector': categorico(lectores_df['genero']),
            'anio_nacimiento': (_normalizar_anio(lectores_df['nacimiento']), None),
        }
        rasgos_libro = {
            'libro': (libros, min(len(libros) // 2 + 1, 50)),
            'editorial': categorico(libros_df['editorial']),
            'autor': categorico(libros_df['autor']),
            'genero': categorico(libros_df['genero']),
            'anio_edicion': (_normalizar_anio(libros_df['anio_edicion']), None),
        }
        # como en el notebook: hasta 100 épocas, cortando cuando el loss no mejora en 3 seguidas
        corte = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        m.update(_entrenar_ranking(keras, rasgos_lector, rasgos_libro, u, i, y, epochs=100, batch_size=1024, callbacks=[corte]))

    conn.close()
    return m


def _modelo_recuperacion(keras, num_lectores, num_libros, dim):
    class Recuperacion(keras.Model):
        # equivale al RetrievalModel del notebook sin la capa BruteForceRetrieval
        # de keras-rs: la búsqueda de los mejores libros se hace al predecir con NumPy
        def __init__(self):
            super().__init__()
            self.emb_lector = keras.layers.Embedding(num_lectores, dim)
            self.emb_libro = keras.layers.Embedding(num_libros, dim)

        def call(self, inputs):
            return keras.ops.sum(self.emb_lector(inputs['lector']) * self.emb_libro(inputs['libro']), axis=1, keepdims=True)

    return Recuperacion()


if __name__ == '__main__':
    # entrena y guarda los modelos que falten: python modelos_torres.py
    import recomendador
    for n in NOMBRES:
        modelo(n, recomendador.DATABASE_FILE)
        print(n, 'listo')
