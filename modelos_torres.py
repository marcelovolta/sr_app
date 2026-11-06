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
# - dos_torres_v2: alternativa a dos_torres que NO sigue el notebook. La torre de
#   recuperación se entrena con negativos en lugar de aproximar el rating (ver
#   _entrenar_recuperacion_v2), y sus libros se recomiendan en ese orden, sin modelo
#   de ranking: en un test con el 20% de las valoraciones, reordenar con el ranking
#   bajaba el NDCG@20 de 0,36 a 0,10.
# - dos_torres_v2_completa: el mismo modelo que dos_torres_v2, pero entrenado con la
#   base completa en lugar de la reducida. Lo usa cascada_v1.
# - gran_torre: como en clase13_Una_gran_torre. Un único modelo de ranking que
#   además de lector y libro usa autor, género, editorial y año de edición del
#   libro, y género y año de nacimiento del lector. Ordena todos los libros.
# - gran_torre_v2: alternativa a gran_torre que NO sigue el notebook. La misma red
#   con los mismos rasgos, pero aprende a separar los libros que el lector eligió
#   de libros al azar, en lugar de aproximar el rating (ver _entrenar_gran_torre_v2).
#
# A diferencia de los notebooks, se entrena sin separar un conjunto de test, y
# con los mismos libros y lectores que los modelos de Surprise (los que tienen más
# de modelos_surprise.MIN_VALORACIONES valoraciones).
NOMBRES = ('dos_torres', 'dos_torres_v2', 'dos_torres_v2_completa', 'gran_torre', 'gran_torre_v2')
RECUPERACION_V2 = ('dos_torres_v2', 'dos_torres_v2_completa')
CON_RECUPERACION = ('dos_torres', *RECUPERACION_V2)
CANDIDATOS_DOS_TORRES = 100
RATING_MIN, RATING_MAX = 1, 10

# Torre de recuperación de dos_torres_v2
V2_RATING_POSITIVO = 8      # desde este rating, el libro cuenta como elegido por el lector
V2_DIM = 256
V2_TEMPERATURA = 0.1
V2_LEARNING_RATE = 0.01
V2_EPOCHS = 10
V2_BATCH_SIZE = 1024

# gran_torre_v2
GT2_NEGATIVOS_POR_POSITIVO = 8
GT2_LEARNING_RATE = 0.001
GT2_EPOCHS = 8  # el óptimo en el test fue 10 épocas con el 80% de los datos: son los mismos pasos
GT2_BATCH_SIZE = 1024

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
    *ocultas, (w_salida, b_salida) = modelo['capas']
    for w, b in ocultas:
        h = np.maximum(h @ w + b, 0)
    rating = 1 / (1 + np.exp(-(h @ w_salida + b_salida)[:, 0]))  # sigmoide
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

    if nombre in CON_RECUPERACION:
        afinidad = m['recuperacion_libro'] @ m['recuperacion_lector'][u]
        candidatos = (i for i in np.argsort(-afinidad, kind='stable') if es_candidato(m['ids_libros'][i]))
        libros = np.array(list(islice(candidatos, CANDIDATOS_DOS_TORRES)), dtype=np.int64)
        if nombre in RECUPERACION_V2:
            # sin ranking: ya están ordenados por afinidad (no es un rating estimado)
            for j in libros:
                yield m['ids_libros'][j], afinidad[j]
            return
    else:
        libros = np.arange(len(m['ids_libros']))

    if len(libros) == 0:
        return
    est = _puntuar(m, u, libros)
    for j in np.argsort(-est, kind='stable'):
        yield m['ids_libros'][libros[j]], est[j]


def conoce_lector(nombre, database_file, id_lector):
    # False si el lector no estaba al entrenar: no tiene embedding y ranking() usa Baseline
    return id_lector in modelo(nombre, database_file)['indice_lector']


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


def _entrenar_ranking(keras, rasgos_lector, rasgos_libro, u, i, y, epochs, batch_size, callbacks=(), loss=None, optimizer=None, remuestrear=None):
    # Red de ranking: un embedding por cada rasgo categórico (los numéricos van
    # tal cual), todo concatenado y luego Dense(256) -> Dense(64) -> Dense(1).
    # rasgos_*: {nombre: (valores por lector/libro, tamaño del embedding o None si es numérico)}
    # remuestrear: si se pasa, cada época entrena con los (u, i, y) que devuelve
    entradas, vectores = [], []
    for rasgos, filas in ((rasgos_lector, u), (rasgos_libro, i)):
        for nombre, (valores, dim) in rasgos.items():
            entrada = keras.layers.Input(shape=(1,), name=nombre)
            entradas.append(entrada)
            if dim is None:
                vectores.append(entrada)
            else:
                emb = keras.layers.Embedding(int(valores.max()) + 1, dim, name=f'emb_{nombre}')(entrada)
                vectores.append(keras.layers.Flatten()(emb))

    h = keras.layers.Concatenate()(vectores)
    h = keras.layers.Dense(256, activation='relu', name='densa_1')(h)
    h = keras.layers.Dense(256, activation='relu', name='densa_2')(h)
    h = keras.layers.Dense(256, activation='relu', name='densa_3')(h)
    # h = keras.layers.Dense(64, activation='relu', name='densa_2')(h)
    # salida = keras.layers.Dense(1, activation='sigmoid', name='densa_3')(h)
    salida = keras.layers.Dense(1, activation='sigmoid', name='densa_4')(h)
    red = keras.Model(inputs=entradas, outputs=salida)
    red.compile(
        loss=loss or keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.RootMeanSquaredError()],
        optimizer=optimizer or keras.optimizers.Adagrad(learning_rate=0.1),
    )

    def datos(u, i):
        return {nombre: valores[filas].astype(np.float32)
                for rasgos, filas in ((rasgos_lector, u), (rasgos_libro, i))
                for nombre, (valores, _) in rasgos.items()}

    if remuestrear is None:
        red.fit(datos(u, i), y, epochs=epochs, batch_size=batch_size, callbacks=list(callbacks), verbose=2)
    else:
        for _ in range(epochs):
            u, i, y = remuestrear()
            red.fit(datos(u, i), y, epochs=1, batch_size=batch_size, callbacks=list(callbacks), verbose=2)

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
        'capas': [tuple(red.get_layer(n).get_weights()) for n in ('densa_2', 'densa_3', 'densa_4')],
    }

    
def _rasgos_gran_torre(conn, ids_lectores, ids_libros):
    libros_df = pd.read_sql(QUERY_LIBROS, conn).set_index('id_libro').reindex(ids_libros)
    lectores_df = pd.read_sql(QUERY_LECTORES, conn).set_index('id_lector').reindex(ids_lectores)
    lectores = np.arange(len(ids_lectores))
    libros = np.arange(len(ids_libros))

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
    return rasgos_lector, rasgos_libro


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
        recuperacion.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adagrad(learning_rate=0.2))
        recuperacion.fit({'lector': u, 'libro': i}, y, epochs=24, batch_size=1024, verbose=2)
        m['recuperacion_lector'] = recuperacion.emb_lector.get_weights()[0]
        m['recuperacion_libro'] = recuperacion.emb_libro.get_weights()[0]

        # modelo de ranking: sólo los embeddings de lector y libro (dimensión 32)
        m.update(_entrenar_ranking(keras, {'lector': (lectores, 32)}, {'libro': (libros, 32)}, u, i, y, epochs=10, batch_size=256))
    elif nombre in RECUPERACION_V2:
        m['recuperacion_lector'], m['recuperacion_libro'] = _entrenar_recuperacion_v2(
            keras, u, i, df['rating'].to_numpy(), len(ids_lectores), len(ids_libros))
    elif nombre == 'gran_torre_v2':
        rasgos_lector, rasgos_libro = _rasgos_gran_torre(conn, ids_lectores, ids_libros)
        m.update(_entrenar_gran_torre_v2(keras, rasgos_lector, rasgos_libro, u, i, df['rating'].to_numpy(), len(ids_libros)))
    else:
        rasgos_lector, rasgos_libro = _rasgos_gran_torre(conn, ids_lectores, ids_libros)
        # como en el notebook: hasta 100 épocas, cortando cuando el loss no mejora en 3 seguidas
        corte = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        m.update(_entrenar_ranking(keras, rasgos_lector, rasgos_libro, u, i, y, epochs=100, batch_size=1024, callbacks=[corte]))

    conn.close()
    return m


def _entrenar_gran_torre_v2(keras, rasgos_lector, rasgos_libro, u, i, rating, num_libros, epochs=GT2_EPOCHS):
    # La red de gran_torre, entrenada con negativos: la salida pasa a ser la
    # probabilidad de que el lector elija el libro, no su rating. Esta red mira un
    # par (lector, libro) por vez, así que no puede comparar todo el batch como
    # dos_torres_v2: cada positivo (rating >= V2_RATING_POSITIVO) se acompaña de
    # GT2_NEGATIVOS_POR_POSITIVO libros al azar que el lector no valoró.
    #
    # Los negativos se sortean de nuevo en cada época: con negativos fijos la red
    # los memoriza y el NDCG@20 cae rápido (0,22 con 3 épocas, 0,08 con 15). Se
    # sortean con la misma probabilidad para todos los libros, así que los
    # populares siguen saliendo favorecidos; sortearlos por popularidad dio peor.
    # Los libros valorados con rating bajo no se usan como negativos: suben el
    # porcentaje de aciertos que gustan, pero aciertan menos en total.
    rng = np.random.default_rng(42)
    positivos = rating >= V2_RATING_POSITIVO
    u_pos, i_pos = u[positivos], i[positivos]
    valorados = u.astype(np.int64) * num_libros + i

    def con_negativos():
        u_neg = np.repeat(u_pos, GT2_NEGATIVOS_POR_POSITIVO)
        i_neg = rng.integers(num_libros, size=len(u_neg))
        libres = ~np.isin(u_neg.astype(np.int64) * num_libros + i_neg, valorados)
        u_neg, i_neg = u_neg[libres], i_neg[libres]
        y = np.concatenate([np.ones(len(u_pos)), np.zeros(len(u_neg))]).astype(np.float32)
        return np.concatenate([u_pos, u_neg]), np.concatenate([i_pos, i_neg]), y

    return _entrenar_ranking(
        keras, rasgos_lector, rasgos_libro, *con_negativos(), epochs=epochs, batch_size=GT2_BATCH_SIZE,
        loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=GT2_LEARNING_RATE),
        remuestrear=con_negativos,
    )


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


def _entrenar_recuperacion_v2(keras, u, i, rating, num_lectores, num_libros):
    # Recuperación con negativos, en lugar de aproximar el rating. La torre de
    # dos_torres sólo ve pares que existen, casi todos con rating alto, y nunca
    # aprende a bajarle el puntaje a un libro que el lector no eligió. Acá cada
    # ejemplo es un par positivo (lector, libro con rating >= V2_RATING_POSITIVO) y
    # la pregunta es "de los libros de este batch, cuál eligió este lector": los
    # libros de los demás pares del batch son sus negativos (softmax con negativos
    # en el batch).
    #
    # Un libro popular aparece más seguido como negativo, y eso lo castigaría de
    # más. Por eso a cada score se le resta el log de la frecuencia del libro
    # (corrección logQ) sólo al entrenar: sin ella el NDCG@20 baja de 0,36 a 0,14.
    #
    # Devuelve los embeddings normalizados, así que el producto escalar de
    # ranking() es la similitud coseno.
    positivos = rating >= V2_RATING_POSITIVO
    u, i = u[positivos], i[positivos]

    frecuencia = np.bincount(i, minlength=num_libros) + 1
    log_q = np.log(frecuencia / frecuencia.sum()).astype(np.float32)

    class RecuperacionV2(keras.Model):
        def __init__(self):
            super().__init__()
            self.emb_lector = keras.layers.Embedding(num_lectores, V2_DIM)
            self.emb_libro = keras.layers.Embedding(num_libros, V2_DIM)

        def call(self, inputs):
            ops = keras.ops
            libro = inputs['libro']
            a = ops.normalize(self.emb_lector(inputs['lector']), axis=1)
            b = ops.normalize(self.emb_libro(libro), axis=1)
            # scores[f, c]: afinidad del lector de la fila f con el libro de la columna c
            scores = ops.matmul(a, ops.transpose(b)) / V2_TEMPERATURA
            scores = scores - ops.take(ops.convert_to_tensor(log_q), libro)[None, :]
            # si el mismo libro aparece en otra fila no es un negativo de verdad
            mismo = ops.equal(libro[:, None], libro[None, :])
            otra_fila = ops.logical_not(ops.eye(ops.shape(libro)[0], dtype='bool'))
            return ops.where(ops.logical_and(mismo, otra_fila), -1e9, scores)

    def perdida(_, scores):
        # el libro correcto de cada fila es el de su misma posición en el batch
        correcto = keras.ops.arange(keras.ops.shape(scores)[0])
        return keras.ops.sparse_categorical_crossentropy(correcto, scores, from_logits=True)

    recuperacion = RecuperacionV2()
    recuperacion.compile(loss=perdida, optimizer=keras.optimizers.Adam(learning_rate=V2_LEARNING_RATE))
    sin_uso = np.zeros(len(u), dtype=np.float32)  # keras pide un y, pero la pérdida no lo usa
    recuperacion.fit({'lector': u, 'libro': i}, sin_uso, epochs=V2_EPOCHS, batch_size=V2_BATCH_SIZE, verbose=2)

    pesos = [e.get_weights()[0] for e in (recuperacion.emb_lector, recuperacion.emb_libro)]
    return tuple(w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12) for w in pesos)


if __name__ == '__main__':
    # entrena y guarda los modelos que falten: python modelos_torres.py
    import recomendador
    for n in NOMBRES:
        modelo(n, recomendador.DATABASE_FILE if n == 'dos_torres_v2_completa' else recomendador.DATABASE_ENTRENAMIENTO_FILE)
        print(n, 'listo')
