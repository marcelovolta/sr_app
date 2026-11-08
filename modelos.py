import os
import pickle
import sqlite3
import threading

import numpy as np
import pandas as pd
import scipy.sparse as sps


# Modelos de recomendación entrenados (todos menos los de SQL de recomendador.py).
# Cada modelo se guarda como un dict de arrays de NumPy con lo necesario para
# puntuar libros; la predicción no necesita Surprise ni Keras, sólo NumPy.
#
# - naive:      BaselineOnly de Surprise. Puntaje = sesgo del libro (el del lector
#               no cambia el orden), igual para todos los lectores.
# - svd:        SVD de Surprise (factorización de matrices sobre el rating).
# - ease:       EASE (Steck 2019): modelo lineal libro a libro con solución cerrada.
# - dos_torres: embeddings de lector y libro entrenados con softmax sobre los
#               negativos del batch y corrección logQ; puntaje = similitud coseno.
# - gran_torre: red densa sobre lector, libro y sus rasgos (autor, género,
#               editorial, año; género y nacimiento del lector), entrenada para
#               separar libros elegidos de libros al azar.
#
# Se probó agregarles complejidad (atributos y capas en dos torres, más capas,
# dropout, España y antigüedad del libro en gran torre) y no mejoraron: ver
# reportes/intento_complejidad.md.
#
# Los hiperparámetros de cada uno (y el filtro de la base) los elige
# optimizacion.py; acá sólo se definen el espacio de búsqueda y el entrenamiento.
NOMBRES = ('naive', 'svd', 'ease', 'dos_torres', 'gran_torre')
CARPETA_MODELOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelos')
SEMILLA = 42


# ---------------------------------------------------------------------------
# Espacios de búsqueda (Optuna)
# ---------------------------------------------------------------------------

def sugerir(nombre, trial):
    # El filtro de la base es un hiperparámetro más de cada modelo: mínimo de
    # interacciones por lector (no menos de 20) y por libro.
    p = {
        'min_lector': trial.suggest_int('min_lector', 20, 80),
        'min_libro': trial.suggest_int('min_libro', 5, 50, log=True),
    }
    if nombre == 'naive':
        p['reg_i'] = trial.suggest_float('reg_i', 0.1, 200, log=True)
        p['reg_u'] = trial.suggest_float('reg_u', 0.1, 200, log=True)
        p['n_epochs'] = trial.suggest_int('n_epochs', 3, 30)
    elif nombre == 'svd':
        p['n_factors'] = trial.suggest_int('n_factors', 10, 200, log=True)
        p['n_epochs'] = trial.suggest_int('n_epochs', 5, 50)
        p['lr_all'] = trial.suggest_float('lr_all', 1e-3, 0.05, log=True)
        p['reg_all'] = trial.suggest_float('reg_all', 0.005, 0.5, log=True)
    elif nombre == 'ease':
        p['lambda'] = trial.suggest_float('lambda', 1, 1e5, log=True)
    elif nombre == 'dos_torres':
        p['rating_positivo'] = trial.suggest_int('rating_positivo', 1, 9)
        p['dim'] = trial.suggest_categorical('dim', [32, 64, 128, 256])
        p['temperatura'] = trial.suggest_float('temperatura', 0.02, 0.5, log=True)
        p['learning_rate'] = trial.suggest_float('learning_rate', 1e-3, 0.05, log=True)
        p['epochs'] = trial.suggest_int('epochs', 3, 20)
    elif nombre == 'gran_torre':
        p['rating_positivo'] = trial.suggest_int('rating_positivo', 1, 9)
        p['negativos'] = trial.suggest_int('negativos', 2, 10)
        p['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        p['epochs'] = trial.suggest_int('epochs', 2, 10)
    return p


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def entrenar(nombre, df, params, base, al_final_de_epoca=None):
    # df: interacciones ya filtradas (id_lector, id_libro, rating). Devuelve el
    # dict del modelo. Para las redes, al_final_de_epoca(época, pérdida, modelo)
    # se llama después de cada época con una función que arma el modelo en ese
    # punto (sirve para las curvas de entrenamiento). Los índices de lectores y libros son posiciones en las
    # listas ordenadas de ids; 'leidos' sirve para no recomendar lo ya leído y
    # para puntuar con EASE, y 'popularidad' para los lectores que no conoce.
    ids_lectores = np.array(sorted(df['id_lector'].unique()))
    ids_libros = np.array(sorted(df['id_libro'].unique()))
    u = np.searchsorted(ids_lectores, df['id_lector'].to_numpy())
    i = np.searchsorted(ids_libros, df['id_libro'].to_numpy())
    rating = df['rating'].to_numpy()
    nu, nb = len(ids_lectores), len(ids_libros)
    orden = np.lexsort((i, u))
    cortes = np.searchsorted(u[orden], np.arange(nu + 1))

    m = {
        'nombre': nombre,
        'params': params,
        'ids_libros': ids_libros,
        'indice_lector': {x: n for n, x in enumerate(ids_lectores)},
        'leidos': [i[orden][cortes[n]:cortes[n + 1]] for n in range(nu)],
        'popularidad': np.bincount(i, minlength=nb).astype(np.float32),
    }
    if nombre in ('naive', 'svd'):
        m.update(_entrenar_surprise(nombre, df, params, ids_lectores, ids_libros))
    elif nombre == 'ease':
        X = sps.csr_matrix((np.ones(len(u), dtype=np.float32), (u, i)), shape=(nu, nb))
        m['B'] = _ease(X, params['lambda'])
    elif nombre in ('dos_torres', 'gran_torre'):
        aviso = None
        if al_final_de_epoca:
            def aviso(epoca, perdida, pesos):
                # completa el modelo parcial con lo que no son pesos de la red
                al_final_de_epoca(epoca, perdida, lambda: {**m, **pesos()})
        entrenar_red = _entrenar_dos_torres if nombre == 'dos_torres' else _entrenar_gran_torre
        m.update(entrenar_red(u, i, rating, ids_lectores, ids_libros, params, base, aviso))
    return m


def _entrenar_surprise(nombre, df, params, ids_lectores, ids_libros):
    # se usa Surprise para entrenar (como en clase) y se guardan sólo sus arrays,
    # reordenados según ids_lectores / ids_libros
    import surprise as sp
    datos = sp.Dataset.load_from_df(df[['id_lector', 'id_libro', 'rating']], sp.Reader(rating_scale=(1, 10)))
    ts = datos.build_full_trainset()
    fila_libro = np.array([ts.to_inner_iid(x) for x in ids_libros])
    if nombre == 'naive':
        algo = sp.BaselineOnly(bsl_options={'method': 'als', 'reg_i': params['reg_i'],
                                            'reg_u': params['reg_u'], 'n_epochs': params['n_epochs']}, verbose=False)
        algo.fit(ts)
        return {'bi': algo.bi[fila_libro].astype(np.float32)}
    algo = sp.SVD(n_factors=params['n_factors'], n_epochs=params['n_epochs'], lr_all=params['lr_all'],
                  reg_all=params['reg_all'], random_state=SEMILLA)
    algo.fit(ts)
    fila_lector = np.array([ts.to_inner_uid(x) for x in ids_lectores])
    return {'bi': algo.bi[fila_libro].astype(np.float32),
            'pu': algo.pu[fila_lector].astype(np.float32),
            'qi': algo.qi[fila_libro].astype(np.float32)}


def _ease(X, lam):
    # P = (XᵀX + λI)⁻¹,  B = I − P / diag(P),  con diag(B) = 0.
    # X binaria: cuenta que el lector leyó el libro, no el rating
    G = (X.T @ X).toarray().astype(np.float64)
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G)
    B = -P / np.diag(P)
    B[np.diag_indices_from(B)] = 0
    return B.astype(np.float32)


def _keras():
    # Keras (backend JAX) sólo hace falta para entrenar las torres
    os.environ.setdefault('KERAS_BACKEND', 'jax')
    import keras
    keras.utils.set_random_seed(SEMILLA)
    return keras


def _entrenar_dos_torres(u, i, rating, ids_lectores, ids_libros, params, base, al_final_de_epoca):
    # Cada ejemplo es un par (lector, libro elegido): elegido = rating >=
    # rating_positivo. La pérdida pregunta "de los libros del batch, cuál eligió
    # este lector": los de las otras filas son sus negativos. Como un libro popular
    # aparece más como negativo, se le resta el log de su frecuencia (corrección
    # logQ) al entrenar; sin ella el modelo se va a los populares.
    keras = _keras()
    ops = keras.ops
    nu, nb = len(ids_lectores), len(ids_libros)
    positivos = rating >= params['rating_positivo']
    u, i = u[positivos], i[positivos]
    frecuencia = np.bincount(i, minlength=nb) + 1
    log_q = np.log(frecuencia / frecuencia.sum()).astype(np.float32)
    dim, temperatura = params['dim'], params['temperatura']

    class DosTorres(keras.Model):
        # una tabla de embeddings para lectores y otra para libros
        def __init__(self):
            # crea las dos tablas de embeddings
            super().__init__()
            self.emb_lector = keras.layers.Embedding(nu, dim)
            self.emb_libro = keras.layers.Embedding(nb, dim)

        def call(self, inputs):
            # puntajes de cada lector del batch contra cada libro del batch
            libro = inputs['libro']
            a = ops.normalize(self.emb_lector(inputs['lector']), axis=1)
            b = ops.normalize(self.emb_libro(libro), axis=1)
            scores = ops.matmul(a, ops.transpose(b)) / temperatura
            scores = scores - ops.take(ops.convert_to_tensor(log_q), libro)[None, :]
            # el mismo libro en otra fila no es un negativo de verdad
            mismo = ops.equal(libro[:, None], libro[None, :])
            otra_fila = ops.logical_not(ops.eye(ops.shape(libro)[0], dtype='bool'))
            return ops.where(ops.logical_and(mismo, otra_fila), -1e9, scores)

    def perdida(_, scores):
        # softmax: para cada fila, el correcto es el libro de esa misma fila
        correcto = ops.arange(ops.shape(scores)[0])
        return ops.sparse_categorical_crossentropy(correcto, scores, from_logits=True)

    def embeddings():
        # normalizados: el producto escalar al predecir es la similitud coseno
        pesos = [e.get_weights()[0] for e in (modelo.emb_lector, modelo.emb_libro)]
        return {f'emb_{k}': (w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)).astype(np.float32)
                for k, w in zip(('lector', 'libro'), pesos)}

    modelo = DosTorres()
    modelo.compile(loss=perdida, optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']))
    for epoca in range(1, params['epochs'] + 1):
        h = modelo.fit({'lector': u, 'libro': i}, np.zeros(len(u), dtype=np.float32), epochs=1, batch_size=1024, verbose=0)
        if al_final_de_epoca:
            al_final_de_epoca(epoca, h.history['loss'][-1], embeddings)
    return embeddings()


def _rasgos_gran_torre(base, ids_lectores, ids_libros):
    # {rasgo: (valor por lector/libro, dimensión del embedding o None si es numérico)}
    with sqlite3.connect(base) as conn:
        libros = pd.read_sql('SELECT id_libro, autor, genero, editorial, anio_edicion FROM libros', conn)
        lectores = pd.read_sql('SELECT id_lector, genero, nacimiento FROM lectores', conn)
    libros = libros.drop_duplicates('id_libro').set_index('id_libro').reindex(ids_libros)
    lectores = lectores.drop_duplicates('id_lector').set_index('id_lector').reindex(ids_lectores)

    def categorico(serie):
        # los nulos van a un código propio
        codigos = serie.astype('category').cat.codes.to_numpy().astype(np.int64)
        n = int(codigos.max()) + 2
        codigos[codigos < 0] = n - 1
        return codigos, min(n // 2 + 1, 50)

    def anio(serie):
        # año a número, nulos en 0, escalado entre 0 y 1
        x = pd.to_numeric(serie, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        return (x - x.min()) / max(x.max() - x.min(), 1.0), None

    rasgos_lector = {
        'lector': (np.arange(len(ids_lectores)), min(len(ids_lectores) // 2 + 1, 50)),
        'genero_lector': categorico(lectores['genero']),
        'nacimiento': anio(lectores['nacimiento']),
    }
    rasgos_libro = {
        'libro': (np.arange(len(ids_libros)), min(len(ids_libros) // 2 + 1, 50)),
        'autor': categorico(libros['autor']),
        'genero': categorico(libros['genero']),
        'editorial': categorico(libros['editorial']),
        'anio_edicion': anio(libros['anio_edicion']),
    }
    return rasgos_lector, rasgos_libro


def _entrenar_gran_torre(u, i, rating, ids_lectores, ids_libros, params, base, al_final_de_epoca):
    # La red mira un par (lector, libro) por vez y estima la probabilidad de que
    # lo elija. Cada positivo (rating >= rating_positivo) va con `negativos`
    # libros al azar que el lector no leyó, sorteados de nuevo en cada época (con
    # negativos fijos la red los memoriza).
    keras = _keras()
    rasgos_lector, rasgos_libro = _rasgos_gran_torre(base, ids_lectores, ids_libros)
    nb = len(ids_libros)
    rng = np.random.default_rng(SEMILLA)
    positivos = rating >= params['rating_positivo']
    u_pos, i_pos = u[positivos], i[positivos]
    leidos = u.astype(np.int64) * nb + i

    def con_negativos():
        # los positivos más negativos sorteados de nuevo (uno por época)
        u_neg = np.repeat(u_pos, params['negativos'])
        i_neg = rng.integers(nb, size=len(u_neg))
        libres = ~np.isin(u_neg.astype(np.int64) * nb + i_neg, leidos)
        u_neg, i_neg = u_neg[libres], i_neg[libres]
        y = np.concatenate([np.ones(len(u_pos)), np.zeros(len(u_neg))]).astype(np.float32)
        return np.concatenate([u_pos, u_neg]), np.concatenate([i_pos, i_neg]), y

    # embeddings de cada rasgo concatenados -> 3 capas densas de 256 -> sigmoide
    entradas, vectores = [], []
    for rasgos in (rasgos_lector, rasgos_libro):
        for nombre, (valores, dim) in rasgos.items():
            entrada = keras.layers.Input(shape=(1,), name=nombre)
            entradas.append(entrada)
            if dim is None:
                vectores.append(entrada)
            else:
                emb = keras.layers.Embedding(int(valores.max()) + 1, dim, name=f'emb_{nombre}')(entrada)
                vectores.append(keras.layers.Flatten()(emb))
    h = keras.layers.Concatenate()(vectores)
    for n in (1, 2, 3):
        h = keras.layers.Dense(256, activation='relu', name=f'densa_{n}')(h)
    salida = keras.layers.Dense(1, activation='sigmoid', name='salida')(h)
    red = keras.Model(inputs=entradas, outputs=salida)
    red.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']))

    def pesos():
        # La primera capa es lineal sobre [rasgos del lector, rasgos del libro]: se
        # precalcula la parte de cada uno, y al predecir sólo se suman (mucho más
        # rápido que pasar cada par por la red)
        w1, b1 = red.get_layer('densa_1').get_weights()
        partes, fila = [], 0
        for rasgos in (rasgos_lector, rasgos_libro):
            bloques = [valores[:, None].astype(np.float32) if dim is None
                       else red.get_layer(f'emb_{nombre}').get_weights()[0][valores]
                       for nombre, (valores, dim) in rasgos.items()]
            X = np.concatenate(bloques, axis=1)
            partes.append(X @ w1[fila:fila + X.shape[1]])
            fila += X.shape[1]
        return {
            'parte_lector': (partes[0] + b1).astype(np.float32),
            'parte_libro': partes[1].astype(np.float32),
            'capas': [tuple(w.astype(np.float32) for w in red.get_layer(n).get_weights())
                      for n in ('densa_2', 'densa_3', 'salida')],
        }

    for epoca in range(1, params['epochs'] + 1):
        uu, ii, y = con_negativos()
        x = {nombre: valores[filas].astype(np.float32)
             for rasgos, filas in ((rasgos_lector, uu), (rasgos_libro, ii))
             for nombre, (valores, _) in rasgos.items()}
        h = red.fit(x, y, epochs=1, batch_size=1024, verbose=0)
        if al_final_de_epoca:
            al_final_de_epoca(epoca, h.history['loss'][-1], pesos)
    return pesos()


# ---------------------------------------------------------------------------
# Predicción (sólo NumPy)
# ---------------------------------------------------------------------------

def puntajes(m, lectores):
    # puntaje de cada libro del modelo para cada lector (índices internos):
    # matriz len(lectores) × libros. Mayor = más recomendable
    lectores = np.asarray(lectores)
    nombre = m['nombre']
    if nombre == 'naive':
        return np.tile(m['bi'], (len(lectores), 1))
    if nombre == 'svd':
        return m['bi'] + m['pu'][lectores] @ m['qi'].T
    if nombre == 'ease':
        nb = len(m['ids_libros'])
        filas = [m['leidos'][u] for u in lectores]
        X = sps.csr_matrix((np.ones(sum(map(len, filas)), dtype=np.float32), np.concatenate(filas),
                            np.cumsum([0] + [len(f) for f in filas])), shape=(len(lectores), nb))
        return np.asarray(X @ m['B'])
    if nombre == 'dos_torres':
        return m['emb_lector'][lectores] @ m['emb_libro'].T
    # gran_torre: las capas densas en lotes, para no armar matrices enormes
    nb = len(m['ids_libros'])
    salida = np.empty((len(lectores), nb), dtype=np.float32)
    lote = max(1, 250_000 // nb)
    *ocultas, (w_salida, b_salida) = m['capas']
    for desde in range(0, len(lectores), lote):
        us = lectores[desde:desde + lote]
        h = np.maximum(m['parte_lector'][us][:, None, :] + m['parte_libro'][None, :, :], 0).reshape(-1, m['parte_libro'].shape[1])
        for w, b in ocultas:
            h = np.maximum(h @ w + b, 0)
        salida[desde:desde + lote] = (h @ w_salida + b_salida).reshape(len(us), nb)  # logit: mismo orden que la sigmoide
    return salida


_modelos = {}
_lock = threading.Lock()


def guardar(m):
    # guarda el modelo en modelos/<nombre>.pkl y descarta la copia en memoria
    os.makedirs(CARPETA_MODELOS, exist_ok=True)
    archivo = os.path.join(CARPETA_MODELOS, f"{m['nombre']}.pkl")
    with open(archivo + '.tmp', 'wb') as f:
        pickle.dump(m, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(archivo + '.tmp', archivo)  # un corte a mitad no deja un .pkl roto
    with _lock:
        _modelos.pop(m['nombre'], None)
    return archivo


def modelo(nombre):
    # se carga la primera vez que se usa y queda en memoria. Los modelos no se
    # entrenan solos: hay que correr pipeline.py
    with _lock:
        if nombre not in _modelos:
            archivo = os.path.join(CARPETA_MODELOS, f'{nombre}.pkl')
            if not os.path.exists(archivo):
                raise FileNotFoundError(f'Falta {archivo}: correr python pipeline.py')
            with open(archivo, 'rb') as f:
                _modelos[nombre] = pickle.load(f)
        return _modelos[nombre]


def conoce_lector(nombre, id_lector):
    # False si el lector quedó afuera del filtro al entrenar
    return id_lector in modelo(nombre)['indice_lector']


def ranking(nombre, id_lector):
    # libros del modelo de mayor a menor puntaje. A un lector que no conoce le
    # da los más leídos del modelo (en las cascadas esos lectores van a visión de túnel)
    m = modelo(nombre)
    u = m['indice_lector'].get(id_lector)
    p = m['popularidad'] if u is None else puntajes(m, [u])[0]
    for j in np.argsort(-p, kind='stable'):
        yield m['ids_libros'][j], p[j]
