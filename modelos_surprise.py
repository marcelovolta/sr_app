import os
import pickle
import sqlite3
import threading

import numpy as np
import pandas as pd
import surprise as sp


# Algoritmos de la librería Surprise, con los mismos hiperparámetros que en
# clase09_suprise.ipynb. KNN usa KNNWithMeans, que en el notebook tuvo mejor
# RMSE que KNNBasic.
CONSTRUCTORES = {
    'baseline': lambda: sp.BaselineOnly(verbose=False),
    'knn': lambda: sp.KNNWithMeans(k=40, min_k=5, sim_options={'name': 'pearson', 'user_based': True}, verbose=False),
    'svd': lambda: sp.SVD(n_factors=500, n_epochs=20, random_state=42),
}

# Los modelos se entrenan sólo con los libros y lectores con muchas valoraciones,
# que dan mejores resultados que entrenar con todos: los libros (de la tabla
# libros) con más de MIN_VALORACIONES valoraciones, y los lectores (de la tabla
# lectores) con más de MIN_VALORACIONES valoraciones de esos libros. Se recomienda
# igual a todos los lectores: los que quedan afuera reciben el ranking de Baseline
MIN_VALORACIONES = 20

QUERY_INTERACCIONES = f"""
    WITH libros_entrenamiento AS (
        SELECT id_libro FROM interacciones
        WHERE id_libro IN (SELECT id_libro FROM libros)
        GROUP BY id_libro HAVING COUNT(*) > {MIN_VALORACIONES}
    ),
    lectores_entrenamiento AS (
        SELECT id_lector FROM interacciones
        WHERE id_libro IN libros_entrenamiento AND id_lector IN (SELECT id_lector FROM lectores)
        GROUP BY id_lector HAVING COUNT(*) > {MIN_VALORACIONES}
    )
    SELECT id_lector, id_libro, rating FROM interacciones
    WHERE COALESCE(rating, 0) > 0
    AND id_libro IN libros_entrenamiento AND id_lector IN lectores_entrenamiento
"""

# como en el notebook, KNN además se queda sólo con los lectores y libros con más
# de 20 valoraciones dentro de ese conjunto
QUERY_INTERACCIONES_KNN = f"""
    WITH entrenamiento AS ({QUERY_INTERACCIONES})
    SELECT * FROM entrenamiento
    WHERE id_libro IN (SELECT id_libro FROM entrenamiento GROUP BY 1 HAVING COUNT(*) > 20)
    AND id_lector IN (SELECT id_lector FROM entrenamiento GROUP BY 1 HAVING COUNT(*) > 20)
"""

# los modelos entrenados se guardan con pickle en esta carpeta, uno por algoritmo
CARPETA_MODELOS = os.path.join(os.path.dirname(__file__), 'modelos')

_modelos = {}
_lock = threading.Lock()


def _entrenar(nombre, database_file):
    conn = sqlite3.connect(database_file)
    df = pd.read_sql(QUERY_INTERACCIONES_KNN if nombre == 'knn' else QUERY_INTERACCIONES, conn)
    conn.close()

    reader = sp.Reader(rating_scale=(1, 10))
    data = sp.Dataset.load_from_df(df[['id_lector', 'id_libro', 'rating']], reader) # cuidado con el orden
    algo = CONSTRUCTORES[nombre]()
    algo.fit(data.build_full_trainset())
    return algo


def _cargar_o_entrenar(nombre, database_file):
    # si ya hay un modelo guardado se usa ese; si no, se entrena y se guarda.
    # Para reentrenar (por ejemplo, con valoraciones nuevas) hay que borrar el .pkl
    archivo = os.path.join(CARPETA_MODELOS, f'{nombre}.pkl')
    if os.path.exists(archivo):
        with open(archivo, 'rb') as f:
            return pickle.load(f)

    algo = _entrenar(nombre, database_file)
    os.makedirs(CARPETA_MODELOS, exist_ok=True)
    # se escribe a un archivo temporal y después se renombra, para que un corte a
    # mitad de camino no deje un .pkl incompleto
    with open(archivo + '.tmp', 'wb') as f:
        pickle.dump(algo, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(archivo + '.tmp', archivo)
    return algo


def modelo(nombre, database_file):
    # se carga (o entrena) la primera vez que se usa y queda en memoria; el lock
    # evita que los hilos de la API de predicción lo carguen varias veces a la vez
    with _lock:
        if nombre not in _modelos:
            _modelos[nombre] = _cargar_o_entrenar(nombre, database_file)
        return _modelos[nombre]


def conoce_lector(nombre, database_file, id_lector):
    # False si el lector no estaba al entrenar
    try:
        modelo(nombre, database_file).trainset.to_inner_uid(id_lector)
    except ValueError:
        return False
    return True


def _estimaciones(nombre, algo, id_lector):
    # rating estimado para cada libro del trainset, indexado por id interno
    trainset = algo.trainset
    if nombre == 'knn':
        return np.array([algo.predict(id_lector, trainset.to_raw_iid(i)).est for i in trainset.all_items()])

    try:
        u = trainset.to_inner_uid(id_lector)
    except ValueError:
        u = None

    # Baseline y SVD: la misma cuenta que hace algo.predict(), pero para todos los
    # libros a la vez (llamar a predict() libro por libro tarda ~0.5 s por lector).
    # Si el lector no está en el modelo, su sesgo y sus factores cuentan como 0
    est = trainset.global_mean + algo.bi
    if u is not None:
        est = est + algo.bu[u]
        if nombre == 'svd':
            est = est + algo.qi @ algo.pu[u]
    return est


def ranking(nombre, database_file, id_lector):
    # recorre los libros que conoce el modelo, de mayor a menor rating estimado
    algo = modelo(nombre, database_file)
    trainset = algo.trainset

    # KNN no puede estimar nada para un lector que no está en su matriz de
    # similitud (devolvería la media global para todos los libros), así que se usa Baseline
    if nombre == 'knn':
        try:
            trainset.to_inner_uid(id_lector)
        except ValueError:
            yield from ranking('baseline', database_file, id_lector)
            return

    est = _estimaciones(nombre, algo, id_lector)
    for i in np.argsort(-est, kind='stable'):
        yield trainset.to_raw_iid(int(i)), est[i]
