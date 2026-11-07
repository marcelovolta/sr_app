import os
import sqlite3
import time
import warnings

import numpy as np
import pandas as pd
import surprise as sp

import recomendador
import modelos_ease
import modelos_surprise
import modelos_torres


# Compara los modelos con la misma partición temporal que usa EASE
# (modelos_ease.particion_temporal): todos entrenan con train + valid y se miden
# en test, el 15% de interacciones más recientes, con las mismas métricas
# (modelos_ease.Evaluador): NDCG@20, Recall@20, Precision@10, cobertura y top-1.
# Los modelos se entrenan de nuevo con estos datos, con los mismos
# hiperparámetros que en modelos_surprise y modelos_torres; no se usan ni se
# modifican los .pkl, salvo el de EASE, del que se toma el λ elegido.
#
#     python evaluacion_temporal.py
#
# Imprime la tabla y la guarda en data/evaluacion_temporal.csv. Tarda unos minutos.
ARCHIVO_RESULTADOS = os.path.join(os.path.dirname(__file__), 'data', 'evaluacion_temporal.csv')


def evaluar(database_file=recomendador.DATABASE_ENTRENAMIENTO_FILE):
    train, valid, test, _, corte_test = modelos_ease.particion_temporal(modelos_ease._leer(database_file))
    train = pd.concat([train, valid])
    ev = modelos_ease.Evaluador(train, test)

    lector = {x: n for n, x in enumerate(ev.ids_lectores)}
    libro = {x: n for n, x in enumerate(ev.ids_libros)}
    u = train['id_lector'].map(lector).to_numpy()
    i = train['id_libro'].map(libro).to_numpy()
    rating = train['rating'].to_numpy()
    y = ((rating - modelos_torres.RATING_MIN) / (modelos_torres.RATING_MAX - modelos_torres.RATING_MIN)).astype(np.float32)
    num_lectores, num_libros = len(ev.ids_lectores), len(ev.ids_libros)
    evaluados = ev.usuarios
    todos = np.arange(num_libros)
    print(f'train + valid hasta {corte_test:%d-%m-%Y}: {len(train)} interacciones, {num_lectores} lectores, '
          f'{num_libros} libros; test: {len(test)}; lectores evaluados: {len(evaluados)}', flush=True)

    filas = []

    def registrar(nombre, puntajes, inicio):
        # puntajes: lectores evaluados × libros
        top = ev.top(np.asarray(puntajes, dtype=np.float64))
        m = ev.metricas(top)
        m['top1'] = float(np.bincount(top[:, 0]).max() / len(evaluados))
        m['segundos'] = round(time.time() - inicio)
        filas.append({'modelo': nombre, **m})
        print(nombre, {k: round(v, 4) for k, v in m.items()}, flush=True)

    inicio = time.time()
    registrar('popularidad', np.tile(ev.X.sum(0), (len(evaluados), 1)), inicio)

    inicio = time.time()
    lam = modelos_ease.modelo('ease', database_file)['optimizacion']['lambda']
    registrar('ease', ev.X[evaluados] @ modelos_ease._pesos(ev.X, lam), inicio)

    inicio = time.time()
    trainset = sp.Dataset.load_from_df(train[['id_lector', 'id_libro', 'rating']],
                                       sp.Reader(rating_scale=(1, 10))).build_full_trainset()
    svd = modelos_surprise.CONSTRUCTORES['svd']()
    svd.fit(trainset)
    columnas = np.array([libro[trainset.to_raw_iid(k)] for k in trainset.all_items()])
    puntajes = np.empty((len(evaluados), num_libros))
    for n, a in enumerate(evaluados):
        puntajes[n, columnas] = modelos_surprise._estimaciones('svd', svd, ev.ids_lectores[a])
    registrar('svd', puntajes, inicio)

    keras = modelos_torres._keras()

    inicio = time.time()
    emb_lector, emb_libro = modelos_torres._entrenar_recuperacion_v2(keras, u, i, rating, num_lectores, num_libros)
    registrar('dos_torres_v2', emb_lector[evaluados] @ emb_libro.T, inicio)

    conn = sqlite3.connect(database_file)
    rasgos_lector, rasgos_libro = modelos_torres._rasgos_gran_torre(conn, ev.ids_lectores, ev.ids_libros)
    conn.close()

    inicio = time.time()
    m = modelos_torres._entrenar_gran_torre_v2(keras, rasgos_lector, rasgos_libro, u, i, rating, num_libros)
    registrar('gran_torre_v2', np.stack([modelos_torres._puntuar(m, a, todos) for a in evaluados]), inicio)

    inicio = time.time()
    corte = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    m = modelos_torres._entrenar_ranking(keras, rasgos_lector, rasgos_libro, u, i, y, epochs=100, batch_size=1024, callbacks=[corte])
    registrar('gran_torre', np.stack([modelos_torres._puntuar(m, a, todos) for a in evaluados]), inicio)

    # dos_torres: la recuperación elige 100 candidatos no leídos y el ranking los ordena
    inicio = time.time()
    recuperacion = modelos_torres._modelo_recuperacion(keras, num_lectores, num_libros, 256)
    recuperacion.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adagrad(learning_rate=0.2))
    recuperacion.fit({'lector': u, 'libro': i}, y, epochs=24, batch_size=1024, verbose=0)
    rec_lector, rec_libro = (e.get_weights()[0] for e in (recuperacion.emb_lector, recuperacion.emb_libro))
    m = modelos_torres._entrenar_ranking(keras, {'lector': (np.arange(num_lectores), 32)}, {'libro': (todos, 32)},
                                         u, i, y, epochs=10, batch_size=256)
    puntajes = np.full((len(evaluados), num_libros), -np.inf)
    for n, a in enumerate(evaluados):
        afinidad = rec_libro @ rec_lector[a]
        afinidad[ev.X[a] > 0] = -np.inf
        candidatos = np.argsort(-afinidad, kind='stable')[:modelos_torres.CANDIDATOS_DOS_TORRES]
        puntajes[n, candidatos] = modelos_torres._puntuar(m, a, candidatos)
    registrar('dos_torres', puntajes, inicio)

    return pd.DataFrame(filas).sort_values('ndcg', ascending=False)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    resultados = evaluar()
    resultados.to_csv(ARCHIVO_RESULTADOS, index=False)
    pd.set_option('display.width', 200)
    print(resultados.round(4).to_string(index=False))
