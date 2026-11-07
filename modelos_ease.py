import os
import pickle
import sqlite3
import threading

import numpy as np
import pandas as pd

import modelos_surprise


# EASE (Embarrassingly Shallow Autoencoder, Steck 2019). Un modelo lineal libro a
# libro: el puntaje de un libro para un lector es la suma de los pesos B[j, i] de
# los libros j que leyó. B tiene solución cerrada:
#
#     P = (XᵀX + λI)⁻¹        B = I - P / diag(P)     (con diag(B) = 0)
#
# donde X es la matriz lector × libro de interacciones (1 si lo leyó, sin importar
# el rating). El único hiperparámetro es la regularización λ, que se elige con
# Optuna sobre una partición temporal en tres tramos, como en
# https://www.stepbystepdatascience.com/ease (ver optimizar):
#   1. se entrena con train y se elige el λ con mejor NDCG@20 en valid;
#   2. con ese λ se entrena con train + valid y se mide en test, el tramo más
#      reciente, que la búsqueda no vio (así el resultado no queda inflado);
#   3. con ese λ se entrena el modelo final con todas las interacciones.
#
# Usa los mismos libros y lectores que los modelos de Surprise y de torres
# (modelos_surprise.QUERY_INTERACCIONES sobre la base reducida). Como no tiene
# embeddings de lectores, para recomendar sólo necesita los libros que leyó el
# lector, que se guardan junto con B.
NOMBRES = ('ease',)

PROPORCION_VALID = 0.15    # tramos temporales: train, valid (15%) y test (15%, el más reciente)
PROPORCION_TEST = 0.15
RATING_RELEVANTE = 8       # en valid y test, desde este rating el libro cuenta como acierto
K = 20                     # se optimiza NDCG@20, como en las comparaciones anteriores
LAMBDA_MIN, LAMBDA_MAX = 0.1, 1e5
LAMBDAS_INICIALES = (10, 100, 1000, 10000)  # se prueban primero, para orientar la búsqueda
OPTUNA_TRIALS = 50
SEMILLA = 42

QUERY_INTERACCIONES = f"""
    WITH entrenamiento AS ({modelos_surprise.QUERY_INTERACCIONES})
    SELECT e.id_lector, e.id_libro, e.rating, i.fecha
    FROM entrenamiento e JOIN interacciones i USING (id_lector, id_libro)
"""

_modelos = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def _pesos(X, lam):
    # B de EASE para la matriz binaria X (lectores × libros)
    G = (X.T @ X).astype(np.float64)
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G)
    B = -P / np.diag(P)
    B[np.diag_indices_from(B)] = 0
    return B.astype(np.float32)


def _matriz(u, i, num_lectores, num_libros):
    X = np.zeros((num_lectores, num_libros), dtype=np.float32)
    X[u, i] = 1
    return X


# ---------------------------------------------------------------------------
# Partición temporal y evaluación
# ---------------------------------------------------------------------------

def _leer(database_file):
    conn = sqlite3.connect(database_file)
    df = pd.read_sql(QUERY_INTERACCIONES, conn)
    conn.close()
    # casi todas las fechas son dd-mm-aaaa; las pocas que no se pueden leer
    # quedan en el entrenamiento
    fecha = pd.to_datetime(df['fecha'], format='%d-%m-%Y', errors='coerce')
    otra = pd.to_datetime(df['fecha'], format='ISO8601', errors='coerce')
    df['fecha'] = fecha.fillna(otra).fillna(pd.Timestamp.min)
    return df


def particion_temporal(df):
    # cortes globales por fecha (no por lector): train < corte_valid <= valid <
    # corte_test <= test, así el modelo nunca ve el futuro de lo que se evalúa
    corte_valid = df['fecha'].quantile(1 - PROPORCION_VALID - PROPORCION_TEST)
    corte_test = df['fecha'].quantile(1 - PROPORCION_TEST)
    train = df[df['fecha'] < corte_valid]
    valid = df[(df['fecha'] >= corte_valid) & (df['fecha'] < corte_test)]
    test = df[df['fecha'] >= corte_test]
    return train, valid, test, corte_valid, corte_test


class Evaluador:
    # entrena EASE con train y mide NDCG@K, Recall@K y cobertura en test. Los libros
    # que el lector ya leyó en train no se recomiendan
    def __init__(self, train, test):
        self.ids_lectores = np.array(sorted(train['id_lector'].unique()))
        self.ids_libros = np.array(sorted(train['id_libro'].unique()))
        lector = {x: n for n, x in enumerate(self.ids_lectores)}
        libro = {x: n for n, x in enumerate(self.ids_libros)}
        # en la evaluación quedan sólo los lectores y libros que están en train
        # (a los demás EASE no los puede evaluar)
        test = test[test['id_lector'].isin(lector) & test['id_libro'].isin(libro)]
        self.X = _matriz(train['id_lector'].map(lector).to_numpy(), train['id_libro'].map(libro).to_numpy(),
                         len(self.ids_lectores), len(self.ids_libros))
        self.G = (self.X.T @ self.X).astype(np.float64)

        relevantes = test[test['rating'] >= RATING_RELEVANTE]
        self.relevantes = {lector[l]: {libro[b] for b in g} for l, g in relevantes.groupby('id_lector')['id_libro']}
        self.usuarios = np.array(sorted(self.relevantes))
        self.descuento = 1 / np.log2(np.arange(2, K + 2))

    def top(self, puntajes):
        puntajes = puntajes.copy()
        puntajes[self.X[self.usuarios] > 0] = -np.inf
        return np.argsort(-puntajes, axis=1)[:, :K]

    def metricas(self, top):
        ndcg, recall, precision = [], [], []
        for fila, u in zip(top, self.usuarios):
            rel = self.relevantes[u]
            aciertos = np.array([j in rel for j in fila], dtype=float)
            ideal = self.descuento[:min(len(rel), K)].sum()
            ndcg.append((aciertos * self.descuento).sum() / ideal)
            recall.append(aciertos.sum() / min(len(rel), K))
            precision.append(aciertos[:10].mean())
        return {'ndcg': float(np.mean(ndcg)), 'recall': float(np.mean(recall)), 'precision@10': float(np.mean(precision)),
                'lectores': len(self.usuarios),
                'cobertura': len(np.unique(top)) / len(self.ids_libros)}

    def ease(self, lam):
        G = self.G.copy()
        G[np.diag_indices_from(G)] += lam
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        B[np.diag_indices_from(B)] = 0
        return self.metricas(self.top(self.X[self.usuarios] @ B))

    def popularidad(self):
        # referencia: los libros más leídos en train, iguales para todos
        return self.metricas(self.top(np.tile(self.X.sum(0), (len(self.usuarios), 1))))


def optimizar(database_file, trials=OPTUNA_TRIALS):
    # elige λ en valid con Optuna y lo mide en test
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train, valid, test, corte_valid, corte_test = particion_temporal(_leer(database_file))
    print(f'train hasta {corte_valid:%d-%m-%Y}: {len(train)} interacciones; '
          f'valid hasta {corte_test:%d-%m-%Y}: {len(valid)}; test: {len(test)}')

    ev_valid = Evaluador(train, valid)

    def objetivo(trial):
        m = ev_valid.ease(trial.suggest_float('lambda', LAMBDA_MIN, LAMBDA_MAX, log=True))
        for k, v in m.items():
            trial.set_user_attr(k, v)
        return m['ndcg']

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEMILLA))
    for lam in LAMBDAS_INICIALES:
        study.enqueue_trial({'lambda': lam})
    study.optimize(objetivo, n_trials=trials)
    lam = study.best_params['lambda']

    ev_test = Evaluador(pd.concat([train, valid]), test)
    resultado = {
        'cortes': (corte_valid, corte_test),
        'lambda': lam,
        'valid': study.best_trial.user_attrs,
        'test': ev_test.ease(lam),
        'popularidad_valid': ev_valid.popularidad(),
        'popularidad_test': ev_test.popularidad(),
        'trials': [(t.params['lambda'], t.user_attrs['ndcg']) for t in study.trials],
    }
    print(f"mejor λ {lam:.1f}")
    for k in ('valid', 'popularidad_valid', 'test', 'popularidad_test'):
        print(f"  {k:18}", {m: round(v, 4) for m, v in resultado[k].items()})
    return resultado


# ---------------------------------------------------------------------------
# Entrenamiento final y predicción
# ---------------------------------------------------------------------------

def _entrenar(nombre, database_file):
    optimizacion = optimizar(database_file)

    # modelo final: el λ elegido, con todas las interacciones
    df = _leer(database_file)
    ids_lectores = sorted(df['id_lector'].unique())
    ids_libros = sorted(df['id_libro'].unique())
    indice_lector = {x: n for n, x in enumerate(ids_lectores)}
    indice_libro = {x: n for n, x in enumerate(ids_libros)}
    u = df['id_lector'].map(indice_lector).to_numpy()
    i = df['id_libro'].map(indice_libro).to_numpy()
    X = _matriz(u, i, len(ids_lectores), len(ids_libros))

    leidos = pd.Series(i).groupby(u).apply(lambda s: np.sort(s.to_numpy()))
    return {
        'B': _pesos(X, optimizacion['lambda']),
        'ids_libros': ids_libros,
        'indice_lector': indice_lector,
        'leidos': [leidos[n] for n in range(len(ids_lectores))],
        'optimizacion': optimizacion,
    }


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


def modelo(nombre, database_file):
    # se carga (o entrena) la primera vez que se usa y queda en memoria
    with _lock:
        if nombre not in _modelos:
            _modelos[nombre] = _cargar_o_entrenar(nombre, database_file)
        return _modelos[nombre]


def conoce_lector(nombre, database_file, id_lector):
    # False si el lector no estaba al entrenar
    return id_lector in modelo(nombre, database_file)['indice_lector']


def ranking(nombre, database_file, id_lector):
    # recorre los libros de mayor a menor puntaje. Los lectores que no estaban al
    # entrenar reciben el ranking de Baseline, como en los otros modelos
    m = modelo(nombre, database_file)
    u = m['indice_lector'].get(id_lector)
    if u is None:
        yield from modelos_surprise.ranking('baseline', database_file, id_lector)
        return

    puntaje = m['B'][m['leidos'][u]].sum(0)
    for j in np.argsort(-puntaje, kind='stable'):
        yield m['ids_libros'][j], puntaje[j]


if __name__ == '__main__':
    # optimiza λ, entrena y guarda el modelo: python modelos_ease.py
    import recomendador
    modelo('ease', recomendador.DATABASE_ENTRENAMIENTO_FILE)
    print('ease listo')
