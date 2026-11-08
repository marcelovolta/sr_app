import os
import shutil
import sqlite3
import time

import pandas as pd


# Datos para entrenar y evaluar los modelos: copia de la base, lectura de las
# interacciones en orden cronológico, filtro recursivo y partición train/test.
RAIZ = os.path.dirname(os.path.abspath(__file__))
BASE_ORIGEN = os.path.join(RAIZ, 'data', 'data.db')
BASE_TRABAJO = os.path.join(RAIZ, 'data', 'trabajo', 'data.db')
N_TEST = 20  # las últimas 20 interacciones de cada lector son su test


def copiar_base(origen=BASE_ORIGEN, destino=BASE_TRABAJO, intentos=30):
    # La base de origen la está modificando otro proceso y no se puede bloquear:
    # se copia el archivo sin abrirlo con SQLite (sin locks) y se acepta la copia
    # sólo si el archivo no cambió durante la copia, no había una transacción a
    # medio escribir (archivo -journal) y la copia pasa el chequeo de integridad.
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    for _ in range(intentos):
        antes = os.stat(origen)
        en_transaccion = os.path.exists(origen + '-journal')
        shutil.copyfile(origen, destino + '.tmp')
        despues = os.stat(origen)
        if not en_transaccion and not os.path.exists(origen + '-journal') \
                and (antes.st_mtime_ns, antes.st_size) == (despues.st_mtime_ns, despues.st_size):
            with sqlite3.connect(destino + '.tmp') as conn:
                ok = conn.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
            if ok:
                os.replace(destino + '.tmp', destino)
                return destino
        time.sleep(2)
    raise RuntimeError(f'No se pudo obtener una copia consistente de {origen}')


def leer_interacciones(base=BASE_TRABAJO):
    # Valoraciones (rating > 0) de libros y lectores que existen en sus tablas,
    # ordenadas cronológicamente por lector. Las fechas son por día y hay muchos
    # empates (cargas masivas), así que se desempata por orden de inserción (rowid).
    # Las fechas que no se pueden leer van al principio.
    # (el filtro por libros y lectores se hace en pandas: con IN (SELECT ...) SQLite
    # elige un plan que cruza todos los lectores con todos los libros)
    with sqlite3.connect(base) as conn:
        df = pd.read_sql('SELECT rowid AS orden, id_lector, id_libro, rating, fecha FROM interacciones '
                         'WHERE COALESCE(rating, 0) > 0', conn)
        libros = pd.read_sql('SELECT id_libro FROM libros', conn)['id_libro']
        lectores = pd.read_sql('SELECT id_lector FROM lectores', conn)['id_lector']
    df = df[df['id_libro'].isin(libros) & df['id_lector'].isin(lectores)]
    fecha = pd.to_datetime(df['fecha'], format='%d-%m-%Y', errors='coerce')
    df['fecha'] = fecha.fillna(pd.to_datetime(df['fecha'], format='ISO8601', errors='coerce')).fillna(pd.Timestamp.min)
    return df.sort_values(['id_lector', 'fecha', 'orden'], kind='stable').reset_index(drop=True)


def filtrar(df, min_lector, min_libro):
    # Filtro recursivo (como en stepbystepdatascience.com/ease): se sacan los
    # libros con menos de min_libro interacciones y los lectores con menos de
    # min_lector, y se repite hasta que no cambie nada, porque cada paso puede
    # dejar a otros por debajo del mínimo.
    while True:
        n = len(df)
        por_libro = df.groupby('id_libro')['id_lector'].transform('size')
        df = df[por_libro >= min_libro]
        por_lector = df.groupby('id_lector')['id_libro'].transform('size')
        df = df[por_lector >= min_lector]
        if len(df) == n:
            return df


def particionar(df, n_test=N_TEST):
    # Partición cronológica por lector: las últimas n_test interacciones son test
    # y las anteriores train, así ningún modelo ve el futuro de lo que se evalúa.
    # Los lectores con n_test interacciones o menos quedan enteros en train (sirven
    # para entrenar, pero no se pueden evaluar).
    posicion_desde_el_final = df.groupby('id_lector').cumcount(ascending=False)
    total = df.groupby('id_lector')['id_libro'].transform('size')
    es_test = (posicion_desde_el_final < n_test) & (total > n_test)
    return df[~es_test], df[es_test]
