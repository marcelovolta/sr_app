import os
import shutil
import sqlite3
import time

import pandas as pd


# Datos para entrenar y evaluar los modelos: copia de la base, lectura de las
# interacciones en orden cronológico, universo de lectores, filtro de libros y
# partición train / validación / test.
RAIZ = os.path.dirname(os.path.abspath(__file__))
BASE_ORIGEN = os.path.join(RAIZ, 'data', 'data.db')
BASE_TRABAJO = os.path.join(RAIZ, 'data', 'trabajo', 'data.db')
MIN_LECTOR = 30    # universo: lectores con al menos 30 interacciones
N_TEST = 10        # las últimas 10 de cada lector son test
N_VALIDACION = 10  # las 10 anteriores, validación (el resto, train)
MIN_LECTOR_FINAL = 10  # universo del modelo final (el que recomienda): ver universo


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


def universo(df, min_lector=MIN_LECTOR):
    # Lectores del universo de los modelos: los que tienen al menos min_lector
    # interacciones en total. Todos tienen test, validación y al menos
    # min_lector - N_TEST - N_VALIDACION interacciones de train, así que todo
    # lector con el que se entrena también se evalúa, y viceversa. Es un valor
    # fijo (no un hiperparámetro): si cambiara entre trials, cada trial se
    # evaluaría con otros lectores y los NDCG no serían comparables.
    # El modelo final usa otro universo, más amplio (MIN_LECTOR_FINAL: tantas
    # interacciones como el train mínimo de la optimización), para recomendar
    # también a los lectores con esa historia; con los hiperparámetros elegidos
    # en el universo de evaluación.
    por_lector = df.groupby('id_lector')['id_libro'].transform('size')
    return df[por_lector >= min_lector]


def filtrar_libros(df, min_libro):
    # Saca los libros con menos de min_libro interacciones en df (un libro casi
    # sin lectores no se puede aprender bien). Se cuenta sobre los datos de
    # entrenamiento de cada etapa, para no usar lecturas futuras. No saca
    # lectores: el universo es fijo.
    por_libro = df.groupby('id_libro')['id_lector'].transform('size')
    return df[por_libro >= min_libro]


def particionar(df, n):
    # Partición cronológica por lector: las últimas n interacciones de cada
    # lector por un lado y las anteriores por el otro. Ningún modelo ve el
    # futuro de lo que se evalúa.
    posicion_desde_el_final = df.groupby('id_lector').cumcount(ascending=False)
    ultimas = posicion_desde_el_final < n
    return df[~ultimas], df[ultimas]


def particiones(df):
    # Universo y partición en tres tramos por lector:
    #   test: las últimas N_TEST; mide el resultado que se informa
    #   validación: las N_VALIDACION anteriores; con ellas Optuna elige los
    #     hiperparámetros, así el test no participa de ninguna elección
    #   train: el resto
    # Devuelve (universo, train, validacion, test).
    u = universo(df)
    resto, test = particionar(u, N_TEST)
    train, validacion = particionar(resto, N_VALIDACION)
    return u, train, validacion, test
