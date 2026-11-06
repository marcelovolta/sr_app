import sqlite3
import os
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from datetime import datetime

import modelos_surprise
import modelos_torres


DATABASE_FILE = os.path.dirname(__file__) + '/data/data.db'
# los modelos de Surprise y de torres se entrenan con esta versión reducida de la
# base (salvo dos_torres_v2_completa); todo lo demás (top_n, visión de túnel, los
# libros ya vistos) usa la completa
DATABASE_ENTRENAMIENTO_FILE = os.path.dirname(__file__) + '/data/backup/data_filtrada_min35.db'
PREDICCION_BASE_FILE = os.path.dirname(__file__) + '/data/prediction_base.csv'
CASCADA_ARRANQUE_FRIO = 2
CASCADA_POCAS = 10
CASCADA_V2_MAX_VISION_TUNEL = 35  # la base reducida sólo tiene lectores con más de 35 interacciones

def sql_select(query, params=(), database_file=DATABASE_FILE):
    conn = sqlite3.connect(database_file)
    conn.row_factory = sqlite3.Row # esto es para que devuelva regisros en el fetchall
    cur = conn.cursor()
    
    rec = cur.execute(query, params).fetchall()
    
    cur.close()
    conn.close()
    
    return rec


def sql_execute(query, params=()):
    conn = sqlite3.connect(DATABASE_FILE)
    cur = conn.cursor()
    
    cur.execute(query, params)
    
    conn.commit()
    cur.close()
    conn.close()


def item_valorados(id_lector):
    query = """
        SELECT COUNT(*) AS item_valorados 
        FROM libros 
        WHERE id_libro IN (SELECT id_libro FROM interacciones WHERE id_lector = ? AND rating > 0)
        AND titulo IS NOT NULL
    """
    
    
    return sql_select(query, (id_lector,))


def items_vistos(id_lector):
    query = """
        SELECT COUNT(*) AS items_vistos 
        FROM libros 
        WHERE id_libro IN (SELECT id_libro FROM vistas WHERE id_lector = ?)
        AND titulo IS NOT NULL
    """
    return sql_select(query, (id_lector,))

def items_desconocidos(id_lector):
    query = """
        SELECT * AS items_desconocidos 
        FROM libros 
        WHERE id_libro NOT IN (SELECT id_libro FROM interacciones WHERE id_lector = ?)
        AND id_libro NOT IN (SELECT id_libro FROM vistas WHERE id_lector = ?)
        AND titulo IS NOT NULL
    """
    return sql_select(query, (id_lector, id_lector))


def insertar_vistos(id_lector, libros):
    query="""
        INSERT INTO vistas (id_lector, id_libro, fecha) VALUES (?, ?, ?) ON CONFLICT DO NOTHING
    """
    
    for i in libros:
        sql_execute(query, (id_lector, i['id_libro'], datetime.now()))


def insertar_valorado(id_lector, id_libro, rating):
    # TODO: permitir que el lector actualice una valoración ya existente
    # (hoy ON CONFLICT DO NOTHING ignora la nueva puntuación en silencio)
    fecha = datetime.now()
    query="""
        INSERT INTO interacciones (id_lector, id_libro, fecha, rating) VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING
    """
    sql_execute(query, (id_lector, id_libro, fecha, rating))


def recomendador_random(id_lector, N=9):
    query = f"""
        SELECT * 
        FROM libros 
        WHERE id_libro NOT IN (SELECT id_libro FROM interacciones WHERE id_lector = ? AND COALESCE(rating, 0) > 0)
        AND id_libro NOT IN (SELECT id_libro FROM vistas WHERE id_lector = ?)
        AND titulo IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {N}
    """
    return sql_select(query, (id_lector, id_lector))  

def recomendador_top_n(id_lector, N=9):
    # se exige un mínimo de 5 valoraciones para que un único 10 no domine el ranking
    query = f"""
        WITH t1 AS (
        SELECT li.id_libro, AVG(i.rating) AS promedio_rating 
        FROM libros AS li
        INNER JOIN interacciones AS i ON li.id_libro = i.id_libro
        WHERE li.id_libro NOT IN (SELECT id_libro FROM interacciones WHERE id_lector = ? AND COALESCE(rating, 0) > 0)
        AND li.id_libro NOT IN (SELECT id_libro FROM vistas WHERE id_lector = ?)
        AND li.titulo IS NOT NULL
        GROUP BY li.id_libro
        HAVING COUNT(*) >= 5
        ORDER BY promedio_rating DESC
        LIMIT {N}
    )
    SELECT libros.* FROM libros
    INNER JOIN t1 ON t1.id_libro = libros.id_libro
    ORDER BY t1.promedio_rating DESC
    """
    return sql_select(query, (id_lector, id_lector))

def recomendador_mas_valorados(id_lector, N=9):
    query = f"""
        WITH t1 AS (
        SELECT li.id_libro, COUNT(*) AS valoraciones 
        FROM libros AS li
        INNER JOIN interacciones AS i ON li.id_libro = i.id_libro
        WHERE li.id_libro NOT IN (SELECT id_libro FROM interacciones WHERE id_lector = ? AND COALESCE(rating, 0) > 0)
        AND li.id_libro NOT IN (SELECT id_libro FROM vistas WHERE id_lector = ?)
        AND li.titulo IS NOT NULL
        GROUP BY li.id_libro
        ORDER BY valoraciones DESC
        LIMIT {N}
    )
    SELECT libros.* FROM libros
    INNER JOIN t1 ON t1.id_libro = libros.id_libro
    ORDER BY t1.valoraciones DESC
    """
    return sql_select(query, (id_lector, id_lector))

def recomendador_perfil_balanceado(id_lector, N=9):
    # combina los géneros y autores que el lector valoró (al menos 3 veces)
    # con la valoración general de cada libro popular (más de 5 valoraciones)
    query = f"""
    WITH generos_pref AS (
        SELECT
            li.genero,
            AVG(i.rating) AS promedio_rating,
            COUNT(*) AS cant_valoraciones,
            (0.8 * AVG(i.rating)) + (0.2 * COUNT(*)) AS rating_corr_genero
        FROM interacciones AS i
        INNER JOIN libros AS li ON li.id_libro = i.id_libro
        WHERE i.id_lector = ?
        AND COALESCE(i.rating, 0) > 0
        AND li.genero IS NOT NULL
        GROUP BY li.genero
        HAVING COUNT(*) > 2  -- hiperparámetro
    ),

    autores_pref AS (
        SELECT
            li.autor,
            AVG(i.rating) AS promedio_rating,
            COUNT(*) AS cant_valoraciones,
            (0.8 * AVG(i.rating)) + (0.2 * COUNT(*)) AS rating_corr_autor
        FROM interacciones AS i
        INNER JOIN libros AS li ON li.id_libro = i.id_libro
        WHERE i.id_lector = ?
        AND COALESCE(i.rating, 0) > 0
        AND li.autor IS NOT NULL
        GROUP BY li.autor
        HAVING COUNT(*) > 2  -- hiperparámetro
    ),

    libros_populares AS (
        SELECT
            id_libro,
            AVG(rating) AS promedio_rating,
            COUNT(*) AS cant_valoraciones,
            (0.8 * AVG(rating)) + (0.2 * COUNT(*)) AS rating_corr
        FROM interacciones
        GROUP BY id_libro
        HAVING COUNT(*) > 5
    ),

    seleccion_final AS (
        SELECT
            lp.id_libro,
            (0.4 * COALESCE(gp.rating_corr_genero, 0))
              + (0.4 * COALESCE(ap.rating_corr_autor, 0))
              + (0.2 * lp.rating_corr) AS rating_global
        FROM libros_populares AS lp
        INNER JOIN libros AS li ON li.id_libro = lp.id_libro
        LEFT JOIN generos_pref AS gp ON gp.genero = li.genero
        LEFT JOIN autores_pref AS ap ON ap.autor = li.autor
        WHERE lp.id_libro NOT IN (SELECT id_libro FROM interacciones WHERE id_lector = ? AND COALESCE(rating, 0) > 0)
        AND lp.id_libro NOT IN (SELECT id_libro FROM vistas WHERE id_lector = ?)
        AND li.titulo IS NOT NULL
        ORDER BY rating_global DESC, lp.id_libro
        LIMIT {N}
    )
    SELECT libros.* FROM libros
    INNER JOIN seleccion_final AS sf ON sf.id_libro = libros.id_libro
    ORDER BY sf.rating_global DESC, sf.id_libro
    """
    return sql_select(query, (id_lector,) * 4)

def recomendador_vision_tunel(id_lector, N=9):
    # más estricto que perfil_balanceado: el libro tiene que coincidir en género
    # Y en autor con los favoritos del lector; si no alcanzan N, se completa
    # con libros populares (orden = 1)
    query = f"""
    WITH generos_fav AS (
        SELECT
            li.genero,
            COUNT(*) AS cant_genero,
            AVG(i.rating) AS promedio_rating,
            (0.8 * AVG(i.rating)) + (0.2 * COUNT(*)) AS rating_corr_genero
        FROM interacciones AS i
        INNER JOIN libros AS li ON li.id_libro = i.id_libro
        WHERE i.id_lector = ?
        AND COALESCE(i.rating, 0) > 0
        AND li.genero IS NOT NULL
        GROUP BY li.genero
    ),

    autores_fav AS (
        SELECT
            li.autor,
            COUNT(*) AS cant_autor,
            AVG(i.rating) AS promedio_rating,
            (0.8 * AVG(i.rating)) + (0.2 * COUNT(*)) AS rating_corr_autor
        FROM interacciones AS i
        INNER JOIN libros AS li ON li.id_libro = i.id_libro
        WHERE i.id_lector = ?
        AND COALESCE(i.rating, 0) > 0
        AND li.autor IS NOT NULL
        GROUP BY li.autor
    ),

    vision_tunel AS (
        SELECT
            li.id_libro,
            af.rating_corr_autor + gf.rating_corr_genero AS rating_final,
            0 AS orden
        FROM libros AS li
        INNER JOIN generos_fav AS gf ON gf.genero = li.genero
        INNER JOIN autores_fav AS af ON af.autor = li.autor
        WHERE li.id_libro NOT IN (SELECT id_libro FROM interacciones WHERE id_lector = ? AND COALESCE(rating, 0) > 0)
        AND li.id_libro NOT IN (SELECT id_libro FROM vistas WHERE id_lector = ?)
        AND li.titulo IS NOT NULL
    ),

    -- se excluyen explícitamente los libros de vision_tunel para que un libro
    -- que califica en ambos niveles no aparezca dos veces
    libros_populares AS (
        SELECT
            id_libro,
            (0.8 * AVG(rating)) + (0.2 * COUNT(*)) AS rating_corr,
            1 AS orden
        FROM interacciones
        WHERE id_libro NOT IN (SELECT id_libro FROM vision_tunel)
        AND id_libro NOT IN (SELECT id_libro FROM interacciones WHERE id_lector = ? AND COALESCE(rating, 0) > 0)
        AND id_libro NOT IN (SELECT id_libro FROM vistas WHERE id_lector = ?)
        GROUP BY id_libro
        HAVING COUNT(*) > 5
    ),

    preseleccion AS (
        SELECT id_libro, rating_final, orden FROM vision_tunel
        UNION ALL
        SELECT id_libro, rating_corr AS rating_final, orden FROM libros_populares
    )
    SELECT libros.* FROM preseleccion AS ps
    INNER JOIN libros ON libros.id_libro = ps.id_libro
    WHERE libros.titulo IS NOT NULL
    ORDER BY ps.orden ASC, ps.rating_final DESC, ps.id_libro
    LIMIT {N}
    """
    return sql_select(query, (id_lector,) * 6)

def _recomendador_por_ranking(id_lector, N, ranking):
    # los N primeros libros del ranking de un modelo (de mayor a menor rating
    # estimado), excluyendo los que el lector ya valoró o vio, y los que no tienen título.
    # ranking(es_candidato) devuelve los pares (id_libro, rating estimado)
    excluidos = {r['id_libro'] for r in sql_select("""
        SELECT id_libro FROM interacciones WHERE id_lector = ? AND COALESCE(rating, 0) > 0
        UNION
        SELECT id_libro FROM vistas WHERE id_lector = ?
    """, (id_lector, id_lector))}
    con_titulo = {r['id_libro'] for r in sql_select("SELECT id_libro FROM libros WHERE titulo IS NOT NULL")}
    def es_candidato(id_libro):
        return id_libro not in excluidos and id_libro in con_titulo

    top = list(islice((id_libro for id_libro, _ in ranking(es_candidato) if es_candidato(id_libro)), N))
    if not top:
        return []

    libros = sql_select(f"SELECT * FROM libros WHERE id_libro IN ({', '.join('?' * len(top))})", top)
    posicion = {id_libro: i for i, id_libro in enumerate(top)}
    return sorted(libros, key=lambda r: posicion[r['id_libro']])

def recomendador_surprise(nombre, id_lector, N=9):
    return _recomendador_por_ranking(id_lector, N, lambda es_candidato: modelos_surprise.ranking(nombre, DATABASE_ENTRENAMIENTO_FILE, id_lector))

def recomendador_baseline(id_lector, N=9):
    return recomendador_surprise('baseline', id_lector, N)

def recomendador_knn(id_lector, N=9):
    return recomendador_surprise('knn', id_lector, N)

def recomendador_svd(id_lector, N=9):
    return recomendador_surprise('svd', id_lector, N)

def recomendador_torres(nombre, id_lector, N=9, database_file=DATABASE_ENTRENAMIENTO_FILE):
    return _recomendador_por_ranking(id_lector, N, lambda es_candidato: modelos_torres.ranking(nombre, database_file, id_lector, es_candidato))

def recomendador_dos_torres(id_lector, N=9):
    return recomendador_torres('dos_torres', id_lector, N)

def recomendador_dos_torres_v2(id_lector, N=9):
    return recomendador_torres('dos_torres_v2', id_lector, N)

def recomendador_dos_torres_v2_completa(id_lector, N=9):
    return recomendador_torres('dos_torres_v2_completa', id_lector, N, DATABASE_FILE)

def recomendador_gran_torre(id_lector, N=9):
    return recomendador_torres('gran_torre', id_lector, N)

def recomendador_gran_torre_v2(id_lector, N=9):
    return recomendador_torres('gran_torre_v2', id_lector, N)

def cant_interacciones(id_lector, database_file=DATABASE_FILE):
    query = "SELECT COUNT(*) AS cant FROM interacciones WHERE id_lector = ? AND COALESCE(rating, 0) > 0"
    return sql_select(query, (id_lector,), database_file)[0]['cant']

def recomendador_cascada_v1(id_lector, N=9):
    # no usa la base reducida: elige el algoritmo según las interacciones del lector
    # en la base completa. Arranque en frío (hasta CASCADA_ARRANQUE_FRIO) con los mejor
    # valorados, pocas (hasta CASCADA_POCAS) con visión de túnel, y el resto con dos
    # torres. Los lectores con muchas interacciones que quedaron afuera del modelo
    # (casi todas son de libros con pocas valoraciones) también van a visión de túnel
    cant = cant_interacciones(id_lector)
    if cant <= CASCADA_ARRANQUE_FRIO:
        return recomendador_top_n(id_lector, N)
    if cant <= CASCADA_POCAS or not modelos_torres.conoce_lector('dos_torres_v2_completa', DATABASE_FILE, id_lector):
        return recomendador_vision_tunel(id_lector, N)
    return recomendador_dos_torres_v2_completa(id_lector, N)

def _cascada_base_reducida(recomendador_modelo, conoce_lector, id_lector, N):
    # elige el algoritmo según las interacciones del lector en la base completa:
    # arranque en frío (hasta CASCADA_ARRANQUE_FRIO) con los mejor valorados, hasta
    # CASCADA_V2_MAX_VISION_TUNEL con visión de túnel, y el resto con recomendador_modelo,
    # entrenado con la base reducida. Los que no entraron en esa base no están en el
    # modelo (muchas de sus interacciones son de libros poco valorados), y también
    # van a visión de túnel
    cant = cant_interacciones(id_lector)
    if cant <= CASCADA_ARRANQUE_FRIO:
        return recomendador_top_n(id_lector, N)
    if cant <= CASCADA_V2_MAX_VISION_TUNEL or not conoce_lector(id_lector):
        return recomendador_vision_tunel(id_lector, N)
    return recomendador_modelo(id_lector, N)

def recomendador_cascada_v2(id_lector, N=9):
    conoce = lambda id_lector: modelos_torres.conoce_lector('dos_torres_v2', DATABASE_ENTRENAMIENTO_FILE, id_lector)
    return _cascada_base_reducida(recomendador_dos_torres_v2, conoce, id_lector, N)

def recomendador_cascada_v3(id_lector, N=9):
    conoce = lambda id_lector: modelos_torres.conoce_lector('gran_torre_v2', DATABASE_ENTRENAMIENTO_FILE, id_lector)
    return _cascada_base_reducida(recomendador_gran_torre_v2, conoce, id_lector, N)

def recomendador_cascada_v4(id_lector, N=9):
    conoce = lambda id_lector: modelos_surprise.conoce_lector('svd', DATABASE_ENTRENAMIENTO_FILE, id_lector)
    return _cascada_base_reducida(recomendador_svd, conoce, id_lector, N)

def resetear_recientes(id_lector, fecha_limite='2026-09-01 23:59:59'):
    sql_execute("DELETE FROM vistas WHERE id_lector = ? AND fecha > ?", (id_lector, fecha_limite))
    sql_execute("DELETE FROM interacciones WHERE id_lector = ? AND fecha > ?", (id_lector, fecha_limite))


ALGORITMOS = {
    'random': recomendador_random,
    'top_n': recomendador_top_n,
    'mas_valorados': recomendador_mas_valorados,
    'perfil_balanceado': recomendador_perfil_balanceado,
    'vision_tunel': recomendador_vision_tunel,
    'baseline': recomendador_baseline,
    'knn': recomendador_knn,
    'svd': recomendador_svd,
    'dos_torres': recomendador_dos_torres,
    'dos_torres_v2': recomendador_dos_torres_v2,
    'gran_torre': recomendador_gran_torre,
    'gran_torre_v2': recomendador_gran_torre_v2,
    'cascada_v1': recomendador_cascada_v1,
    'cascada_v2': recomendador_cascada_v2,
    'cascada_v3': recomendador_cascada_v3,
    'cascada_v4': recomendador_cascada_v4,
}
ALGORITMO_POR_DEFECTO = 'perfil_balanceado'


def recomendar(id_lector, algoritmo=ALGORITMO_POR_DEFECTO, N=9):
    recomendador = ALGORITMOS.get(algoritmo, ALGORITMOS[ALGORITMO_POR_DEFECTO])
    rec = recomendador(id_lector, N)

    insertar_vistos(id_lector, rec)
    return rec


def generar_predicciones(algoritmo, workers=8):
    # completa prediction_base.csv con los libros recomendados a cada lector,
    # de mayor a menor puntaje. A diferencia de recomendar(), no los registra como vistos
    recomendador = ALGORITMOS[algoritmo]

    with open(PREDICCION_BASE_FILE, newline='', encoding='utf-8') as f:
        filas = [fila['id_lector'] for fila in csv.DictReader(f)]

    # cada lector pide tantos libros como veces aparece en el archivo base (20)
    cant_por_lector = Counter(filas)
    def libros_de(id_lector):
        return [r['id_libro'] for r in recomendador(id_lector, cant_por_lector[id_lector])]

    # cada consulta abre su propia conexión, así que se pueden correr en paralelo.
    # Los modelos de Surprise y de torres no: calculan en Python y no en SQLite, y
    # con varios hilos compitiendo por el GIL tardan bastante más que con uno solo.
    # Las cascadas usan dos torres con la mayoría de los lectores
    if algoritmo in modelos_surprise.CONSTRUCTORES or algoritmo in modelos_torres.NOMBRES or algoritmo.startswith('cascada'):
        workers = 1
    with ThreadPoolExecutor(workers) as pool:
        libros = dict(zip(cant_por_lector, pool.map(libros_de, cant_por_lector)))

    archivo = os.path.dirname(__file__) + f'/data/prediction_{algoritmo}.csv'
    with open(archivo, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['id_lector', 'id_libro'])
        posicion = Counter()
        for id_lector in filas:
            # si el algoritmo devolvió menos libros, el resto queda vacío
            recomendados = libros[id_lector]
            i = posicion[id_lector]
            writer.writerow([id_lector, recomendados[i] if i < len(recomendados) else ''])
            posicion[id_lector] += 1

    return archivo, len(cant_por_lector)
