import sqlite3
import os
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from datetime import datetime
import threading

import ensamble
import modelos


# Base que usan los algoritmos. La app usa la base real; pipeline.py la cambia
# por su copia de trabajo (set_base) para generar las predicciones.
DATABASE_FILE = os.path.dirname(os.path.abspath(__file__)) + '/data/data.db'
PREDICCION_BASE_FILE = os.path.dirname(os.path.abspath(__file__)) + '/data/prediction_base.csv'
CASCADA_ARRANQUE_FRIO = 2  # hasta estas valoraciones, el lector recibe top_n


def set_base(database_file):
    # cambia la base que usan todos los algoritmos (pipeline.py usa su copia)
    global DATABASE_FILE
    DATABASE_FILE = database_file


def sql_select(query, params=(), database_file=None):
    # ejecuta una consulta y devuelve sus filas (accesibles por nombre de columna)
    conn = sqlite3.connect(database_file or DATABASE_FILE)
    conn.row_factory = sqlite3.Row # esto es para que devuelva regisros en el fetchall
    cur = conn.cursor()
    
    rec = cur.execute(query, params).fetchall()
    
    cur.close()
    conn.close()
    
    return rec


def sql_execute(query, params=()):
    # ejecuta una sentencia que modifica la base
    conn = sqlite3.connect(DATABASE_FILE)
    cur = conn.cursor()
    
    cur.execute(query, params)
    
    conn.commit()
    cur.close()
    conn.close()


def item_valorados(id_lector):
    # cantidad de libros que el lector valoró (para mostrar en la app)
    query = """
        SELECT COUNT(*) AS item_valorados 
        FROM libros 
        WHERE id_libro IN (SELECT id_libro FROM interacciones WHERE id_lector = ? AND rating > 0)
        AND titulo IS NOT NULL
    """
    
    
    return sql_select(query, (id_lector,))


def items_vistos(id_lector):
    # cantidad de libros que se le mostraron al lector (para mostrar en la app)
    query = """
        SELECT COUNT(*) AS items_vistos 
        FROM libros 
        WHERE id_libro IN (SELECT id_libro FROM vistas WHERE id_lector = ?)
        AND titulo IS NOT NULL
    """
    return sql_select(query, (id_lector,))

def insertar_vistos(id_lector, libros):
    # registra los libros mostrados al lector, para no volver a recomendárselos
    query="""
        INSERT INTO vistas (id_lector, id_libro, fecha) VALUES (?, ?, ?) ON CONFLICT DO NOTHING
    """
    
    for i in libros:
        sql_execute(query, (id_lector, i['id_libro'], datetime.now()))


def insertar_valorado(id_lector, id_libro, rating):
    # guarda la valoración que el lector hizo en la app
    # TODO: permitir que el lector actualice una valoración ya existente
    # (hoy ON CONFLICT DO NOTHING ignora la nueva puntuación en silencio)
    fecha = datetime.now()
    query="""
        INSERT INTO interacciones (id_lector, id_libro, fecha, rating) VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING
    """
    sql_execute(query, (id_lector, id_libro, fecha, rating))


def recomendador_random(id_lector, N=9):
    # N libros al azar que el lector no valoró ni vio
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

# ---------------------------------------------------------------------------
# Algoritmos basados en SQL
#
# Optimización: antes cada uno era una única consulta por lector que, además de
# lo propio del lector, recalculaba sobre toda la tabla de interacciones un
# ranking global que es igual para todos (promedio o cantidad de valoraciones
# por libro). Con 832 lectores eso eran 832 agregaciones de ~1 s, y en paralelo
# los hilos se trababan entre sí (top_n tardaba ~25 min).
# Ahora la parte global se calcula una vez y queda en memoria (_global), y por
# lector sólo se consulta lo suyo (sus géneros y autores, con el índice de
# interacciones) y se recorre el ranking en Python salteando lo que ya leyó o
# vio (_recomendador_por_ranking). Los resultados son los mismos que con las
# consultas originales; los empates se desempatan por id_libro.
# ---------------------------------------------------------------------------

_cache = {}
_cache_lock = threading.RLock()  # reentrante: un cálculo global puede pedir otro


def _global(nombre, calcular):
    # Resultado de calcular() guardado por base y por fecha de modificación del
    # archivo: si la base cambia (por ejemplo, la app guarda una valoración) se
    # recalcula en la próxima consulta, así nunca se usa un ranking desactualizado
    base = DATABASE_FILE
    clave = (nombre, base, os.stat(base).st_mtime_ns)
    with _cache_lock:
        if clave not in _cache:
            for vieja in [c for c in _cache if c[:2] == clave[:2]]:
                del _cache[vieja]
            _cache[clave] = calcular()
        return _cache[clave]


def _estadisticas_libros():
    # por libro: valoraciones (COUNT), promedio (AVG) y rating corregido
    # 0.8 * promedio + 0.2 * valoraciones, igual que en las consultas originales
    filas = sql_select("SELECT id_libro, COUNT(*) AS cant, AVG(rating) AS promedio FROM interacciones GROUP BY id_libro")
    return {r['id_libro']: (r['cant'], r['promedio'], (0.8 * r['promedio']) + (0.2 * r['cant'])) for r in filas}


def _libros():
    # por libro de la tabla libros: (género, autor, tiene título)
    return {r['id_libro']: (r['genero'], r['autor'], r['titulo'] is not None)
            for r in sql_select("SELECT id_libro, genero, autor, titulo FROM libros")}


def _ordenar(puntajes):
    # ids de mayor a menor puntaje, desempatando por id_libro
    return [b for b, _ in sorted(puntajes.items(), key=lambda x: (-x[1], x[0]))]


def recomendador_top_n(id_lector, N=9):
    # mejor promedio, con un mínimo de 5 valoraciones para que un único 10 no domine
    def calcular():
        # ranking global (igual para todos los lectores)
        libros, est = _global('libros', _libros), _global('estadisticas', _estadisticas_libros)
        return _ordenar({b: prom for b, (cant, prom, _) in est.items() if cant >= 5 and b in libros})
    return _recomendador_por_ranking(id_lector, N, _global('top_n', calcular))


def recomendador_mas_valorados(id_lector, N=9):
    # los libros con más valoraciones
    def calcular():
        # ranking global (igual para todos los lectores)
        libros, est = _global('libros', _libros), _global('estadisticas', _estadisticas_libros)
        return _ordenar({b: cant for b, (cant, _, _) in est.items() if b in libros})
    return _recomendador_por_ranking(id_lector, N, _global('mas_valorados', calcular))


def _preferencias(id_lector, campo, minimo):
    # rating corregido de cada género o autor que el lector valoró al menos `minimo` veces
    # (campo es 'genero' o 'autor', no viene del usuario)
    filas = sql_select(f"""
        SELECT li.{campo} AS valor, COUNT(*) AS cant, AVG(i.rating) AS promedio
        FROM interacciones AS i
        INNER JOIN libros AS li ON li.id_libro = i.id_libro
        WHERE i.id_lector = ? AND COALESCE(i.rating, 0) > 0 AND li.{campo} IS NOT NULL
        GROUP BY li.{campo}
        HAVING COUNT(*) >= ?
    """, (id_lector, minimo))
    return {r['valor']: (0.8 * r['promedio']) + (0.2 * r['cant']) for r in filas}


def recomendador_perfil_balanceado(id_lector, N=9):
    # combina los géneros y autores que el lector valoró (al menos 3 veces) con la
    # valoración general de cada libro popular (más de 5 valoraciones):
    # 0.4 * género + 0.4 * autor + 0.2 * rating corregido del libro
    generos = _preferencias(id_lector, 'genero', 3)   # hiperparámetro
    autores = _preferencias(id_lector, 'autor', 3)    # hiperparámetro

    def calcular():
        # los libros populares con su género, autor y rating corregido
        libros, est = _global('libros', _libros), _global('estadisticas', _estadisticas_libros)
        return [(b, libros[b][0], libros[b][1], rc) for b, (cant, _, rc) in est.items() if cant > 5 and b in libros]

    puntajes = {b: (0.4 * generos.get(g, 0)) + (0.4 * autores.get(a, 0)) + (0.2 * rc)
                for b, g, a, rc in _global('populares_perfil', calcular)}
    return _recomendador_por_ranking(id_lector, N, _ordenar(puntajes))


def recomendador_vision_tunel(id_lector, N=9):
    # más estricto que perfil_balanceado: el libro tiene que coincidir en género Y
    # en autor con los que valoró el lector (puntaje = autor + género); si no
    # alcanzan N, se completa con los populares (más de 5 valoraciones)
    generos = _preferencias(id_lector, 'genero', 1)
    autores = _preferencias(id_lector, 'autor', 1)

    def por_genero_y_autor():
        # índice (género, autor) -> libros, para encontrar rápido los afines
        indice = {}
        for b, (g, a, _) in _global('libros', _libros).items():
            indice.setdefault((g, a), []).append(b)
        return indice

    def populares():
        # los populares (más de 5 valoraciones) por rating corregido
        return _ordenar({b: rc for b, (cant, _, rc) in _global('estadisticas', _estadisticas_libros).items() if cant > 5})

    indice = _global('libros_por_genero_y_autor', por_genero_y_autor)
    afines = {b: autores[a] + generos[g] for a in autores for g in generos for b in indice.get((g, a), ())}
    ranking = _ordenar(afines) + [b for b in _global('populares_tunel', populares) if b not in afines]
    return _recomendador_por_ranking(id_lector, N, ranking)


def _recomendador_por_ranking(id_lector, N, ranking):
    # los N primeros libros de un ranking (ids, o pares id_libro, puntaje), sin
    # los que el lector ya valoró o vio ni los que no tienen título
    excluidos = {r['id_libro'] for r in sql_select("""
        SELECT id_libro FROM interacciones WHERE id_lector = ? AND COALESCE(rating, 0) > 0
        UNION
        SELECT id_libro FROM vistas WHERE id_lector = ?
    """, (id_lector, id_lector))}
    libros = _global('libros', _libros)

    ids = (x if isinstance(x, str) else x[0] for x in ranking)
    top = list(islice((b for b in ids if b not in excluidos and b in libros and libros[b][2]), N))
    if not top:
        return []
    filas = sql_select(f"SELECT * FROM libros WHERE id_libro IN ({', '.join('?' * len(top))})", top)
    posicion = {id_libro: i for i, id_libro in enumerate(top)}
    return sorted(filas, key=lambda r: posicion[r['id_libro']])


NOMBRES_MODELOS = (*modelos.NOMBRES, 'ensamble')


def _ranking(nombre, id_lector):
    # ranking de un modelo entrenado; el ensamble vive en su propio módulo
    return ensamble.ranking(id_lector) if nombre == 'ensamble' else modelos.ranking(nombre, id_lector)


def _conoce_lector(nombre, id_lector):
    # si el modelo tiene al lector (si no, las cascadas usan visión de túnel)
    return ensamble.conoce_lector(id_lector) if nombre == 'ensamble' else modelos.conoce_lector(nombre, id_lector)


def recomendador_modelo(nombre):
    # recomendador con uno de los modelos entrenados (modelos.py o ensamble.py)
    def recomendador(id_lector, N=9):
        # los N primeros del ranking del modelo, sin lo ya leído o visto
        return _recomendador_por_ranking(id_lector, N, _ranking(nombre, id_lector))
    return recomendador


def cant_interacciones(id_lector):
    # valoraciones del lector, para elegir la rama de las cascadas
    query = "SELECT COUNT(*) AS cant FROM interacciones WHERE id_lector = ? AND COALESCE(rating, 0) > 0"
    return sql_select(query, (id_lector,))[0]['cant']


def recomendador_cascada(nombre):
    # Elige el algoritmo según cuánto se conoce al lector: arranque en frío
    # (hasta CASCADA_ARRANQUE_FRIO valoraciones) con top_n; si el modelo no lo
    # conoce (quedó afuera del filtro de la base, cuyo mínimo eligió Optuna) con
    # visión de túnel, que puede recomendar cualquier libro; el resto con el modelo
    modelo = recomendador_modelo(nombre)
    def recomendador(id_lector, N=9):
        # la cascada para un lector
        if cant_interacciones(id_lector) <= CASCADA_ARRANQUE_FRIO:
            return recomendador_top_n(id_lector, N)
        if not _conoce_lector(nombre, id_lector):
            return recomendador_vision_tunel(id_lector, N)
        return modelo(id_lector, N)
    return recomendador


def resetear_recientes(id_lector, fecha_limite='2026-09-01 23:59:59'):
    # borra lo que el lector vio y valoró desde fecha_limite (las pruebas en la app)
    sql_execute("DELETE FROM vistas WHERE id_lector = ? AND fecha > ?", (id_lector, fecha_limite))
    sql_execute("DELETE FROM interacciones WHERE id_lector = ? AND fecha > ?", (id_lector, fecha_limite))


ALGORITMOS = {
    'random': recomendador_random,
    'top_n': recomendador_top_n,
    'mas_valorados': recomendador_mas_valorados,
    'perfil_balanceado': recomendador_perfil_balanceado,
    'vision_tunel': recomendador_vision_tunel,
    **{nombre: recomendador_modelo(nombre) for nombre in NOMBRES_MODELOS},
    **{f'cascada_{nombre}': recomendador_cascada(nombre) for nombre in NOMBRES_MODELOS},
    # los candidatos del ensamble ordenados por el promedio simple de los
    # puntajes estandarizados, en vez de LightGBM (en la optimización rindió igual)
    'promedio': lambda id_lector, N=9: _recomendador_por_ranking(id_lector, N, ensamble.ranking(id_lector, promedio=True)),
}
ALGORITMO_POR_DEFECTO = 'perfil_balanceado'
SQL = ('random', 'top_n', 'mas_valorados', 'perfil_balanceado', 'vision_tunel')


def recomendar(id_lector, algoritmo=ALGORITMO_POR_DEFECTO, N=9):
    # recomendación para la app: además de calcularla, la registra como vista
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
        # ids de los libros recomendados a un lector
        return [r['id_libro'] for r in recomendador(id_lector, cant_por_lector[id_lector])]

    # los algoritmos SQL se pueden correr en paralelo (cada consulta abre su
    # conexión); los modelos y las cascadas calculan con NumPy y con varios hilos
    # compiten por el GIL, así que van en uno solo
    if algoritmo not in SQL:
        workers = 1
    with ThreadPoolExecutor(workers) as pool:
        libros = dict(zip(cant_por_lector, pool.map(libros_de, cant_por_lector)))

    archivo = os.path.dirname(os.path.abspath(__file__)) + f'/data/prediction_{algoritmo}.csv'
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
