import sqlite3
import os
from datetime import datetime


DATABASE_FILE = os.path.dirname(__file__) + '/data/data.db'

def sql_select(query, params=()):
    conn = sqlite3.connect(DATABASE_FILE)
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
    query = f"""
        WITH t1 AS (
        SELECT li.id_libro, AVG(i.rating) AS promedio_rating 
        FROM libros AS li
        INNER JOIN interacciones AS i ON li.id_libro = i.id_libro
        WHERE li.id_libro NOT IN (SELECT id_libro FROM interacciones WHERE id_lector = ? AND COALESCE(rating, 0) > 0)
        AND li.id_libro NOT IN (SELECT id_libro FROM vistas WHERE id_lector = ?)
        AND li.titulo IS NOT NULL
        GROUP BY li.id_libro
        ORDER BY promedio_rating DESC
        LIMIT {N}
    )
    SELECT * FROM libros WHERE id_libro IN (SELECT id_libro FROM t1)
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
    SELECT * FROM libros WHERE id_libro IN (SELECT id_libro FROM t1)
    """
    return sql_select(query, (id_lector, id_lector))

def resetear_recientes(id_lector, fecha_limite='2026-09-01 23:59:59'):
    sql_execute("DELETE FROM vistas WHERE id_lector = ? AND fecha > ?", (id_lector, fecha_limite))
    sql_execute("DELETE FROM interacciones WHERE id_lector = ? AND fecha > ?", (id_lector, fecha_limite))


def recomendar(id_lector, N=9):
    # return recomendador_random(id_lector, N)
    # return recomendador_top_n(id_lector, N)
    rec = recomendador_mas_valorados(id_lector, N)

    insertar_vistos(id_lector, rec)
    return rec
