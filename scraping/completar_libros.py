"""
Completa los campos vacíos (titulo, autor, genero, editorial, anio_edicion, resumen)
de la tabla `libros` en data/data.db haciendo scraping de quelibroleo.com.

Script independiente del resto del proyecto.

Uso:
    python scraping/completar_libros.py                 # procesa todas las filas incompletas
    python scraping/completar_libros.py --limite 50     # solo 50 filas
    python scraping/completar_libros.py --pausa 1.5     # segundos entre requests
    python scraping/completar_libros.py --dry-run       # no escribe en la base

Es reanudable: solo toma filas que aún tengan algún campo vacío y guarda
cada fila al procesarla. Las URLs que fallan (404, etc.) se registran en la
tabla `libros_scraping_fallidos` y se saltean en ejecuciones siguientes
(usar --reintentar-fallidos para volver a intentarlas).
"""

import argparse
import re
import sqlite3
import time
from pathlib import Path

from scrapling.fetchers import Fetcher

BASE_URL = "https://quelibroleo.com/"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "data.db"
ANIO_DESCONOCIDO = 1900
CAMPOS = ["titulo", "autor", "genero", "editorial", "anio_edicion", "resumen"]

# Etiquetas de "Información del libro" -> columna en la tabla
ETIQUETAS = {
    "género": "genero",
    "editorial": "editorial",
    "año de edición": "anio_edicion",
}


def limpiar(texto):
    if texto is None:
        return None
    texto = re.sub(r"\s+", " ", str(texto)).strip()
    return texto or None


def primero(nodo, selector):
    """Primer elemento que coincide con el selector CSS, o None."""
    resultados = nodo.css(selector)
    return resultados[0] if resultados else None


def extraer_datos(page):
    """Devuelve un dict con los campos encontrados en la página."""
    datos = {}
    info = primero(page, "div.libro_info")
    if info is not None:
        # Título: texto propio del <h3>, sin el <small> del autor
        titulo = primero(info, "h3")
        if titulo is not None:
            # en mayúsculas, como el resto de los títulos de la base
            datos["titulo"] = (limpiar(titulo.text) or "").upper() or None

        # Autor: debajo del título (<h3>TÍTULO <small><a>AUTOR</a></small></h3>)
        autor = primero(info, "h3 small")
        if autor is not None:
            datos["autor"] = limpiar(autor.get_all_text())

        # Información del libro: <li><span>Etiqueta</span> valor</li>
        for li in info.css("#info ul.list li"):
            span = primero(li, "span")
            if span is None:
                continue
            etiqueta = limpiar(span.get_all_text()).lower()
            campo = ETIQUETAS.get(etiqueta)
            if not campo:
                continue
            valor = limpiar(li.get_all_text())
            if valor and valor.lower().startswith(etiqueta):
                valor = limpiar(valor[len(etiqueta):])
            datos[campo] = valor

    # Resumen: <h3 id="resumen"> seguido de <div class="content_libro">
    # (también se contempla un div con id="resumen")
    resumen = primero(page, "div#resumen")
    if resumen is None:
        resumen = primero(page, "#resumen + div.content_libro")
    if resumen is not None:
        datos["resumen"] = limpiar(resumen.get_all_text(separator=" "))

    if "anio_edicion" in datos:
        m = re.search(r"\d{4}", datos["anio_edicion"] or "")
        anio = int(m.group()) if m else None
        # quelibroleo usa 1900 como marcador de "año desconocido"
        datos["anio_edicion"] = anio if anio and anio != ANIO_DESCONOCIDO else None

    return {k: v for k, v in datos.items() if v not in (None, "")}


def preparar_tabla_fallidos(conn):
    conn.execute(
        """CREATE TABLE IF NOT EXISTS libros_scraping_fallidos (
               id_libro TEXT PRIMARY KEY,
               motivo   TEXT,
               fecha    TEXT DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    conn.commit()


def filas_incompletas(conn, limite, reintentar_fallidos):
    vacio = " OR ".join(f"{c} IS NULL OR TRIM({c}) = ''" for c in CAMPOS)
    sql = f"SELECT id_libro, {', '.join(CAMPOS)} FROM libros WHERE ({vacio})"
    if not reintentar_fallidos:
        sql += " AND id_libro NOT IN (SELECT id_libro FROM libros_scraping_fallidos)"
    if limite:
        sql += f" LIMIT {int(limite)}"
    return conn.execute(sql).fetchall()


def registrar_fallo(conn, id_libro, motivo):
    conn.execute(
        "INSERT OR REPLACE INTO libros_scraping_fallidos (id_libro, motivo) VALUES (?, ?)",
        (id_libro, motivo),
    )
    conn.commit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--limite", type=int, default=0, help="máximo de filas a procesar (0 = todas)")
    ap.add_argument("--pausa", type=float, default=1.0, help="segundos entre requests")
    ap.add_argument("--dry-run", action="store_true", help="no escribe en la base")
    ap.add_argument("--reintentar-fallidos", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    preparar_tabla_fallidos(conn)
    filas = filas_incompletas(conn, args.limite, args.reintentar_fallidos)
    total = len(filas)
    print(f"Filas con campos vacíos a procesar: {total}")

    actualizadas = fallidas = sin_cambios = 0
    for i, fila in enumerate(filas, 1):
        id_libro = fila[0]
        actuales = dict(zip(CAMPOS, fila[1:]))
        faltantes = [c for c in CAMPOS if actuales[c] is None or str(actuales[c]).strip() == ""]
        url = BASE_URL + id_libro

        try:
            page = Fetcher.get(url, stealthy_headers=True, timeout=30)
        except Exception as e:  # error de red
            print(f"[{i}/{total}] {id_libro}: ERROR {e}")
            fallidas += 1
            time.sleep(args.pausa * 3)
            continue

        if page.status != 200:
            print(f"[{i}/{total}] {id_libro}: HTTP {page.status}")
            fallidas += 1
            if not args.dry_run and page.status in (404, 410):
                registrar_fallo(conn, id_libro, f"HTTP {page.status}")
            time.sleep(args.pausa)
            continue

        datos = extraer_datos(page)
        nuevos = {c: datos[c] for c in faltantes if c in datos}

        if nuevos:
            if not args.dry_run:
                sets = ", ".join(f"{c} = ?" for c in nuevos)
                conn.execute(f"UPDATE libros SET {sets} WHERE id_libro = ?", (*nuevos.values(), id_libro))
                conn.commit()
                # si la página no tenía todos los campos faltantes, no volver a pedirla
                pendientes = [c for c in faltantes if c not in nuevos]
                if pendientes:
                    registrar_fallo(conn, id_libro, "incompleto en la página: " + ", ".join(pendientes))
            actualizadas += 1
            resumen_log = {k: (v[:40] + "…" if isinstance(v, str) and len(v) > 40 else v) for k, v in nuevos.items()}
            print(f"[{i}/{total}] {id_libro}: {resumen_log}")
        else:
            sin_cambios += 1
            print(f"[{i}/{total}] {id_libro}: sin datos nuevos (faltaban {faltantes})")
            if not args.dry_run:
                registrar_fallo(conn, id_libro, "sin datos en la página")

        time.sleep(args.pausa)

    conn.close()
    print(f"\nListo. Actualizadas: {actualizadas} | Sin datos: {sin_cambios} | Fallidas: {fallidas}")


if __name__ == "__main__":
    main()
