import argparse
import csv
import datetime
import glob
import os
import warnings

import datos
import ensamble
import modelos
import optimizacion
import recomendador


# Proceso completo, para volver a correrlo cuando cambie la base:
#   1. copia data/data.db a data/trabajo/data.db (sin bloquear la original)
#   2. partición cronológica train/test (últimas 20 interacciones por lector)
#   3. por cada modelo: Optuna sobre train/test, entrenamiento final con todos
#      los datos, modelos/<modelo>.pkl y reportes/<modelo>.md (las redes,
#      además, con su curva de entrenamiento: reportes/curva_<modelo>.png)
#   4. ensamble: LightGBM sobre las predicciones de SVD, EASE y las torres
#      (ensamble.py), con modelos/ensamble.pkl y reportes/ensamble.md
#   5. borra los data/prediction_*.csv viejos y genera los de todos los
#      algoritmos (SQL, modelos y cascadas) con la copia de trabajo
#   6. reportes/resumen.md
#
#   python pipeline.py                         todo
#   python pipeline.py --modelos ease svd      sólo esos modelos (y sus reportes);
#                                              'ensamble' necesita los modelos base ya entrenados
#   python pipeline.py --trials 5              menos trials (prueba rápida)
#   python pipeline.py --sin-copia             usa la copia de trabajo existente
#   python pipeline.py --continuar             retoma estudios de Optuna cortados
#   python pipeline.py --sin-predicciones      no genera los CSV
#   python pipeline.py --solo-predicciones     sólo pasos 5 y 6, con los .pkl actuales


EPOCAS_CURVA = {'dos_torres': 30, 'gran_torre': 20}  # mínimo de épocas de las curvas


def optimizar_modelos(nombres, trials=None, continuar=False):
    # paso 2 a 4: partición, y por modelo Optuna, entrenamiento final, curva y reporte
    df = datos.leer_interacciones()
    train, test = datos.particionar(df)
    print(f'{len(df)} interacciones: train {len(train)}, test {len(test)} '
          f"({test['id_lector'].nunique()} lectores evaluados)", flush=True)
    for nombre in nombres:
        print(f'== {nombre}', flush=True)
        if nombre == 'ensamble':
            m = ensamble.optimizar_y_entrenar(train, test, datos.BASE_TRABAJO, trials or ensamble.TRIALS)
            print(f"   mejor ndcg {m['optimizacion']['ndcg']:.4f} con {m['params']}", flush=True)
            continue
        estudio, ndcg_popularidad = optimizacion.optimizar(nombre, train, test, datos.BASE_TRABAJO, trials, continuar)
        m = optimizacion.entrenar_final(nombre, estudio, df, datos.BASE_TRABAJO, ndcg_popularidad)
        puntos = None
        if nombre in EPOCAS_CURVA:
            # el doble de las épocas elegidas, con un mínimo, para ver qué pasa después
            epocas = max(EPOCAS_CURVA[nombre], 2 * m['params']['epochs'])
            puntos = optimizacion.curva(nombre, m['params'], train, test, datos.BASE_TRABAJO, epocas)
        optimizacion.reporte_modelo(nombre, estudio, m, puntos)
        print(f"   mejor ndcg {estudio.best_value:.4f} con {m['params']}", flush=True)


def generar_predicciones():
    # los CSV se generan con la copia de trabajo, la misma con la que se entrenó
    recomendador.set_base(datos.BASE_TRABAJO)
    for viejo in glob.glob(os.path.join(datos.RAIZ, 'data', 'prediction_*.csv')):
        if not viejo.endswith('prediction_base.csv'):
            os.remove(viejo)
    for algoritmo in recomendador.ALGORITMOS:
        archivo, n = recomendador.generar_predicciones(algoritmo)
        print(f'{os.path.basename(archivo)}: {n} lectores', flush=True)


def reporte_resumen():
    # reportes/resumen.md: tabla comparativa de los modelos y reparto de las cascadas
    recomendador.set_base(datos.BASE_TRABAJO)
    df = datos.leer_interacciones(datos.BASE_TRABAJO)
    train, test = datos.particionar(df)
    with open(recomendador.PREDICCION_BASE_FILE, newline='', encoding='utf-8') as f:
        lectores_kaggle = sorted({fila['id_lector'] for fila in csv.DictReader(f)})
    cant = {l: recomendador.cant_interacciones(l) for l in lectores_kaggle}

    filas, reparto = [], []
    for nombre in modelos.NOMBRES:
        try:
            m = modelos.modelo(nombre)
        except FileNotFoundError:
            continue
        o, p = m['optimizacion'], m['params']
        filas.append(f"| [{nombre}]({nombre}.md) | **{o['ndcg']:.4f}** | {o['ndcg'] / o['ndcg_popularidad']:.2f}× "
                     f"| {o['evaluacion']['lectores_cubiertos']:.0%} | {p['min_lector']} | {p['min_libro']} "
                     f"| {len(m['indice_lector'])} | {len(m['ids_libros'])} | {o['trials']} |")
        frio = sum(c <= recomendador.CASCADA_ARRANQUE_FRIO for c in cant.values())
        modelo = sum(c > recomendador.CASCADA_ARRANQUE_FRIO and l in m['indice_lector'] for l, c in cant.items())
        reparto.append(f'| cascada_{nombre} | {frio} | {len(cant) - frio - modelo} | {modelo} |')
    ndcg_pop = o['ndcg_popularidad'] if filas else float('nan')
    try:
        o = modelos.modelo('ensamble')['optimizacion']
        filas.append(f"| [ensamble](ensamble.md) | **{o['ndcg']:.4f}** | {o['ndcg'] / o['ndcg_popularidad']:.2f}× "
                     f"| {o['lectores_cubiertos']:.0%} | — | — | — | — | {o['trials']} |")
        modelo = sum(c > recomendador.CASCADA_ARRANQUE_FRIO and ensamble.conoce_lector(l) for l, c in cant.items())
        reparto.append(f'| cascada_ensamble | {frio} | {len(cant) - frio - modelo} | {modelo} |')
    except FileNotFoundError:
        pass

    lineas = [
        '# Optimización de los recomendadores', '',
        f"Generado el {datetime.datetime.now():%Y-%m-%d %H:%M} con una copia de `data/data.db`.", '',
        '## Datos', '',
        f"- Interacciones (rating > 0): {len(df)}, de {df['id_lector'].nunique()} lectores y {df['id_libro'].nunique()} libros.",
        f"- Partición cronológica por lector: las últimas {datos.N_TEST} interacciones son test y las anteriores train.",
        f"- Train: {len(train)} interacciones. Test: {len(test)} ({test['id_lector'].nunique()} lectores con más de {datos.N_TEST} interacciones).",
        '', '## Método', '',
        '- Cada modelo se ajusta con Optuna (TPE) maximizando NDCG@20 en el test.',
        '- El filtro de la base es un hiperparámetro más: mínimo de interacciones por lector (≥ 20) y por libro, '
        'aplicado de forma recursiva y sólo sobre train.',
        '- Se evalúa siempre a todos los lectores del test; los que el filtro deja afuera reciben los más leídos, '
        'así un filtro exigente no se ve favorecido por evaluar sólo a lectores fáciles.',
        '- Relevante = cualquier libro leído en el test, sin importar el rating.',
        '- Con los mejores hiperparámetros se entrena el modelo final con todos los datos.',
        '- El NDCG se mide en el mismo test con el que se eligieron los hiperparámetros: es algo optimista.',
        '', '## Resultados', '',
        f'Referencia: recomendar a todos los más leídos da NDCG@20 = {ndcg_pop:.4f}.', '',
        '| Modelo | NDCG@20 | vs popularidad | Lectores del test cubiertos | min_lector | min_libro | Lectores (final) | Libros (final) | Trials |',
        '|---|---|---|---|---|---|---|---|---|',
        *filas,
        '', '## Cascadas', '',
        f'Hasta {recomendador.CASCADA_ARRANQUE_FRIO} valoraciones: top_n. Si el modelo no conoce al lector '
        '(quedó afuera del filtro): visión de túnel. El resto: el modelo.', '',
        f'Reparto de los {len(cant)} lectores de `prediction_base.csv`:', '',
        '| Cascada | top_n | visión de túnel | modelo |', '|---|---|---|---|',
        *reparto,
    ]
    os.makedirs(optimizacion.CARPETA_REPORTES, exist_ok=True)
    with open(os.path.join(optimizacion.CARPETA_REPORTES, 'resumen.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lineas) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimiza y entrena los modelos y genera las predicciones')
    nombres = [*modelos.NOMBRES, 'ensamble']  # el ensamble va último: usa los demás
    parser.add_argument('--modelos', nargs='+', choices=nombres, default=nombres)
    parser.add_argument('--trials', type=int)
    parser.add_argument('--sin-copia', action='store_true')
    parser.add_argument('--continuar', action='store_true')
    parser.add_argument('--sin-predicciones', action='store_true')
    parser.add_argument('--solo-predicciones', action='store_true')
    args = parser.parse_args()
    warnings.filterwarnings('ignore')  # avisos de JAX y Surprise sin consecuencias

    if not args.solo_predicciones:
        if not args.sin_copia:
            print('copia:', datos.copiar_base(), flush=True)
        optimizar_modelos(args.modelos, args.trials, args.continuar)
    if not args.sin_predicciones:
        generar_predicciones()
    reporte_resumen()
