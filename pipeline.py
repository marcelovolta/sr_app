import argparse
import csv
import datetime
import glob
import os
import warnings

import pandas as pd

import datos
import ensamble
import modelos
import optimizacion
import recomendador


# Proceso completo, para volver a correrlo cuando cambie la base:
#   1. copia data/data.db a data/trabajo/data.db (sin bloquear la original)
#   2. universo (lectores con al menos 20 interacciones) y partición
#      cronológica por lector: últimas 10 test, 5 anteriores validación, resto train
#   3. por cada modelo: Optuna (train → validación), medición en test
#      (train + validación → test), entrenamiento final con los lectores con al
#      menos 10 interacciones (con todas sus interacciones),
#      modelos/<modelo>.pkl y reportes/<modelo>.md (las redes, además, con su
#      curva de entrenamiento: reportes/curva_<modelo>.png)
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
    # pasos 2 a 4: partición, y por modelo Optuna, medición en test,
    # entrenamiento final, curva y reporte
    df = datos.leer_interacciones()
    universo, train, validacion, test = datos.particiones(df)
    resto = pd.concat([train, validacion])
    universo_final = datos.universo(df, datos.MIN_LECTOR_FINAL)
    print(f"{len(df)} interacciones; universo: {len(universo)} de {universo['id_lector'].nunique()} lectores "
          f'(train {len(train)}, validación {len(validacion)}, test {len(test)})', flush=True)
    for nombre in nombres:
        print(f'== {nombre}', flush=True)
        if nombre == 'ensamble':
            m = ensamble.optimizar_y_entrenar(train, validacion, resto, test, datos.BASE_TRABAJO, trials or ensamble.TRIALS)
            print(f"   ndcg test {m['optimizacion']['ndcg']:.4f} con {m['params']}", flush=True)
            continue
        estudio = optimizacion.optimizar(nombre, train, validacion, datos.BASE_TRABAJO, trials, continuar)
        params = optimizacion.mejores_params(nombre, estudio)
        prueba = optimizacion.medir_test(nombre, params, resto, test, datos.BASE_TRABAJO)
        m = optimizacion.entrenar_final(nombre, estudio, universo_final, datos.BASE_TRABAJO, prueba)
        puntos = None
        if nombre in EPOCAS_CURVA:
            # el doble de las épocas elegidas, con un mínimo, para ver qué pasa después
            epocas = max(EPOCAS_CURVA[nombre], 2 * m['params']['epochs'])
            puntos = optimizacion.curva(nombre, m['params'], train, validacion, datos.BASE_TRABAJO, epocas)
        optimizacion.reporte_modelo(nombre, estudio, m, puntos)
        print(f"   ndcg validación {estudio.best_value:.4f}, test {prueba['ndcg']:.4f} con {m['params']}", flush=True)


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
    universo, train, validacion, test = datos.particiones(df)
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
                     f"| {o['ndcg_validacion']:.4f} | {o['evaluacion']['lectores_cubiertos']:.0%} | {p['min_libro']} "
                     f"| {len(m['indice_lector'])} | {len(m['ids_libros'])} | {o['trials']} |")
        frio = sum(c <= recomendador.CASCADA_ARRANQUE_FRIO for c in cant.values())
        modelo = sum(c > recomendador.CASCADA_ARRANQUE_FRIO and l in m['indice_lector'] for l, c in cant.items())
        reparto.append(f'| cascada_{nombre} | {frio} | {len(cant) - frio - modelo} | {modelo} |')
    ndcg_pop = o['ndcg_popularidad'] if filas else float('nan')
    try:
        o = modelos.modelo('ensamble')['optimizacion']
        filas.append(f"| [ensamble](ensamble.md) | **{o['ndcg']:.4f}** | {o['ndcg'] / o['ndcg_popularidad']:.2f}× "
                     f"| {o['ndcg_validacion']:.4f} | {o['lectores_cubiertos']:.0%} | — | — | — | {o['trials']} |")
        modelo = sum(c > recomendador.CASCADA_ARRANQUE_FRIO and ensamble.conoce_lector(l) for l, c in cant.items())
        reparto.append(f'| cascada_ensamble | {frio} | {len(cant) - frio - modelo} | {modelo} |')
    except FileNotFoundError:
        pass

    lineas = [
        '# Optimización de los recomendadores', '',
        f"Generado el {datetime.datetime.now():%Y-%m-%d %H:%M} con una copia de `data/data.db`.", '',
        '## Datos', '',
        f"- Interacciones (rating > 0): {len(df)}, de {df['id_lector'].nunique()} lectores y {df['id_libro'].nunique()} libros.",
        f"- Universo de los modelos: los {universo['id_lector'].nunique()} lectores con al menos {datos.MIN_LECTOR} interacciones "
        f"({len(universo)} interacciones).",
        f"- Partición cronológica por lector: las últimas {datos.N_TEST} interacciones son test, las "
        f"{datos.N_VALIDACION} anteriores validación y el resto train.",
        f"- Train: {len(train)} interacciones. Validación: {len(validacion)}. Test: {len(test)}.",
        '', '## Método', '',
        '- Cada modelo se ajusta con Optuna (TPE): entrena con train y maximiza NDCG@20 en validación.',
        '- Con los mejores hiperparámetros se reentrena con train + validación y se mide en test: es el NDCG que se '
        'informa (el test no participó de ninguna elección).',
        f'- El modelo final se entrena con todos los lectores con al menos {datos.MIN_LECTOR_FINAL} interacciones '
        '(todas sus interacciones), con los hiperparámetros elegidos.',
        '- Todos los lectores del universo se entrenan y se evalúan, siempre los mismos.',
        '- El mínimo de interacciones por libro es un hiperparámetro, contado sólo sobre los datos de entrenamiento.',
        '- Relevante = cualquier libro leído, sin importar el rating.',
        '', '## Resultados', '',
        f'Referencia: recomendar a todos los más leídos da NDCG@20 = {ndcg_pop:.4f} en test.', '',
        '| Modelo | NDCG@20 (test) | vs popularidad | NDCG@20 (validación) | Lectores del universo que conoce | min_libro | Lectores (final) | Libros (final) | Trials |',
        '|---|---|---|---|---|---|---|---|---|',
        *filas,
        '', '## Cascadas', '',
        f'Hasta {recomendador.CASCADA_ARRANQUE_FRIO} valoraciones: top_n. Si el modelo no conoce al lector '
        f'(menos de {datos.MIN_LECTOR_FINAL} interacciones): visión de túnel. El resto: el modelo.', '',
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
