import datetime
import os
import sqlite3
import time

import numpy as np
import optuna
import pandas as pd

import datos
import modelos
import optimizacion


# Ensamble de los modelos no SQL: un LightGBM (LambdaRank) reordena los
# candidatos que proponen SVD, EASE, dos torres y gran torre, usando como
# variables el puntaje y la posición que le da cada modelo a cada candidato, y
# variables de contenido: qué parte de lo que leyó el lector comparte autor,
# género o editorial con el candidato, el año de edición y la popularidad.
#
# Para no tener fuga de datos (stacking), LightGBM nunca aprende de predicciones
# de modelos que vieron las respuestas, y el test no participa de ninguna
# elección (mismas etapas que los modelos sueltos, ver optimizacion.py):
# - etapa A (validación): los modelos base se entrenan con train y LightGBM
#   aprende a predecir la validación. Optuna elige sus hiperparámetros
#   entrenando con el 80% de los lectores y midiendo en el 20% restante;
# - etapa B (test): los modelos base se entrenan con train + validación y el
#   LightGBM de la etapa A se evalúa en test, igual que los demás modelos
#   (NDCG@20 comparable con su tabla);
# - final: LightGBM se reentrena con la etapa B (las respuestas más recientes)
#   y se aplica sobre los modelos finales (entrenados con todo el universo).
# Como referencia se mide también un promedio simple: la media de los puntajes
# de cada modelo, estandarizados por lector.
MODELOS_BASE = ('svd', 'ease', 'dos_torres', 'gran_torre')
CANDIDATOS = 50  # por modelo; los candidatos son la unión
TRIALS = 30
FRACCION_EVALUACION = 0.2  # lectores de la etapa A apartados para que Optuna mida
CONTENIDO = ['afinidad_autor', 'afinidad_genero', 'afinidad_editorial', 'anio_edicion', 'popularidad']
COLUMNAS = [f'{n}_{v}' for n in MODELOS_BASE for v in ('puntaje', 'posicion')] + CONTENIDO


def tabla_libros(base):
    # autor, género y editorial (códigos, -1 = sin dato) y año de edición de
    # todos los libros, para las variables de contenido
    with sqlite3.connect(base) as conn:
        libros = pd.read_sql('SELECT id_libro, autor, genero, editorial, anio_edicion FROM libros', conn)
    libros = libros.drop_duplicates('id_libro').set_index('id_libro')
    tabla = pd.DataFrame({a: libros[a].astype('category').cat.codes.astype(np.int32)
                          for a in ('autor', 'genero', 'editorial')}, index=libros.index)
    anio = pd.to_numeric(libros['anio_edicion'], errors='coerce')
    tabla['anio_edicion'] = anio.where(anio > 0).astype(np.float32)
    return tabla


def _contenido(libros, id_lector, m_ease, tabla):
    # variables de contenido de los candidatos `libros` de un lector. Lo leído
    # sale del modelo EASE de la misma etapa (el de catálogo más grande), así
    # que nunca incluye lo que se está prediciendo
    t = tabla.reindex(libros)
    u = m_ease['indice_lector'].get(id_lector)
    columnas = {}
    if u is None:
        for a in ('autor', 'genero', 'editorial'):
            columnas[f'afinidad_{a}'] = np.full(len(libros), np.nan, dtype=np.float32)
    else:
        leido = tabla.reindex(m_ease['ids_libros'][m_ease['leidos'][u]])
        for a in ('autor', 'genero', 'editorial'):
            frecuencia = leido[a][leido[a] >= 0].value_counts() / max(len(leido), 1)
            columnas[f'afinidad_{a}'] = t[a].map(frecuencia).fillna(0).where(t[a] >= 0).to_numpy(dtype=np.float32)
    columnas['anio_edicion'] = t['anio_edicion'].to_numpy(dtype=np.float32)
    j = pd.Index(m_ease['ids_libros']).get_indexer(libros)
    columnas['popularidad'] = np.where(j >= 0, np.log1p(m_ease['popularidad'][j]), np.nan).astype(np.float32)
    return columnas


def candidatos(ms, lectores, tabla, lote=300):
    # Candidatos de cada lector (los CANDIDATOS primeros de cada modelo, sin lo
    # ya leído) con el puntaje y la posición en cada modelo y las variables de
    # contenido. Si un modelo no conoce al lector o al libro, sus columnas
    # quedan vacías (LightGBM lo maneja)
    indices = {n: pd.Index(m['ids_libros']) for n, m in ms.items()}
    partes = []
    for desde in range(0, len(lectores), lote):
        grupo = lectores[desde:desde + lote]
        por_modelo = {}
        for n, m in ms.items():
            conocidos = [l for l in grupo if l in m['indice_lector']]
            if not conocidos:
                continue
            us = np.array([m['indice_lector'][l] for l in conocidos])
            p = np.asarray(modelos.puntajes(m, us), dtype=np.float32)
            filas = np.repeat(np.arange(len(us)), [len(m['leidos'][u]) for u in us])
            p[filas, np.concatenate([m['leidos'][u] for u in us])] = np.nan
            orden = np.argsort(np.where(np.isnan(p), np.inf, -p), axis=1, kind='stable')
            posicion = np.empty_like(orden)
            np.put_along_axis(posicion, orden, np.arange(p.shape[1])[None, :], axis=1)
            for k, l in enumerate(conocidos):
                top = orden[k, :CANDIDATOS]
                top = top[~np.isnan(p[k, top])]  # con un catálogo chico podrían colarse leídos
                por_modelo.setdefault(l, {})[n] = (p[k], posicion[k], top)
        for l, info in por_modelo.items():
            libros = pd.unique(np.concatenate([ms[n]['ids_libros'][top] for n, (_, _, top) in info.items()]))
            fila = {'id_lector': l, 'id_libro': libros}
            for n in MODELOS_BASE:
                if n in info:
                    p, posicion, _ = info[n]
                    j = indices[n].get_indexer(libros)
                    fila[f'{n}_puntaje'] = np.where(j >= 0, p[j], np.nan)
                    fila[f'{n}_posicion'] = np.where(j >= 0, posicion[j] + 1, np.nan).astype(np.float32)
                else:
                    fila[f'{n}_puntaje'] = fila[f'{n}_posicion'] = np.full(len(libros), np.nan, dtype=np.float32)
            fila.update(_contenido(libros, l, ms['ease'], tabla))
            partes.append(pd.DataFrame(fila))
    return pd.concat(partes, ignore_index=True)


def _con_etiquetas(df, relevantes):
    # 1 si el candidato está entre lo que el lector leyó después
    df['relevante'] = [b in relevantes.get(l, ()) for l, b in zip(df['id_lector'], df['id_libro'])]
    return df


def _listas(df, puntaje):
    # top 20 de cada lector según `puntaje`
    df = df.assign(_p=puntaje).sort_values(['id_lector', '_p'], ascending=[True, False], kind='stable')
    return df.groupby('id_lector')['id_libro'].apply(lambda s: list(s[:optimizacion.K])).to_dict()


def promedio_simple(df):
    # media de los puntajes de cada modelo, estandarizados por lector
    z = []
    for n in MODELOS_BASE:
        g = df.groupby('id_lector')[f'{n}_puntaje']
        z.append((df[f'{n}_puntaje'] - g.transform('mean')) / g.transform('std').replace(0, 1))
    return pd.concat(z, axis=1).mean(axis=1, skipna=True).fillna(-np.inf)


def _sugerir(trial):
    # espacio de búsqueda de los hiperparámetros de LightGBM
    return {
        'num_leaves': trial.suggest_int('num_leaves', 8, 128, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 300, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
    }


def _ajustar(df, params):
    # API nativa de LightGBM (la de estilo scikit-learn pide instalar scikit-learn).
    # Cada lector es un grupo: LambdaRank aprende a ordenar dentro de cada uno
    import lightgbm as lgb
    df = df.sort_values('id_lector', kind='stable')
    grupos = df.groupby('id_lector', sort=False).size().to_numpy()
    datos_lgb = lgb.Dataset(df[COLUMNAS], df['relevante'].astype(int), group=grupos)
    return lgb.train({'objective': 'lambdarank', 'seed': modelos.SEMILLA, 'verbose': -1, **params}, datos_lgb)


def _bases(df, base):
    # los modelos base con sus hiperparámetros ya elegidos, entrenados con df
    ms = {}
    for n in MODELOS_BASE:
        p = modelos.modelo(n)['params']
        ms[n] = modelos.entrenar(n, datos.filtrar_libros(df, p['min_libro']), p, base)
    return ms


def optimizar_y_entrenar(train, validacion, resto, test, base, trials=TRIALS):
    # etapas A y B (ver arriba), Optuna sobre LightGBM, modelo final y reporte.
    # resto = train + validación
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    inicio = time.time()

    # etapa A: candidatos de los modelos entrenados con train, etiquetados con
    # la validación. Los lectores se reparten entre ajuste y evaluación
    relevantes_a = validacion.groupby('id_lector')['id_libro'].apply(set).to_dict()
    tabla = tabla_libros(base)
    cand_a = _con_etiquetas(candidatos(_bases(train, base), sorted(relevantes_a), tabla), relevantes_a)
    lectores_a = np.array(sorted(relevantes_a))
    np.random.default_rng(modelos.SEMILLA).shuffle(lectores_a)
    evaluados = set(lectores_a[:int(len(lectores_a) * FRACCION_EVALUACION)])
    ajuste = cand_a[~cand_a['id_lector'].isin(evaluados)]
    ajuste = ajuste[ajuste.groupby('id_lector')['relevante'].transform('any')]  # sin positivos no enseña nada
    cand_eval = cand_a[cand_a['id_lector'].isin(evaluados)]
    ev_a = optimizacion.Evaluador(train, validacion[validacion['id_lector'].isin(evaluados)])
    print(f'  ensamble etapa A: {len(ajuste)} candidatos para ajustar, {len(evaluados)} lectores para evaluar', flush=True)

    def objetivo(trial):
        # LightGBM con los hiperparámetros del trial, medido en los lectores apartados
        ranker = _ajustar(ajuste, _sugerir(trial))
        valor = ev_a.ndcg({**ev_a.respaldo, **_listas(cand_eval, ranker.predict(cand_eval[COLUMNAS]))})
        print(f'  ensamble trial {trial.number}: ndcg (validación) {valor:.4f}', flush=True)
        return valor

    estudio = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=modelos.SEMILLA))
    estudio.optimize(objetivo, n_trials=trials)
    params = estudio.best_params

    # etapa B: LightGBM de la etapa A (con todos sus lectores) evaluado en test,
    # sobre los modelos base entrenados con train + validación
    ranker_a = _ajustar(cand_a[cand_a.groupby('id_lector')['relevante'].transform('any')], params)
    ev = optimizacion.Evaluador(resto, test)
    relevantes_b = test.groupby('id_lector')['id_libro'].apply(set).to_dict()
    bases_b = _bases(resto, base)
    cand_b = _con_etiquetas(candidatos(bases_b, ev.lectores, tabla), relevantes_b)

    def ndcg(listas):
        # NDCG@20 en test de todos los lectores del universo (respaldo para los no cubiertos)
        return ev.ndcg({**ev.respaldo, **listas})

    ndcg_test = ndcg(_listas(cand_b, ranker_a.predict(cand_b[COLUMNAS])))
    referencia = {n: ev.evaluar(m)['ndcg'] for n, m in bases_b.items()}
    referencia['promedio simple'] = ndcg(_listas(cand_b, promedio_simple(cand_b)))
    print(f'  ensamble test: {ndcg_test:.4f}; referencias: { {k: round(v, 4) for k, v in referencia.items()} }', flush=True)

    # final: LightGBM con la etapa B (los modelos base finales ya están guardados)
    final = _ajustar(cand_b[cand_b.groupby('id_lector')['relevante'].transform('any')], params)
    m = {
        'nombre': 'ensamble',
        'params': params,
        'booster': final.model_to_string(),
        'libros': tabla,
        'optimizacion': {
            'fecha': datetime.datetime.now().isoformat(timespec='minutes'),
            'ndcg': ndcg_test,
            'ndcg_validacion': estudio.best_value,
            'ndcg_popularidad': ev.ndcg(ev.respaldo),
            'referencia': referencia,
            'trials': trials,
            'lectores_cubiertos': cand_b['id_lector'].nunique() / len(ev.lectores),
            'importancia': dict(zip(COLUMNAS, final.feature_importance('gain').tolist())),
            'minutos': round((time.time() - inicio) / 60),
        },
    }
    modelos.guardar(m)
    reporte(m)
    return m


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------

_booster = {}


def _ranker():
    # el LightGBM del modelo guardado, cargado una sola vez
    import lightgbm as lgb
    m = modelos.modelo('ensamble')
    if id(m) not in _booster:
        _booster.clear()
        _booster[id(m)] = lgb.Booster(model_str=m['booster'])
    return _booster[id(m)]


def conoce_lector(id_lector):
    # lo conoce si lo conoce alguno de los modelos base
    return any(modelos.conoce_lector(n, id_lector) for n in MODELOS_BASE)


def ranking(id_lector, promedio=False):
    # candidatos de los modelos finales reordenados por LightGBM (o, con
    # promedio=True, por el promedio simple de los puntajes estandarizados).
    # Si ningún modelo conoce al lector, los más leídos (de EASE, el de
    # catálogo más grande)
    ms = {n: modelos.modelo(n) for n in MODELOS_BASE}
    if not conoce_lector(id_lector):
        yield from modelos.ranking('ease', id_lector)
        return
    df = candidatos(ms, [id_lector], modelos.modelo('ensamble')['libros'])
    puntaje = promedio_simple(df).to_numpy() if promedio else _ranker().predict(df[COLUMNAS])
    for j in np.argsort(-puntaje, kind='stable'):
        yield df['id_libro'].iat[j], puntaje[j]
    # si el lector ya vio todos los candidatos (en la app), se sigue con EASE
    propuestos = set(df['id_libro'])
    yield from ((b, p) for b, p in modelos.ranking('ease', id_lector) if b not in propuestos)


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def reporte(m):
    # reportes/ensamble.md: comparación, importancia de variables e hiperparámetros
    o = m['optimizacion']
    importancia = sorted(o['importancia'].items(), key=lambda x: -x[1])
    total = sum(v for _, v in importancia) or 1
    lineas = [
        '# Ensamble (LightGBM)', '',
        f"Optimizado el {o['fecha']} con {o['trials']} trials de Optuna (TPE), maximizando NDCG@{optimizacion.K} "
        'en validación (lectores apartados de la etapa A).', '',
        '## Cómo funciona', '',
        f'- Candidatos: los {CANDIDATOS} primeros libros de cada modelo base ({", ".join(MODELOS_BASE)}), unidos.',
        '- Variables: el puntaje y la posición de cada candidato en cada modelo, y de contenido: qué parte de lo '
        'que leyó el lector comparte autor, género o editorial con el candidato, el año de edición y la popularidad.',
        '- LightGBM con objetivo LambdaRank reordena los candidatos.',
        '- Sin fuga de datos: LightGBM aprende a predecir la validación con modelos base entrenados con train, '
        'y se evalúa en test con modelos base entrenados con train + validación.',
        '', '## Resultado (NDCG@20 en test)', '',
        '| Modelo | NDCG@20 |', '|---|---|',
        f"| **Ensamble LightGBM** | **{o['ndcg']:.4f}** |",
        *[f'| {k} | {v:.4f} |' for k, v in sorted(o['referencia'].items(), key=lambda x: -x[1])],
        f"| popularidad | {o['ndcg_popularidad']:.4f} |",
        '', f"NDCG@20 en validación (el que maximizó Optuna): {o['ndcg_validacion']:.4f}.",
        '', f"Lectores del universo que conoce algún modelo base: {o['lectores_cubiertos']:.0%}.",
        '', '## Importancia de las variables (ganancia)', '',
        '| Variable | Importancia |', '|---|---|',
        *[f'| {k} | {v / total:.0%} |' for k, v in importancia],
        '', '## Hiperparámetros de LightGBM', '',
        '| Hiperparámetro | Valor |', '|---|---|',
        *[f'| {k} | {optimizacion._fmt(v)} |' for k, v in m['params'].items()],
    ]
    os.makedirs(optimizacion.CARPETA_REPORTES, exist_ok=True)
    with open(os.path.join(optimizacion.CARPETA_REPORTES, 'ensamble.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lineas) + '\n')
