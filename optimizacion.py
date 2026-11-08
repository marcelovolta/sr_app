import datetime
import os
import time

import numpy as np
import optuna

import datos
import modelos


# Ajuste de hiperparámetros con Optuna, maximizando NDCG@20 en la partición
# cronológica de datos.particionar (las últimas 20 interacciones de cada lector).
#
# Decisiones de diseño:
# - El filtro de la base (mínimo de interacciones por lector y por libro) es un
#   hiperparámetro y se aplica sólo a train, para no usar información del test.
# - Se evalúa siempre a los mismos lectores (todos los que tienen test), sea
#   cual sea el filtro. Los que el filtro deja afuera reciben los libros más
#   leídos de train. Si no, un filtro más exigente se evaluaría sólo con
#   lectores más fáciles y parecería mejor de lo que es.
# - Relevante = cualquier libro leído en el test (el rating no importa), igual
#   que en las predicciones de Kaggle: acertar qué va a leer el lector.
# - El NDCG de los mejores hiperparámetros se mide en el mismo test con el que
#   se eligieron, así que es algo optimista.
K = 20
TRIALS = {'naive': 50, 'svd': 50, 'ease': 50, 'dos_torres': 50, 'gran_torre': 30}
ESTUDIOS = 'sqlite:///' + os.path.join(datos.RAIZ, 'data', 'trabajo', 'optuna.db')
CARPETA_REPORTES = os.path.join(datos.RAIZ, 'reportes')
DESCUENTO = 1 / np.log2(np.arange(2, K + 2))


class Evaluador:
    # Mide NDCG@K de un modelo entrenado sobre train contra el test
    def __init__(self, train, test):
        self.test = test.groupby('id_lector')['id_libro'].apply(set).to_dict()
        self.lectores = sorted(self.test)
        leidos = train.groupby('id_lector')['id_libro'].apply(set).to_dict()
        # respaldo para los lectores que el modelo no conoce: los más leídos de
        # train (sin filtrar) que el lector todavía no leyó
        populares = train['id_libro'].value_counts().index.to_numpy()
        self.respaldo = {}
        for l in self.lectores:
            ya = leidos.get(l, set())
            self.respaldo[l] = [b for b in populares[:K + len(ya)] if b not in ya][:K]

    def ndcg(self, listas):
        # NDCG@K binario promediado entre todos los lectores evaluados
        valores = []
        for l in self.lectores:
            rel = self.test[l]
            aciertos = np.array([b in rel for b in listas[l][:K]] + [False] * (K - len(listas[l][:K])))
            valores.append((aciertos * DESCUENTO).sum() / DESCUENTO[:min(len(rel), K)].sum())
        return float(np.mean(valores))

    def evaluar(self, m):
        # top 20 del modelo para los lectores que conoce (respaldo para el
        # resto) y su NDCG, más datos de cobertura
        listas = dict(self.respaldo)
        conocidos = [l for l in self.lectores if l in m['indice_lector']]
        indices = np.array([m['indice_lector'][l] for l in conocidos], dtype=np.int64)
        for desde in range(0, len(indices), 500):
            us = indices[desde:desde + 500]
            p = np.asarray(modelos.puntajes(m, us), dtype=np.float32)
            # sin los libros que el lector ya leyó en train
            filas = np.repeat(np.arange(len(us)), [len(m['leidos'][u]) for u in us])
            p[filas, np.concatenate([m['leidos'][u] for u in us])] = -np.inf
            top = np.argpartition(-p, K, axis=1)[:, :K]
            top = np.take_along_axis(top, np.argsort(-np.take_along_axis(p, top, axis=1), axis=1, kind='stable'), axis=1)
            for l, fila in zip(conocidos[desde:desde + 500], top):
                listas[l] = list(m['ids_libros'][fila])
        return {'ndcg': self.ndcg(listas), 'lectores_cubiertos': len(conocidos) / len(self.lectores),
                'libros': len(m['ids_libros']), 'lectores_modelo': len(m['indice_lector'])}


def optimizar(nombre, train, test, base, trials=None, continuar=False):
    # Busca los hiperparámetros con Optuna (TPE). El estudio se guarda en
    # data/trabajo/optuna.db, así que con continuar=True se retoma uno cortado
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if not continuar:
        try:
            optuna.delete_study(study_name=nombre, storage=ESTUDIOS)
        except KeyError:
            pass
    estudio = optuna.create_study(study_name=nombre, storage=ESTUDIOS, direction='maximize', load_if_exists=True,
                                  sampler=optuna.samplers.TPESampler(seed=modelos.SEMILLA))
    ev = Evaluador(train, test)

    def objetivo(trial):
        # entrena con los hiperparámetros del trial y devuelve su NDCG@20
        inicio = time.time()
        params = modelos.sugerir(nombre, trial)
        filtrado = datos.filtrar(train, params['min_lector'], params['min_libro'])
        r = ev.evaluar(modelos.entrenar(nombre, filtrado, params, base))
        r['segundos'] = round(time.time() - inicio)
        for k, v in r.items():
            trial.set_user_attr(k, v)
        print(f"  {nombre} trial {trial.number}: ndcg {r['ndcg']:.4f} ({r['segundos']} s) {params}", flush=True)
        return r['ndcg']

    faltan = (trials or TRIALS[nombre]) - len(estudio.trials)
    if faltan > 0:
        estudio.optimize(objetivo, n_trials=faltan)
    return estudio, ev.ndcg(ev.respaldo)


def entrenar_final(nombre, estudio, todos, base, ndcg_popularidad):
    # Con los mejores hiperparámetros: filtro y entrenamiento con todos los datos
    # (train + test). Se guarda en el .pkl el registro de la optimización
    params = modelos.sugerir(nombre, optuna.trial.FixedTrial(estudio.best_params))
    filtrado = datos.filtrar(todos, params['min_lector'], params['min_libro'])
    m = modelos.entrenar(nombre, filtrado, params, base)
    m['optimizacion'] = {
        'fecha': datetime.datetime.now().isoformat(timespec='minutes'),
        'ndcg': estudio.best_value,
        'ndcg_popularidad': ndcg_popularidad,
        'evaluacion': estudio.best_trial.user_attrs,
        'trials': len(estudio.trials),
        'interacciones': len(filtrado),
    }
    modelos.guardar(m)
    return m


def curva(nombre, params, train, test, base, epocas):
    # Curva de entrenamiento de una red: con los hiperparámetros elegidos se
    # entrena sobre train durante `epocas` (más que las elegidas, para ver qué
    # pasa después) y en cada época se registran la pérdida de entrenamiento y
    # el NDCG@20 en test. Sirve para ver si la red sigue aprendiendo (le falta
    # capacidad o épocas) o si empieza a sobreajustar.
    ev = Evaluador(train, test)
    filtrado = datos.filtrar(train, params['min_lector'], params['min_libro'])
    puntos = []

    def al_final_de_epoca(epoca, perdida, armar):
        # registra la pérdida y el NDCG@20 de la red en esta época
        puntos.append((epoca, perdida, ev.evaluar(armar())['ndcg']))
        print(f'  curva {nombre} época {epoca}: pérdida {perdida:.4f}, ndcg {puntos[-1][2]:.4f}', flush=True)

    modelos.entrenar(nombre, filtrado, dict(params, epochs=epocas), base, al_final_de_epoca)
    _graficar_curva(nombre, puntos, params['epochs'])
    return puntos


def _graficar_curva(nombre, puntos, epocas_elegidas):
    # dos paneles con el mismo eje x (pérdida y NDCG tienen escalas distintas:
    # nunca en un mismo gráfico con dos ejes y)
    import matplotlib
    import matplotlib.ticker
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    superficie, serie, tinta, tenue, grilla = '#fcfcfb', '#2a78d6', '#2b2a27', '#6b6a64', '#e4e3de'
    epoca, perdida, ndcg = zip(*puntos)
    fig, ejes = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True, facecolor=superficie)
    for ax, valores, titulo in ((ejes[0], perdida, 'Pérdida de entrenamiento'), (ejes[1], ndcg, 'NDCG@20 en test')):
        ax.set_facecolor(superficie)
        ax.plot(epoca, valores, color=serie, linewidth=2, marker='o', markersize=4)
        ax.axvline(epocas_elegidas, color=tenue, linewidth=1, linestyle='--')
        ax.set_title(titulo, loc='left', fontsize=10, color=tinta)
        ax.grid(axis='y', color=grilla, linewidth=0.8)
        ax.tick_params(colors=tenue, labelsize=8)
        for lado in ('top', 'right'):
            ax.spines[lado].set_visible(False)
        for lado in ('left', 'bottom'):
            ax.spines[lado].set_color(grilla)
    mejor = max(range(len(ndcg)), key=lambda k: ndcg[k])
    ejes[1].annotate(f'máximo {ndcg[mejor]:.4f} (época {epoca[mejor]})', (epoca[mejor], ndcg[mejor]),
                     textcoords='offset points', xytext=(0, 8), ha='center', va='bottom', fontsize=8, color=tinta)
    ejes[1].margins(y=0.15)  # lugar para la etiqueta del máximo
    ejes[1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ejes[0].annotate(f'épocas elegidas: {epocas_elegidas}', (epocas_elegidas, max(perdida)),
                     textcoords='offset points', xytext=(4, -4), fontsize=8, color=tenue, va='top')
    ejes[1].set_xlabel('Época', fontsize=9, color=tenue)
    fig.suptitle(f'{nombre}: curva de entrenamiento', x=0.02, ha='left', fontsize=12, color=tinta)
    fig.tight_layout()
    os.makedirs(CARPETA_REPORTES, exist_ok=True)
    fig.savefig(os.path.join(CARPETA_REPORTES, f'curva_{nombre}.png'), dpi=150, facecolor=superficie)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reportes en markdown
# ---------------------------------------------------------------------------

def _fmt(v):
    # números con 4 cifras significativas para las tablas
    if isinstance(v, float):
        return f'{v:.4g}'
    return str(v)


def reporte_modelo(nombre, estudio, m, puntos_curva=None):
    # reportes/<modelo>.md: mejores hiperparámetros, importancia y mejores trials
    o = m['optimizacion']
    ev = o['evaluacion']
    lineas = [
        f'# {nombre}', '',
        f"Optimizado el {o['fecha']} con {o['trials']} trials de Optuna (TPE), maximizando NDCG@{K}.", '',
        '## Resultado', '',
        '| Métrica | Valor |', '|---|---|',
        f"| NDCG@{K} (test) | **{o['ndcg']:.4f}** |",
        f"| NDCG@{K} de popularidad (referencia) | {o['ndcg_popularidad']:.4f} |",
        f"| Lectores del test cubiertos por el modelo | {ev['lectores_cubiertos']:.1%} |",
        f"| Libros del modelo (en la evaluación) | {ev['libros']} |",
        f"| Tiempo por trial | {ev['segundos']} s |",
        f"| Modelo final: interacciones / lectores / libros | {o['interacciones']} / {len(m['indice_lector'])} / {len(m['ids_libros'])} |",
        '', '## Hiperparámetros elegidos', '',
        '| Hiperparámetro | Valor |', '|---|---|',
        *[f'| {k} | {_fmt(v)} |' for k, v in m['params'].items()],
    ]
    try:
        importancia = optuna.importance.get_param_importances(estudio)
        lineas += ['', '## Importancia de cada hiperparámetro (fANOVA)', '',
                   '| Hiperparámetro | Importancia |', '|---|---|',
                   *[f'| {k} | {v:.2f} |' for k, v in importancia.items()]]
    except Exception:
        pass
    completos = sorted((t for t in estudio.trials if t.value is not None), key=lambda t: -t.value)
    claves = list(m['params'])
    lineas += ['', '## Mejores trials', '',
               '| # | NDCG@20 | ' + ' | '.join(claves) + ' |',
               '|---|---|' + '---|' * len(claves),
               *[f'| {t.number} | {t.value:.4f} | ' + ' | '.join(_fmt(t.params.get(k)) for k in claves) + ' |'
                 for t in completos[:10]]]
    if puntos_curva:
        lineas += ['', '## Curva de entrenamiento', '',
                   f"Con los hiperparámetros elegidos, entrenando sobre train más épocas que las elegidas "
                   f"({m['params']['epochs']}, línea punteada).", '',
                   f'![Curva de entrenamiento de {nombre}](curva_{nombre}.png)', '',
                   '| Época | Pérdida | NDCG@20 |', '|---|---|---|',
                   *[f'| {e} | {p:.4f} | {n:.4f} |' for e, p, n in puntos_curva]]
    os.makedirs(CARPETA_REPORTES, exist_ok=True)
    with open(os.path.join(CARPETA_REPORTES, f'{nombre}.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lineas) + '\n')
