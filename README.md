# Recomendador de libros

Aplicación Flask que recomienda libros a lectores, y generador de las predicciones que se
suben a Kaggle (`data/prediction_<algoritmo>.csv`). Incluye algoritmos basados en SQL,
modelos entrenados (Surprise, EASE, redes neuronales), un ensamble y cascadas que combinan
según cuánto se conoce a cada lector.

Los resultados detallados están en [`reportes/`](reportes/resumen.md).

## Estructura

| Archivo | Qué hace |
|---|---|
| `app.py` | La aplicación web (Flask): ingreso, recomendaciones, valoraciones y `/predecir/<algoritmo>`. |
| `recomendador.py` | Los algoritmos SQL, el acceso a los modelos, las cascadas y la generación de los CSV. |
| `datos.py` | Copia de la base, lectura de interacciones, filtro recursivo y partición train/test. |
| `modelos.py` | Los modelos entrenados: espacio de búsqueda, entrenamiento y predicción con NumPy. |
| `optimizacion.py` | Evaluación (NDCG@20), Optuna, entrenamiento final, curvas de entrenamiento y reportes. |
| `ensamble.py` | El ensamble con LightGBM sobre las predicciones de los modelos. |
| `pipeline.py` | El proceso completo, para volver a correrlo cuando cambie la base. |
| `templates/` | Páginas de la app. |
| `modelos/` | Los modelos entrenados (`.pkl`), no versionados. |
| `reportes/` | Reportes en markdown para la presentación. |

## Cómo se usa

```bash
python pipeline.py                      # todo: copia, optimización, modelos, ensamble, CSV y reportes (~1 h)
python pipeline.py --modelos ease svd   # sólo esos modelos ('ensamble' necesita los demás ya entrenados)
python pipeline.py --trials 5           # prueba rápida
python pipeline.py --sin-copia          # usa la copia de trabajo existente
python pipeline.py --continuar          # retoma estudios de Optuna cortados
python pipeline.py --solo-predicciones  # sólo los CSV y el resumen, con los modelos actuales
flask --app app run                     # la aplicación
```

Dependencias (Python 3.13): Flask, NumPy, pandas, SciPy, scikit-surprise, Keras con JAX,
Optuna, LightGBM y matplotlib.

## El proceso (`pipeline.py`)

1. **Copia de la base.** `data/data.db` puede estar siendo modificada por otro proceso y no
   se puede bloquear. Se copia el archivo sin abrirlo con SQLite, y la copia se acepta sólo si
   el archivo no cambió durante la copia, no había una transacción a medio escribir y pasa el
   chequeo de integridad. Todo lo demás trabaja sobre `data/trabajo/data.db`.
2. **Partición train/test.** Ver [Evaluación](#evaluación).
3. **Optimización de cada modelo** con Optuna, entrenamiento final con todos los datos,
   curva de entrenamiento (redes) y reporte.
4. **Ensamble.**
5. **Predicciones**: borra los CSV viejos y genera los de todos los algoritmos.
6. **Resumen**: `reportes/resumen.md`.

## Datos

- Se usan las valoraciones (rating > 0) de libros y lectores que existen en sus tablas.
- **Orden cronológico por lector.** Las fechas son por día y hay muchos empates (cargas
  masivas), así que se desempata por orden de inserción.
- **Filtro recursivo** (como en el artículo de EASE de stepbystepdatascience.com): se sacan los
  libros con menos de `min_libro` interacciones y los lectores con menos de `min_lector`, y se
  repite hasta que no cambie nada, porque cada paso puede dejar a otros por debajo del mínimo.
  Los dos mínimos son **hiperparámetros de cada modelo** (`min_lector` ≥ 20).

## Evaluación

- **Partición cronológica por lector:** las últimas 20 interacciones de cada lector son test y
  las anteriores train, así ningún modelo ve el futuro de lo que se evalúa. Los lectores con 20
  interacciones o menos quedan enteros en train: sirven para entrenar pero no se evalúan.
  Quedan 3801 lectores evaluados.
- **Métrica: NDCG@20.** Relevante = cualquier libro leído en el test, sin importar el rating
  (igual que en Kaggle: acertar qué va a leer).
- **Siempre los mismos lectores evaluados.** Los que el filtro deja afuera de un modelo reciben
  los más leídos de train. Si no, un filtro exigente se evaluaría sólo con lectores fáciles y
  parecería mejor de lo que es.
- **El filtro se aplica sólo a train**, para no usar información del test.
- Los hiperparámetros se eligen en el mismo test con el que se informa el NDCG, así que los
  valores son algo optimistas. Sirven para comparar modelos entre sí.

## Algoritmos SQL (`recomendador.py`)

No se entrenan: consultan la base directamente.

| Algoritmo | Qué recomienda |
|---|---|
| `random` | Libros al azar. |
| `top_n` | Mejor promedio de rating, con al menos 5 valoraciones (para que un único 10 no domine). |
| `mas_valorados` | Los libros con más valoraciones. |
| `perfil_balanceado` | 0,4 × afinidad con el género + 0,4 × afinidad con el autor + 0,2 × rating corregido del libro, sobre los libros con más de 5 valoraciones. La afinidad sale de los géneros y autores que el lector valoró al menos 3 veces. |
| `vision_tunel` | Primero los libros que coinciden en género **y** autor con los que valoró el lector; si no alcanzan, los populares. Puede recomendar cualquier libro de la base. |

Todos excluyen lo que el lector ya valoró o vio, y los libros sin título.

**Optimización:** antes cada uno era una consulta por lector que recalculaba sobre toda la
tabla un ranking global igual para todos (por ejemplo, el promedio de cada libro). Ahora la
parte global se calcula una vez y queda en memoria, y se recalcula sólo si la base cambia.
Por lector sólo se consulta lo suyo. Generar todos los CSV pasó de más de una hora (sólo
`top_n` tardaba 25 min) a pocos minutos. Los resultados son los mismos; los empates (por ejemplo, varios libros con
promedio 10) ahora se desempatan por id, mientras que antes salían en un orden arbitrario.

## Modelos entrenados (`modelos.py`)

Cada modelo se guarda como un diccionario de arrays de NumPy, y la predicción se calcula con
NumPy: para recomendar no hacen falta Surprise ni Keras.

### naive (BaselineOnly de Surprise)
Rating estimado = media + sesgo del lector + sesgo del libro. Para ordenar sólo importa el
sesgo del libro, así que recomienda lo mismo a todos. Hiperparámetros: regularización del
libro y del lector, épocas.

### svd (SVD de Surprise)
Factorización de matrices sobre el rating: lectores y libros como vectores latentes.
Hiperparámetros: cantidad de factores, épocas, learning rate, regularización.

### ease (Steck, 2019)
Modelo lineal libro a libro con solución cerrada: `B = I − P / diag(P)` con
`P = (XᵀX + λI)⁻¹` y la diagonal de B en 0. X es binaria (leyó o no). El puntaje de un libro
es la suma de los pesos que le dan los libros que el lector leyó. Único hiperparámetro: λ.

### dos_torres
Un embedding por lector y otro por libro; puntaje = similitud coseno. Se entrena con softmax
sobre los **negativos del batch**: cada par (lector, libro elegido) usa como negativos los
libros de las demás filas. A cada libro se le resta el log de su frecuencia (**corrección
logQ**, Yi et al. 2019), porque si no, los populares aparecen demasiado como negativos y el
modelo se va a los populares. Elegido = rating ≥ `rating_positivo`. Hiperparámetros:
`rating_positivo`, dimensión, temperatura, learning rate, épocas.

### gran_torre
Una red que mira cada par (lector, libro): embeddings del lector, del libro y de sus rasgos
(autor, género, editorial y año de edición del libro; género y nacimiento del lector),
concatenados y pasados por 3 capas densas de 256 hasta una sigmoide. Aprende a separar los
libros elegidos de libros al azar no leídos (`negativos` por cada positivo), sorteados de
nuevo en cada época porque con negativos fijos la red los memoriza. Para predecir rápido, la
primera capa se separa en una parte del lector y otra del libro que se precalculan.

## Ensamble (`ensamble.py`)

- **Candidatos:** los 50 primeros libros de cada modelo (SVD, EASE, dos torres y gran torre),
  unidos.
- **Variables:** el puntaje y la posición que le da cada modelo a cada candidato.
- **Modelo:** LightGBM con objetivo LambdaRank reordena los candidatos de cada lector.
- **Sin fuga de datos (stacking):** LightGBM nunca aprende de predicciones de modelos que
  vieron las respuestas.
  - Etapa A: los modelos base se entrenan con train sin sus últimas 20 interacciones por
    lector, y LightGBM aprende a predecir esas 20.
  - Etapa B: los modelos base se entrenan con train, y el ensamble se evalúa en test. Con esto
    Optuna elige los hiperparámetros de LightGBM.
  - Final: LightGBM se entrena con la etapa B y se aplica sobre los modelos finales.

## Cascadas

`cascada_<modelo>` para cada modelo y para el ensamble:

1. Hasta 2 valoraciones (arranque en frío): `top_n`.
2. Si el modelo no conoce al lector (quedó afuera del filtro): `vision_tunel`, que puede
   recomendar cualquier libro.
3. El resto: el modelo.

## Resultados

NDCG@20 en test (detalle en [`reportes/resumen.md`](reportes/resumen.md)):

| Modelo | NDCG@20 | vs popularidad |
|---|---|---|
| ensamble | **0,1251** | 1,41× |
| ease | 0,1211 | 1,36× |
| dos_torres | 0,1178 | 1,33× |
| gran_torre | 0,1070 | 1,20× |
| svd | 0,0880 | 0,99× |
| naive | 0,0878 | 0,99× |
| *popularidad* | *0,0889* | |

## Decisiones tomadas

| Decisión | Por qué |
|---|---|
| Partición por las últimas 20 interacciones de cada lector | Evaluar como en Kaggle (predecir lo próximo que va a leer) sin que ningún modelo vea el futuro. |
| El filtro de la base es un hiperparámetro de cada modelo | Antes se usaba una base filtrada fija (más de 35). Cada modelo prefiere un filtro distinto: EASE y las torres casi el mínimo (más cobertura), SVD y naive el máximo (para usarlos lo menos posible). |
| Evaluar siempre a los mismos lectores, con respaldo de popularidad | Para que un filtro exigente no parezca mejor por evaluar sólo a lectores fáciles. |
| Relevante = cualquier libro leído | Es lo que se predice en Kaggle. En una simulación de Kaggle, los dos criterios (cualquier rating o rating ≥ 8) daban resultados parecidos. |
| Dos torres con negativos y logQ, gran torre con negativos | Las versiones de clase aproximaban el rating y recomendaban casi lo mismo a todos (el 86% de los lectores recibía el mismo primer libro con la gran torre). |
| No se agregó complejidad a las redes | Se probó (atributos y capas en dos torres; más capas, dropout, `espania` y antigüedad del libro en gran torre) y no mejoró: Optuna eligió siempre lo más simple. Ver [`reportes/intento_complejidad.md`](reportes/intento_complejidad.md) y [`reportes/literatura.md`](reportes/literatura.md). |
| Curvas de entrenamiento en los reportes | Muestran que dos torres se estabiliza desde la época 6 y que la gran torre sobreajusta si se entrena de más. |
| Ensamble con LightGBM | Combinar ayuda (+0,004 sobre EASE). LightGBM rinde igual que un promedio simple de los puntajes estandarizados (0,1251 contra 0,1252), probablemente porque aprende de pocos lectores (1768) y sólo con los puntajes. |
| Algoritmos SQL con el ranking global en memoria | Las consultas repetían por lector una agregación de toda la tabla. |
| Copia de la base sin SQLite | No bloquear al proceso que está modificando `data/data.db`. |
| KNN se dejó de usar | Predecir libro por libro es muy lento para evaluarlo con Optuna, y no estaba entre los mejores. |
| LightGBM con su API nativa | La API estilo scikit-learn obliga a instalar scikit-learn. |

## Limitaciones conocidas

- Los NDCG son algo optimistas (los hiperparámetros se eligen en el mismo test).
- `modelos/ease.pkl` pesa ~530 MB (matriz de 11 mil × 11 mil libros).
- Los modelos se entrenan con una copia de la base: lo que se valore en la app después no lo
  ven hasta volver a correr `pipeline.py`.
- En `gran_torre` los años (nacimiento, edición) se escalan entre 0 y 1 con los faltantes en 0,
  lo que deforma la escala.
- `insertar_valorado` ignora una nueva valoración si el lector ya había valorado ese libro.
