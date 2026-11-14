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
| `scraping/completar_libros.py` | Completa los datos faltantes de los libros desde quelibroleo.com (independiente del resto). |
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
Optuna, LightGBM y matplotlib. El script de scraping necesita además `scrapling[fetchers]`.

## Datos de los libros (`scraping/completar_libros.py`)

Completa los campos vacíos de la tabla `libros` (`titulo`, `autor`, `genero`, `editorial`,
`anio_edicion` y `resumen`) con la ficha de cada libro en `https://quelibroleo.com/<id_libro>`,
usando scrapling. Escribe directamente en `data/data.db`.

```bash
python scraping/completar_libros.py --dry-run --limite 20   # prueba sin escribir
python scraping/completar_libros.py                         # todas las filas incompletas
python scraping/completar_libros.py --pausa 1.5             # segundos entre pedidos (1 por defecto)
python scraping/completar_libros.py --reintentar-fallidos   # vuelve a pedir las páginas ya marcadas
```

- Sólo escribe los campos vacíos: nunca pisa datos existentes.
- Guarda cada libro al procesarlo, así que se puede cortar y retomar.
- Los títulos se guardan en mayúsculas, como los que ya estaban en la base.
- El año 1900 se toma como desconocido: el sitio lo muestra cuando no tiene el dato.
- Los libros que dan 404, o cuya página no tiene todos los datos faltantes, quedan en la tabla
  `libros_scraping_fallidos` (con el motivo) y no se vuelven a pedir.
- A ~2,2 s por libro, unos 80 mil libros llevan alrededor de 48 horas.

En noviembre de 2026 completó unos 78 mil libros que no tenían ningún dato. Faltan:

| Campo | Vacíos antes | Vacíos después |
|---|---|---|
| titulo | 78 222 | 281 |
| autor | 78 227 | 286 |
| genero | 78 222 | 281 |
| editorial | 78 230 | 289 |
| anio_edicion | 78 222 | 610 |
| resumen | 79 947 | 14 146 |

Los ~280 que siguen sin título, autor, género ni editorial son libros que ya no existen en el
sitio. Los resúmenes que faltan son de libros que el sitio tiene sin resumen. Al volver a
entrenar, los modelos no cambiaron (no usan el título y los atributos ya estaban completos);
sí cambiaron `vision_tunel` y las cascadas, que ahora pueden recomendar los libros que antes
no tenían título.

## El proceso (`pipeline.py`)

1. **Copia de la base.** `data/data.db` puede estar siendo modificada por otro proceso y no
   se puede bloquear. Se copia el archivo sin abrirlo con SQLite, y la copia se acepta sólo si
   el archivo no cambió durante la copia, no había una transacción a medio escribir y pasa el
   chequeo de integridad. Todo lo demás trabaja sobre `data/trabajo/data.db`.
2. **Universo y partición** train / validación / test. Ver [Evaluación](#evaluación).
3. **Optimización de cada modelo** con Optuna (train → validación), medición en test
   (train + validación → test), entrenamiento final con los lectores con al menos 10
   interacciones, curva de entrenamiento (redes) y reporte.
4. **Ensamble.**
5. **Predicciones**: borra los CSV viejos y genera los de todos los algoritmos.
6. **Resumen**: `reportes/resumen.md`.

## Datos

- Se usan las valoraciones (rating > 0) de libros y lectores que existen en sus tablas.
- **Orden cronológico por lector.** Las fechas son por día y hay muchos empates (cargas
  masivas), así que se desempata por orden de inserción.
- **Universo de evaluación:** los lectores con al menos 30 interacciones (3203). Es fijo, no
  un hiperparámetro: si cambiara entre trials, cada trial se evaluaría con otros lectores y los
  NDCG no serían comparables.
- **Universo del modelo final:** los lectores con al menos 10 interacciones (5241), tantas como
  el train mínimo de la optimización. Los que tienen menos no entran a los modelos; en las
  cascadas los atienden `top_n` o `vision_tunel`.
- **Filtro de libros:** se sacan los libros con menos de `min_libro` interacciones, contadas
  sólo sobre los datos de entrenamiento de cada etapa (para no usar lecturas futuras). Es un
  **hiperparámetro de cada modelo** y no saca lectores.

## Evaluación

- **Partición cronológica por lector**, dentro del universo:

  | Tramo | Interacciones de cada lector |
  |---|---|
  | Test | las últimas 10 |
  | Validación | las 10 anteriores |
  | Train | el resto (al menos 10) |

  Ningún modelo ve el futuro de lo que se evalúa, y **todo lector con el que se entrena también
  se evalúa**, y viceversa.
- **Tres etapas:**
  1. Optuna entrena con train y elige los hiperparámetros maximizando NDCG@20 en validación.
  2. Con los mejores, se reentrena con train + validación y se mide en test. **Ese es el NDCG que
     se informa**: el test no participó de ninguna elección.
  3. El modelo final se entrena con todos los lectores con al menos 10 interacciones y todas sus
     interacciones (para la app y Kaggle), con los hiperparámetros elegidos. A los lectores de
     10 a 29 interacciones no se los puede evaluar con este esquema: para ellos el resultado es
     una extrapolación razonable (tienen tanta historia como el train mínimo de la optimización).
- **Métrica: NDCG@20.** Relevante = cualquier libro leído, sin importar el rating (igual que en
  Kaggle: acertar qué va a leer). Con 10 libros de test por lector el NDCG es más bajo que con
  20, porque hay menos aciertos posibles: no se compara con corridas anteriores (ver
  [Historial de experimentos](#historial-de-experimentos)).

### Por qué este esquema

La versión anterior separaba las últimas 20 interacciones como test y aplicaba un mínimo de
interacciones por lector (`min_lector`, hiperparámetro) sólo sobre train. Eso tenía dos
problemas:

- **Se evaluaba con lectores que el modelo no había visto.** Un lector de 21 interacciones tenía
  1 en train y el filtro lo sacaba: en la evaluación lo atendía la popularidad, pero en el
  modelo final (entrenado con todo) sí entraba. Optuna elegía el filtro mirando una situación
  distinta de la de uso, y modelos flojos como SVD elegían un filtro alto para que la
  popularidad atendiera a casi todos.
- **El NDCG era optimista:** los hiperparámetros se elegían con el mismo test con el que se
  informaba el resultado.

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
es la suma de los pesos que le dan los libros que el lector leyó. Hiperparámetro principal: λ.

**Atributos (opcionales):** cada autor, género, editorial y década es un pseudo-lector que
"leyó" todos los libros con ese valor. Eso suma un peso a cada par de libros que comparten el
atributo, como si fueran algo co-leídos. Optuna elige los cuatro pesos (un peso mínimo equivale
a no usarlo). En libros eligió no usarlos.

### dos_torres
Un embedding por lector y otro por libro; puntaje = similitud coseno. Se entrena con softmax
sobre los **negativos del batch**: cada par (lector, libro elegido) usa como negativos los
libros de las demás filas. A cada libro se le resta el log de su frecuencia (**corrección
logQ**, Yi et al. 2019), porque si no, los populares aparecen demasiado como negativos y el
modelo se va a los populares. Elegido = rating ≥ `rating_positivo`. Hiperparámetros:
`rating_positivo`, dimensión, temperatura, learning rate, épocas y `atributos`.

**Atributos (opcionales, los decide Optuna):** el vector del libro es la suma de los embeddings
de su id, autor, género, editorial y década, y el del lector, la de su id, género, década de
nacimiento y España. En libros Optuna eligió no usarlos (0,1201 contra 0,1366 en validación).

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
- **Variables:** el puntaje y la posición que le da cada modelo a cada candidato, y de
  contenido: qué parte de lo que leyó el lector comparte autor, género o editorial con el
  candidato, el año de edición y la popularidad del libro.
- **Modelo:** LightGBM con objetivo LambdaRank reordena los candidatos de cada lector.
- **Sin fuga de datos (stacking):** LightGBM nunca aprende de predicciones de modelos que
  vieron las respuestas.
  - Etapa A (validación): los modelos base se entrenan con train y LightGBM aprende a predecir
    la validación. Optuna elige sus hiperparámetros entrenando con el 80% de los lectores y
    midiendo en el 20% restante.
  - Etapa B (test): los modelos base se entrenan con train + validación y el LightGBM de la
    etapa A se evalúa en test, como los demás modelos.
  - Final: LightGBM se reentrena con la etapa B y se aplica sobre los modelos finales.
- **`promedio`:** los mismos candidatos ordenados por el promedio simple de los puntajes
  (estandarizados por lector), como referencia y como algoritmo propio
  (`prediction_promedio.csv`).

## Cascadas

`cascada_<modelo>` para cada modelo y para el ensamble:

1. Hasta 2 valoraciones (arranque en frío): `top_n`.
2. Si el modelo no conoce al lector (menos de 10 interacciones): `vision_tunel`, que puede
   recomendar cualquier libro.
3. El resto: el modelo.

## Resultados

### Modelos (NDCG@20 en test, 3203 lectores)

Detalle en [`reportes/resumen.md`](reportes/resumen.md) y en el reporte de cada modelo.

| Modelo | NDCG@20 (test) | vs popularidad | NDCG@20 (validación) | min_libro | Libros (final) |
|---|---|---|---|---|---|
| **ensamble** | **0,1074** | **2,13×** | 0,1566 | — | — |
| ease | 0,0981 | 1,95× | 0,1332 | 6 | 10 174 |
| dos_torres | 0,0977 | 1,94× | 0,1366 | 9 | 7 121 |
| *promedio simple* | *0,0976* | *1,94×* | | | |
| gran_torre | 0,0787 | 1,56× | 0,1231 | 22 | 3 180 |
| *popularidad* | *0,0504* | *1,00×* | | | |
| naive | 0,0316 | 0,63× | 0,0465 | 8 | 7 910 |
| svd | 0,0210 | 0,42× | 0,0323 | 49 | 1 453 |

La validación siempre da más que el test: es el optimismo de elegir hiperparámetros mirando
esos mismos datos, que ahora queda separado del resultado informado.

### Aporte de los atributos

| Dónde | Sin atributos | Con atributos | Qué eligió Optuna |
|---|---|---|---|
| EASE (test) | 0,0980 | 0,0981 | Pesos en el mínimo: no usarlos |
| Dos torres (mejor trial en validación) | 0,1366 | 0,1201 | No usarlos |
| Ensamble (test) | 0,0973 | **0,1074** | — (LightGBM usa afinidad con el autor 9%, año de edición 9%, popularidad 8%) |

En libros, los atributos no mejoran a los modelos sueltos (lo que leyó cada lector ya dice más
que el autor o el género), pero sí al ensamble: le permiten reordenar los candidatos de EASE y
dos torres según la afinidad del lector con el autor.

### Cascadas (832 lectores de Kaggle)

| Cascada | top_n (≤ 2 valoraciones) | vision_tunel (3 a 9) | modelo (≥ 10) |
|---|---|---|---|
| todas las cascadas | 43 | 61 | 728 |

(`cascada_svd` manda un lector más a visión de túnel: el filtro de libros le dejó sin datos.)

### Historial de experimentos

Cada fila cambió el esquema de evaluación, así que los NDCG **no son comparables entre filas**
(cambian los lectores evaluados y la cantidad de libros de test); sirven para comparar modelos
dentro de cada fila.

| Esquema | Lectores evaluados | Ensamble | EASE | Dos torres | Popularidad |
|---|---|---|---|---|---|
| Test = últimas 20, `min_lector` ≥ 20 sobre train, hiperparámetros elegidos en test | 3801 | 0,1251 | 0,1211 | 0,1178 | 0,0889 |
| Igual, `min_lector` desde 1 | 3801 | 0,1545 | 0,1457 | 0,1479 | 0,0889 |
| 20 / 10 / 5 (universo / test / validación) | 3890 | 0,1184 | 0,1169 | 0,1130 | 0,0571 |
| 30 / 10 / 10, modelo final ≥ 10 | 3203 | 0,0973 | 0,0980 | 0,0980 | 0,0504 |
| **30 / 10 / 10 + atributos (actual)** | **3203** | **0,1074** | **0,0981** | **0,0977** | **0,0504** |

Medidos sobre los mismos 3203 lectores, los esquemas 20 / 10 / 5 y 30 / 10 / 10 rinden casi
igual (EASE 0,1017 contra 0,0980): la caída entre esas filas se debe a que los lectores de 20 a
29 interacciones son más fáciles de acertar, no al esquema.

## Decisiones tomadas

| Decisión | Por qué |
|---|---|
| Partición cronológica por lector (últimas 10 test, 10 anteriores validación) | Evaluar como en Kaggle (predecir lo próximo que va a leer) sin que ningún modelo vea el futuro. |
| Modelo final con los lectores de 10 o más interacciones | Recomendar con el modelo a todo lector con tanta historia como el train mínimo de la optimización (728 de los 832 lectores de Kaggle, contra 628 con el universo de 30). |
| Universo fijo: lectores con al menos 30 interacciones | Que todo lector con el que se entrena se evalúe y viceversa, siempre con los mismos lectores. Antes el mínimo por lector era un hiperparámetro aplicado sólo a train, y se evaluaba con lectores que el modelo no había visto (ver [Por qué este esquema](#por-qué-este-esquema)). |
| Validación separada del test | Que el NDCG informado no esté inflado por haber elegido los hiperparámetros con el mismo test. |
| El mínimo por libro es un hiperparámetro de cada modelo | Cada modelo tolera distinto los libros con pocas lecturas. |
| Relevante = cualquier libro leído | Es lo que se predice en Kaggle. En una simulación de Kaggle, los dos criterios (cualquier rating o rating ≥ 8) daban resultados parecidos. |
| Dos torres con negativos y logQ, gran torre con negativos | Las versiones de clase aproximaban el rating y recomendaban casi lo mismo a todos (el 86% de los lectores recibía el mismo primer libro con la gran torre). |
| No se agregó complejidad a las redes | Se probó (atributos y capas en dos torres; más capas, dropout, `espania` y antigüedad del libro en gran torre) y no mejoró: Optuna eligió siempre lo más simple. Ver [`reportes/intento_complejidad.md`](reportes/intento_complejidad.md) y [`reportes/literatura.md`](reportes/literatura.md). |
| Curvas de entrenamiento en los reportes | Muestran si una red sigue aprendiendo o empieza a sobreajustar (la gran torre sobreajusta si se entrena de más). |
| Ensamble con LightGBM y variables de contenido | Con sólo los puntajes, LightGBM rendía igual que un promedio simple. Con la afinidad con el autor, el género y la editorial, el año y la popularidad, mejora a EASE un 9% (0,1074 contra 0,0981). |
| Atributos como hiperparámetros en EASE y dos torres | Que Optuna decida si sirven; el primer trial es la mejor configuración sin atributos. En libros no los eligió. |
| Algoritmos SQL con el ranking global en memoria | Las consultas repetían por lector una agregación de toda la tabla. |
| Copia de la base sin SQLite | No bloquear al proceso que está modificando `data/data.db`. |
| KNN se dejó de usar | Predecir libro por libro es muy lento para evaluarlo con Optuna, y no estaba entre los mejores. |
| LightGBM con su API nativa | La API estilo scikit-learn obliga a instalar scikit-learn. |

## Limitaciones conocidas

- Los lectores con menos de 10 interacciones no tienen recomendaciones personalizadas de los
  modelos: los atiende `vision_tunel` en las cascadas.
- Los lectores de 10 a 29 interacciones reciben el modelo, pero no se pueden evaluar con este
  esquema.
- `modelos/ease.pkl` pesa ~400 MB (matriz de 10 mil × 10 mil libros).
- 281 libros no tienen título y no se recomiendan (la app no los puede mostrar). Son libros que
  ya no existen en quelibroleo.com.
- Hay libros con más de una edición, cada una con su propio `id_libro` (por ejemplo
  `el-principito`, `el-principito-1` y `el-principito-2`), así que sus valoraciones quedan
  repartidas. Unificarlas es una tarea pendiente: no alcanza con el sufijo numérico del id
  (muchos son tomos de una serie), habría que agrupar por título y autor y pasar las
  valoraciones de `interacciones` a un id común.
- Los modelos se entrenan con una copia de la base: lo que se valore en la app después no lo
  ven hasta volver a correr `pipeline.py`.
- En `gran_torre` los años (nacimiento, edición) se escalan entre 0 y 1 con los faltantes en 0,
  lo que deforma la escala.
- `insertar_valorado` ignora una nueva valoración si el lector ya había valorado ese libro.
