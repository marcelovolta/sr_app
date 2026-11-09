# Algoritmos de recomendación y su rendimiento

Este documento resume los algoritmos del recomendador de libros, cómo funciona cada uno
y cómo rinden comparados entre sí. El detalle de cada modelo está en su reporte
([naive](naive.md), [svd](svd.md), [ease](ease.md), [dos_torres](dos_torres.md),
[gran_torre](gran_torre.md), [ensamble](ensamble.md)) y el proceso completo, en el
[README](../README.md).

## Panorama

Hay cuatro familias de algoritmos:

| Familia | Algoritmos | Idea |
|---|---|---|
| Reglas SQL | `random`, `top_n`, `mas_valorados`, `perfil_balanceado`, `vision_tunel` | Consultas fijas sobre la base, sin entrenamiento. |
| Modelos entrenados | `naive`, `svd`, `ease`, `dos_torres`, `gran_torre` | Aprenden de las valoraciones; hiperparámetros ajustados con Optuna. |
| Ensamble | `ensamble`, `promedio` | Combinan las recomendaciones de cuatro modelos (LightGBM o promedio simple). |
| Cascadas | `cascada_<modelo>` (una por modelo y una por el ensamble) | Eligen el algoritmo según cuánto se conoce al lector. |

## Cómo se comparan

Los modelos entrenados y el ensamble se miden con el mismo protocolo:

- **Universo de evaluación:** los 3203 lectores con al menos 30 valoraciones. Es fijo: todos los
  modelos se entrenan y se evalúan con los mismos lectores.
- **Partición cronológica por lector:** las últimas 10 valoraciones son el test, las 10
  anteriores la validación y el resto (al menos 10), el train.
- **Tres etapas:** Optuna elige los hiperparámetros entrenando con train y midiendo en
  validación; con los mejores se reentrena con train + validación y se mide en test (el número
  que se informa, que no participó de ninguna elección); el modelo final se entrena con todos
  los lectores con al menos 10 valoraciones, para recomendar también a los que tienen menos
  historia que el universo de evaluación.
- **Métrica: NDCG@20.** Mide si los libros que el lector efectivamente leyó después aparecen
  entre los 20 recomendados y qué tan arriba están (1 = perfecto, 0 = ningún acierto). Cuenta
  como acierto cualquier libro leído, como en Kaggle.
- **Referencia:** recomendar a todos los más leídos da NDCG@20 = 0,0504 en test.

Estos NDCG no se comparan con los de versiones anteriores del proyecto (cambiaron los lectores
evaluados y la cantidad de libros de test): sólo entre sí y contra la popularidad. El historial
está en el [README](../README.md#historial-de-experimentos).

Los algoritmos SQL no se evalúan con este protocolo, porque consultan la base completa, que
incluye el test. Su comparación es el puntaje de Kaggle.

---

## Reglas SQL

No se entrenan. Todas excluyen lo que el lector ya valoró o vio.

| Algoritmo | Qué recomienda | Personalizado |
|---|---|---|
| `random` | Libros al azar. Sirve como piso. | No |
| `top_n` | Mejor promedio de rating, con al menos 5 valoraciones para que un único 10 no domine. | No |
| `mas_valorados` | Los libros con más valoraciones (popularidad). | No |
| `perfil_balanceado` | 0,4 × afinidad con el género + 0,4 × afinidad con el autor + 0,2 × rating del libro. La afinidad sale de los géneros y autores que el lector valoró al menos 3 veces. | Sí |
| `vision_tunel` | Primero, libros que coinciden en género **y** autor con lo que leyó el lector; si no alcanzan, los populares. | Sí |

`top_n` y `vision_tunel` se usan dentro de las cascadas: el primero para lectores nuevos y el
segundo para lectores que los modelos no conocen, porque puede recomendar cualquier libro de
la base.

## Modelos entrenados

### naive — sesgos (Surprise, BaselineOnly)

Estima el rating como *media general + sesgo del lector + sesgo del libro*. Para ordenar los
libros de un lector solo importa el sesgo del libro, así que **recomienda lo mismo a todos**:
es un "top N" aprendido.

### svd — factorización de matrices (Surprise)

Representa a cada lector y a cada libro con un vector de factores latentes (gustos ocultos), y
el rating estimado es su producto. Aprende a **predecir el rating**, no a predecir qué va a
leer el lector.

### ease — modelo lineal libro a libro (Steck, 2019)

Aprende cuánto "empuja" cada libro a cada otro: si leíste A, ¿cuánto sube la probabilidad de
B? Tiene **solución cerrada**, `B = I − P / diag(P)` con `P = (XᵀX + λI)⁻¹`, sin épocas ni
descenso por gradiente. X solo registra si el lector leyó el libro o no. El puntaje de un
libro es la suma de los pesos que le dan los libros que el lector leyó. Su hiperparámetro
principal es la regularización λ.

Opcionalmente usa **atributos**: cada autor, género, editorial y década actúa como un
pseudo-lector que "leyó" todos los libros con ese valor, así que dos libros del mismo autor
suman co-lectura. Optuna elige el peso de cada atributo; en libros los dejó en el mínimo.

### dos_torres — red neuronal de recuperación

Una "torre" convierte al lector en un vector y otra convierte al libro; el puntaje es la
similitud entre los dos. Se entrena para que, en cada lote, el libro que el lector eligió
quede por encima de los libros elegidos por los demás lectores del lote (**negativos del
lote**). A cada libro se le descuenta su popularidad (**corrección logQ**, Yi et al. 2019);
si no, el modelo termina recomendando solo los populares.

Opcionalmente cada torre suma los embeddings de sus **atributos** (libro: autor, género,
editorial, década; lector: género, década de nacimiento, España). Optuna eligió no usarlos.

### gran_torre — red neuronal sobre pares

Una sola red recibe el par (lector, libro) junto con sus rasgos (autor, género, editorial y
año de edición del libro; género y año de nacimiento del lector) y estima la probabilidad de
que el lector lo elija. Tiene 3 capas de 256 neuronas y aprende a separar los libros leídos de
libros al azar no leídos, que se vuelven a sortear en cada época.

## Ensamble — LightGBM

1. **Candidatos:** los 50 primeros libros de SVD, EASE, dos torres y gran torre, unidos.
2. **Variables:** el puntaje y la posición que le da cada modelo a cada candidato, y de
   contenido: qué parte de lo que leyó el lector comparte autor, género o editorial con el
   candidato, el año de edición y la popularidad.
3. **LightGBM (LambdaRank)** aprende a reordenar los candidatos de cada lector.

Para que no haya fuga de datos, LightGBM aprende de predicciones de modelos que no vieron las
respuestas: los modelos base se entrenan con train y LightGBM aprende a predecir la validación;
se evalúa en test con modelos base entrenados con train + validación.

**`promedio`** ordena los mismos candidatos por el promedio simple de los puntajes de los
cuatro modelos (estandarizados por lector), sin LightGBM.

## Cascadas

| Situación del lector | Algoritmo |
|---|---|
| Hasta 2 valoraciones (arranque en frío) | `top_n` |
| El modelo no lo conoce (menos de 10 valoraciones) | `vision_tunel` |
| El resto | el modelo (o el ensamble) |

---

## Rendimiento comparativo

### NDCG@20

| Puesto | Modelo | NDCG@20 (test) | vs popularidad | NDCG@20 (validación) |
|---|---|---|---|---|
| 1 | **ensamble** | **0,1074** | **2,13×** | 0,1566 |
| 2 | ease | 0,0981 | 1,95× | 0,1332 |
| 3 | dos_torres | 0,0977 | 1,94× | 0,1366 |
| — | *promedio simple* | *0,0976* | *1,94×* | |
| 4 | gran_torre | 0,0787 | 1,56× | 0,1231 |
| — | *popularidad (referencia)* | *0,0504* | *1,00×* | |
| 5 | naive | 0,0316 | 0,63× | 0,0465 |
| 6 | svd | 0,0210 | 0,42× | 0,0323 |

```
ensamble    ██████████████████████████████████████████  0,1074
ease        ██████████████████████████████████████      0,0981
dos_torres  ██████████████████████████████████████      0,0977
gran_torre  ███████████████████████████████             0,0787
popularidad ████████████████████                        0,0504
naive       ████████████                                0,0316
svd         ████████                                    0,0210
```

La validación da siempre más que el test: es el optimismo de elegir los hiperparámetros
mirando esos datos. Por eso el resultado que se informa es el de test.

### Qué eligió Optuna y cuánto cuesta cada modelo

| Modelo | min_libro | Libros en el modelo final | Tiempo por trial | Hiperparámetros clave | Atributos |
|---|---|---|---|---|---|
| naive | 8 | 7 910 | 3 s | reg. libro 191; 20 épocas | — |
| svd | 49 | 1 453 | 5 s | 137 factores; 50 épocas | — |
| ease | 6 | 10 174 | 11 s | λ = 757 | no (pesos en el mínimo) |
| dos_torres | 9 | 7 121 | 28 s | dimensión 256; temperatura 0,11; 17 épocas; positivo = rating ≥ 3 | no |
| gran_torre | 22 | 3 180 | 47 s | 5 negativos; 7 épocas; positivo = rating ≥ 5 | siempre (autor, género, editorial, año) |
| ensamble | — | — | — | 35 hojas; 487 árboles | variables de contenido |

### Aporte de los atributos

| Dónde | Sin atributos | Con atributos |
|---|---|---|
| EASE (test) | 0,0980 | 0,0981 |
| Dos torres (mejor trial en validación) | 0,1366 | 0,1201 |
| Ensamble (test) | 0,0973 | **0,1074** |

### Qué usa el ensamble

Importancia de las variables en LightGBM (ganancia, sumando puntaje y posición de cada modelo):

| Variables | Importancia |
|---|---|
| ease | 29% |
| dos_torres | 16% |
| svd | 11% |
| gran_torre | 10% |
| **contenido** (afinidad con el autor 9%, año de edición 9%, popularidad 8%, afinidad con el género 5%, con la editorial 4%) | **35%** |

## Interpretación

1. **Predecir qué se va a leer supera a predecir el rating.** EASE, dos torres y gran torre
   aprenden "leyó o no leyó" y duplican a la popularidad. SVD y naive aprenden el valor del
   rating y quedan **por debajo** de la popularidad: saber que alguien le pondría un 9 a un
   libro no dice si lo va a leer.
2. **El modelo más simple es el mejor modelo individual.** EASE, con solución cerrada, empata
   con dos torres y supera a la gran torre. Agregar complejidad a las redes no las mejoró; ver
   [intento_complejidad.md](intento_complejidad.md).
3. **Los atributos no ayudan a los modelos sueltos, pero sí al ensamble.** Lo que leyó cada
   lector ya dice más que el autor o el género del libro. En cambio, para reordenar los
   candidatos que proponen los modelos, la afinidad del lector con el autor sí suma: el
   ensamble mejora a EASE un 9% (0,1074 contra 0,0981), y un tercio de lo que usa LightGBM son
   variables de contenido.
4. **Sin variables de contenido, combinar no ayudaba:** LightGBM rendía igual que el promedio
   simple (0,0973 contra 0,0981).
5. **Las cascadas reparten igual.** Todos los modelos conocen a los mismos lectores (los de 10
   o más valoraciones). En los 832 lectores de Kaggle:

| Cascada | top_n (≤ 2 valoraciones) | visión de túnel (3 a 9) | modelo (≥ 10) |
|---|---|---|---|
| todas (`cascada_svd`: 62 / 727) | 43 | 61 | 728 |

## Conclusión

- **Mejor opción: `cascada_ensamble`**, con el mejor NDCG@20 en test (0,1074), un 9% sobre EASE.
- **Alternativa simple: `cascada_ease`.** No necesita entrenar cinco modelos ni LightGBM. Su
  costo es el tamaño del modelo (~400 MB).
- **SVD y naive** rinden peor que recomendar lo más leído: no conviene usarlos.
