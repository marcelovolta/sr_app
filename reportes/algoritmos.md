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
| Ensamble | `ensamble` | LightGBM combina las recomendaciones de cuatro modelos. |
| Cascadas | `cascada_<modelo>` (una por modelo y una por el ensamble) | Eligen el algoritmo según cuánto se conoce al lector. |

## Cómo se comparan

Los modelos entrenados y el ensamble se miden con el mismo protocolo:

- **Datos:** 461 075 valoraciones de 10 667 lectores sobre 48 063 libros.
- **Partición cronológica:** las últimas 20 valoraciones de cada lector son el test y las
  anteriores, el train. Se evalúan los 3801 lectores con más de 20 valoraciones.
- **Métrica: NDCG@20.** Mide si los libros que el lector efectivamente leyó después aparecen
  entre los 20 recomendados y qué tan arriba están (1 = perfecto, 0 = ningún acierto). Cuenta
  como acierto cualquier libro leído, como en Kaggle.
- **Mismos lectores para todos:** a quien el modelo no conoce se le recomiendan los más leídos.
  Así un modelo que cubre a pocos lectores no sale beneficiado.
- **Referencia:** recomendar a todos los más leídos da NDCG@20 = 0,0889.

Los hiperparámetros se eligen con el mismo test con que se mide, así que los valores son algo
optimistas. Sirven para comparar los modelos entre sí, no como estimación absoluta.

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
libro es la suma de los pesos que le dan los libros que el lector leyó. Su único
hiperparámetro es la regularización λ.

### dos_torres — red neuronal de recuperación

Una "torre" convierte al lector en un vector y otra convierte al libro; el puntaje es la
similitud entre los dos. Se entrena para que, en cada lote, el libro que el lector eligió
quede por encima de los libros elegidos por los demás lectores del lote (**negativos del
lote**). A cada libro se le descuenta su popularidad (**corrección logQ**, Yi et al. 2019);
si no, el modelo termina recomendando solo los populares.

### gran_torre — red neuronal sobre pares

Una sola red recibe el par (lector, libro) junto con sus rasgos (autor, género, editorial y
año de edición del libro; género y año de nacimiento del lector) y estima la probabilidad de
que el lector lo elija. Tiene 3 capas de 256 neuronas y aprende a separar los libros leídos de
libros al azar no leídos, que se vuelven a sortear en cada época.

## Ensamble — LightGBM

1. **Candidatos:** los 50 primeros libros de SVD, EASE, dos torres y gran torre, unidos.
2. **Variables:** el puntaje y la posición que le da cada modelo a cada candidato.
3. **LightGBM (LambdaRank)** aprende a reordenar los candidatos de cada lector.

Para que no haya fuga de datos, LightGBM aprende de predicciones de modelos que no vieron las
respuestas: los modelos base se entrenan sin las últimas 20 valoraciones de train y LightGBM
aprende a predecir esas 20.

## Cascadas

| Situación del lector | Algoritmo |
|---|---|
| Hasta 2 valoraciones (arranque en frío) | `top_n` |
| El modelo no lo conoce (quedó afuera del filtro de entrenamiento) | `vision_tunel` |
| El resto | el modelo (o el ensamble) |

---

## Rendimiento comparativo

### NDCG@20 en test

| Puesto | Modelo | NDCG@20 | vs popularidad | Lectores del test cubiertos |
|---|---|---|---|---|
| 1 | **ensamble** | **0,1251** | **1,41×** | 69% |
| 2 | ease | 0,1211 | 1,36× | 69% |
| 3 | dos_torres | 0,1178 | 1,33× | 64% |
| 4 | gran_torre | 0,1070 | 1,20× | 64% |
| — | *popularidad (referencia)* | *0,0889* | *1,00×* | |
| 5 | svd | 0,0880 | 0,99× | 13% |
| 6 | naive | 0,0878 | 0,99× | 14% |

```
ensamble    ██████████████████████████████████████████  0,1251
ease        ████████████████████████████████████████    0,1211
dos_torres  ███████████████████████████████████████     0,1178
gran_torre  ███████████████████████████████████         0,1070
popularidad █████████████████████████████               0,0889
svd         █████████████████████████████               0,0880
naive       █████████████████████████████               0,0878
```

### Qué eligió Optuna y cuánto cuesta cada modelo

| Modelo | Filtro elegido (mín. por lector / por libro) | Libros en el modelo final | Tiempo por trial | Hiperparámetros clave |
|---|---|---|---|---|
| naive | 80 / 47 | 776 | < 1 s | reg. libro 140,7; 9 épocas |
| svd | 80 / 50 | 688 | < 1 s | 11 factores; 5 épocas |
| ease | 20 / 5 | 11 453 | 16 s | λ = 298,8 |
| dos_torres | 21 / 18 | 3 645 | 6 s | dimensión 256; temperatura 0,15; 6 épocas |
| gran_torre | 20 / 23 | 2 880 | 35 s | 6 negativos; 6 épocas |
| ensamble | — | — | — | 9 hojas; 282 árboles |

### Aporte de cada modelo al ensamble

Importancia de las variables en LightGBM (ganancia, sumando puntaje y posición):

| Modelo base | Importancia |
|---|---|
| ease | 52% |
| dos_torres | 31% |
| gran_torre | 12% |
| svd | 5% |

## Interpretación

1. **Predecir qué se va a leer supera a predecir el rating.** EASE, dos torres y gran torre
   aprenden "leyó o no leyó" y superan a la popularidad entre un 20% y un 36%. SVD y naive
   aprenden el valor del rating y no le ganan a la popularidad: saber que alguien le pondría un
   9 a un libro no dice si lo va a leer.
2. **La cobertura pesa.** En todos los modelos, el hiperparámetro más importante (fANOVA) fue el
   mínimo de valoraciones por lector. Los mejores modelos eligieron el filtro más permisivo
   (alrededor de 20) para conocer a más lectores. SVD y naive eligieron el más exigente: como
   no mejoran a la popularidad, les conviene que se use lo menos posible, y cubren solo al
   13-14% de los lectores.
3. **El modelo más simple es el mejor modelo individual.** EASE, con un solo hiperparámetro y
   solución cerrada, supera a las dos redes neuronales. También se probó agregar complejidad a
   las redes (atributos, más capas, dropout) y no mejoraron; ver
   [intento_complejidad.md](intento_complejidad.md).
4. **Combinar ayuda, pero poco.** El ensamble mejora a EASE en 0,004 (+3%). Un promedio simple
   de los puntajes estandarizados logra lo mismo (0,1252), así que la mejora viene de combinar
   los modelos y no de la sofisticación de LightGBM. El ensamble se apoya sobre todo en EASE y
   dos torres.
5. **El ensamble no amplía la cobertura.** Cubre a los mismos lectores que EASE (69%), porque
   EASE es el modelo con el filtro más amplio. En la cascada, los dos reparten igual a los 832
   lectores de Kaggle:

| Cascada | top_n | visión de túnel | modelo |
|---|---|---|---|
| cascada_ensamble | 43 | 131 | 658 |
| cascada_ease | 43 | 131 | 658 |
| cascada_dos_torres | 43 | 158 | 631 |
| cascada_gran_torre | 43 | 161 | 628 |
| cascada_svd | 43 | 592 | 197 |
| cascada_naive | 43 | 581 | 208 |

## Conclusión

- **Mejor opción: `cascada_ensamble`**, con el mejor NDCG@20 (0,1251).
- **Alternativa simple: `cascada_ease`.** Pierde solo un 3% y no necesita entrenar cinco
  modelos ni LightGBM. Su costo es el tamaño del modelo (~530 MB).
- **SVD y naive** no aportan frente a recomendar lo más leído.
