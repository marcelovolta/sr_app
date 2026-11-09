# dos_torres

Optimizado el 2026-11-09T15:34 con 49 trials de Optuna (TPE), maximizando NDCG@20 en validación.

## Resultado

| Métrica | Valor |
|---|---|
| NDCG@20 en test | **0.0977** |
| NDCG@20 en validación (el que maximizó Optuna) | 0.1366 |
| NDCG@20 de popularidad en test (referencia) | 0.0504 |
| Lectores del universo que el modelo conoce (test) | 99.9% |
| Libros del modelo (en test) | 6347 |
| Tiempo por trial | 28 s |
| Modelo final: interacciones / lectores / libros | 356420 / 5224 / 7121 |

## Hiperparámetros elegidos

| Hiperparámetro | Valor |
|---|---|
| min_libro | 9 |
| rating_positivo | 3 |
| dim | 256 |
| temperatura | 0.1057 |
| learning_rate | 0.002632 |
| epochs | 17 |
| atributos | False |

## Importancia de cada hiperparámetro (fANOVA)

| Hiperparámetro | Importancia |
|---|---|
| temperatura | 0.66 |
| min_libro | 0.16 |
| epochs | 0.11 |
| learning_rate | 0.03 |
| rating_positivo | 0.02 |
| dim | 0.02 |
| atributos | 0.00 |

## Mejores trials

| # | NDCG@20 (validación) | min_libro | rating_positivo | dim | temperatura | learning_rate | epochs | atributos |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.1366 | 9 | 3 | 256 | 0.1057 | 0.002632 | 17 | False |
| 43 | 0.1358 | 8 | 5 | 256 | 0.1187 | 0.006955 | 19 | False |
| 0 | 0.1356 | 8 | 5 | 256 | 0.1046 | 0.00968 | 17 | False |
| 35 | 0.1356 | 8 | 5 | 256 | 0.1176 | 0.003643 | 19 | False |
| 37 | 0.1353 | 10 | 6 | 256 | 0.1101 | 0.001616 | 17 | False |
| 41 | 0.1351 | 7 | 4 | 256 | 0.123 | 0.003282 | 19 | False |
| 34 | 0.1351 | 10 | 4 | 256 | 0.1039 | 0.001921 | 20 | False |
| 39 | 0.1350 | 11 | 5 | 256 | 0.1208 | 0.002836 | 15 | False |
| 46 | 0.1344 | 8 | 7 | 256 | 0.1087 | 0.002439 | 18 | False |
| 38 | 0.1337 | 9 | 6 | 256 | 0.1393 | 0.001312 | 18 | False |

## Curva de entrenamiento

Con los hiperparámetros elegidos, entrenando sobre train más épocas que las elegidas (17, línea punteada).

![Curva de entrenamiento de dos_torres](curva_dos_torres.png)

| Época | Pérdida | NDCG@20 (validación) |
|---|---|---|
| 1 | 7.0945 | 0.1163 |
| 2 | 5.8549 | 0.1259 |
| 3 | 5.6550 | 0.1301 |
| 4 | 5.5643 | 0.1318 |
| 5 | 5.5116 | 0.1319 |
| 6 | 5.4754 | 0.1329 |
| 7 | 5.4494 | 0.1346 |
| 8 | 5.4291 | 0.1350 |
| 9 | 5.4133 | 0.1332 |
| 10 | 5.4000 | 0.1352 |
| 11 | 5.3891 | 0.1346 |
| 12 | 5.3794 | 0.1356 |
| 13 | 5.3714 | 0.1347 |
| 14 | 5.3637 | 0.1347 |
| 15 | 5.3568 | 0.1350 |
| 16 | 5.3511 | 0.1348 |
| 17 | 5.3454 | 0.1366 |
| 18 | 5.3409 | 0.1352 |
| 19 | 5.3359 | 0.1354 |
| 20 | 5.3318 | 0.1361 |
| 21 | 5.3275 | 0.1354 |
| 22 | 5.3237 | 0.1363 |
| 23 | 5.3198 | 0.1358 |
| 24 | 5.3164 | 0.1361 |
| 25 | 5.3130 | 0.1362 |
| 26 | 5.3099 | 0.1367 |
| 27 | 5.3070 | 0.1357 |
| 28 | 5.3044 | 0.1348 |
| 29 | 5.3012 | 0.1358 |
| 30 | 5.2985 | 0.1361 |
| 31 | 5.2961 | 0.1358 |
| 32 | 5.2934 | 0.1361 |
| 33 | 5.2910 | 0.1357 |
| 34 | 5.2890 | 0.1366 |
