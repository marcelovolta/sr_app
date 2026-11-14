# gran_torre

Optimizado el 2026-11-11T20:16 con 30 trials de Optuna (TPE), maximizando NDCG@20 en validación.

## Resultado

| Métrica | Valor |
|---|---|
| NDCG@20 en test | **0.0787** |
| NDCG@20 en validación (el que maximizó Optuna) | 0.1231 |
| NDCG@20 de popularidad en test (referencia) | 0.0504 |
| Lectores del universo que el modelo conoce (test) | 99.9% |
| Libros del modelo (en test) | 2776 |
| Tiempo por trial | 35 s |
| Modelo final: interacciones / lectores / libros | 303794 / 5218 / 3180 |

## Hiperparámetros elegidos

| Hiperparámetro | Valor |
|---|---|
| min_libro | 22 |
| rating_positivo | 5 |
| negativos | 5 |
| learning_rate | 0.002484 |
| epochs | 7 |

## Importancia de cada hiperparámetro (fANOVA)

| Hiperparámetro | Importancia |
|---|---|
| negativos | 0.76 |
| learning_rate | 0.09 |
| min_libro | 0.07 |
| epochs | 0.07 |
| rating_positivo | 0.01 |

## Mejores trials

| # | NDCG@20 (validación) | min_libro | rating_positivo | negativos | learning_rate | epochs |
|---|---|---|---|---|---|---|
| 20 | 0.1231 | 22 | 5 | 5 | 0.002484 | 7 |
| 27 | 0.1208 | 40 | 2 | 5 | 0.004655 | 5 |
| 24 | 0.1194 | 33 | 3 | 5 | 0.002261 | 7 |
| 23 | 0.1190 | 38 | 3 | 4 | 0.003862 | 6 |
| 15 | 0.1183 | 31 | 4 | 3 | 0.004881 | 8 |
| 26 | 0.1181 | 25 | 3 | 5 | 0.001364 | 9 |
| 13 | 0.1176 | 47 | 5 | 3 | 0.008804 | 6 |
| 7 | 0.1175 | 32 | 3 | 2 | 0.002336 | 5 |
| 16 | 0.1174 | 32 | 6 | 5 | 0.005809 | 8 |
| 4 | 0.1171 | 20 | 2 | 4 | 0.0005404 | 6 |

## Curva de entrenamiento

Con los hiperparámetros elegidos, entrenando sobre train más épocas que las elegidas (7, línea punteada).

![Curva de entrenamiento de gran_torre](curva_gran_torre.png)

| Época | Pérdida | NDCG@20 (validación) |
|---|---|---|
| 1 | 0.3560 | 0.0984 |
| 2 | 0.2985 | 0.1107 |
| 3 | 0.2765 | 0.1124 |
| 4 | 0.2629 | 0.1169 |
| 5 | 0.2516 | 0.1201 |
| 6 | 0.2416 | 0.1168 |
| 7 | 0.2329 | 0.1231 |
| 8 | 0.2248 | 0.1221 |
| 9 | 0.2165 | 0.1194 |
| 10 | 0.2080 | 0.1136 |
| 11 | 0.2013 | 0.1170 |
| 12 | 0.1936 | 0.1113 |
| 13 | 0.1865 | 0.1157 |
| 14 | 0.1796 | 0.1116 |
| 15 | 0.1736 | 0.1172 |
| 16 | 0.1672 | 0.1115 |
| 17 | 0.1619 | 0.1135 |
| 18 | 0.1569 | 0.1084 |
| 19 | 0.1524 | 0.1107 |
| 20 | 0.1477 | 0.1097 |
