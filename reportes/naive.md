# naive

Optimizado el 2026-11-09T15:00 con 50 trials de Optuna (TPE), maximizando NDCG@20 en validación.

## Resultado

| Métrica | Valor |
|---|---|
| NDCG@20 en test | **0.0316** |
| NDCG@20 en validación (el que maximizó Optuna) | 0.0465 |
| NDCG@20 de popularidad en test (referencia) | 0.0504 |
| Lectores del universo que el modelo conoce (test) | 99.9% |
| Libros del modelo (en test) | 7068 |
| Tiempo por trial | 3 s |
| Modelo final: interacciones / lectores / libros | 362732 / 5224 / 7910 |

## Hiperparámetros elegidos

| Hiperparámetro | Valor |
|---|---|
| min_libro | 8 |
| reg_i | 191.3 |
| reg_u | 1.388 |
| n_epochs | 20 |

## Importancia de cada hiperparámetro (fANOVA)

| Hiperparámetro | Importancia |
|---|---|
| reg_i | 0.64 |
| reg_u | 0.24 |
| n_epochs | 0.10 |
| min_libro | 0.02 |

## Mejores trials

| # | NDCG@20 (validación) | min_libro | reg_i | reg_u | n_epochs |
|---|---|---|---|---|---|
| 32 | 0.0465 | 8 | 191.3 | 1.388 | 20 |
| 16 | 0.0465 | 10 | 192.1 | 3.617 | 22 |
| 44 | 0.0465 | 16 | 196.5 | 1.377 | 21 |
| 34 | 0.0465 | 7 | 183.5 | 1.687 | 23 |
| 38 | 0.0464 | 7 | 177.9 | 2.207 | 24 |
| 33 | 0.0464 | 11 | 183.8 | 8.725 | 19 |
| 22 | 0.0461 | 10 | 176 | 3.584 | 21 |
| 48 | 0.0452 | 15 | 164.5 | 1.602 | 19 |
| 43 | 0.0447 | 12 | 161.1 | 11.27 | 21 |
| 42 | 0.0447 | 6 | 150.7 | 1.267 | 26 |
