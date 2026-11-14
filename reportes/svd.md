# svd

Optimizado el 2026-11-11T19:27 con 50 trials de Optuna (TPE), maximizando NDCG@20 en validación.

## Resultado

| Métrica | Valor |
|---|---|
| NDCG@20 en test | **0.0210** |
| NDCG@20 en validación (el que maximizó Optuna) | 0.0323 |
| NDCG@20 de popularidad en test (referencia) | 0.0504 |
| Lectores del universo que el modelo conoce (test) | 99.8% |
| Libros del modelo (en test) | 1249 |
| Tiempo por trial | 3 s |
| Modelo final: interacciones / lectores / libros | 248998 / 5205 / 1453 |

## Hiperparámetros elegidos

| Hiperparámetro | Valor |
|---|---|
| min_libro | 49 |
| n_factors | 137 |
| n_epochs | 50 |
| lr_all | 0.002331 |
| reg_all | 0.006012 |

## Importancia de cada hiperparámetro (fANOVA)

| Hiperparámetro | Importancia |
|---|---|
| lr_all | 0.71 |
| min_libro | 0.14 |
| n_epochs | 0.07 |
| n_factors | 0.04 |
| reg_all | 0.03 |

## Mejores trials

| # | NDCG@20 (validación) | min_libro | n_factors | n_epochs | lr_all | reg_all |
|---|---|---|---|---|---|---|
| 45 | 0.0323 | 49 | 137 | 50 | 0.002331 | 0.006012 |
| 37 | 0.0321 | 36 | 181 | 47 | 0.001341 | 0.006061 |
| 43 | 0.0314 | 41 | 158 | 50 | 0.001356 | 0.01078 |
| 49 | 0.0314 | 41 | 154 | 42 | 0.001402 | 0.005539 |
| 41 | 0.0313 | 38 | 171 | 50 | 0.001344 | 0.007596 |
| 10 | 0.0311 | 6 | 149 | 44 | 0.001313 | 0.005177 |
| 39 | 0.0310 | 40 | 186 | 49 | 0.001762 | 0.005203 |
| 44 | 0.0306 | 41 | 180 | 49 | 0.001432 | 0.009878 |
| 30 | 0.0302 | 19 | 173 | 45 | 0.001004 | 0.005131 |
| 42 | 0.0299 | 40 | 200 | 46 | 0.002237 | 0.005833 |
