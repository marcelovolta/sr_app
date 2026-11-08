# svd

Optimizado el 2026-11-08T12:15 con 50 trials de Optuna (TPE), maximizando NDCG@20.

## Resultado

| Métrica | Valor |
|---|---|
| NDCG@20 (test) | **0.0880** |
| NDCG@20 de popularidad (referencia) | 0.0889 |
| Lectores del test cubiertos por el modelo | 12.8% |
| Libros del modelo (en la evaluación) | 556 |
| Tiempo por trial | 0 s |
| Modelo final: interacciones / lectores / libros | 83964 / 653 / 688 |

## Hiperparámetros elegidos

| Hiperparámetro | Valor |
|---|---|
| min_lector | 80 |
| min_libro | 50 |
| n_factors | 11 |
| n_epochs | 5 |
| lr_all | 0.001647 |
| reg_all | 0.01032 |

## Importancia de cada hiperparámetro (fANOVA)

| Hiperparámetro | Importancia |
|---|---|
| min_lector | 0.39 |
| n_epochs | 0.33 |
| min_libro | 0.14 |
| lr_all | 0.11 |
| reg_all | 0.03 |
| n_factors | 0.00 |

## Mejores trials

| # | NDCG@20 | min_lector | min_libro | n_factors | n_epochs | lr_all | reg_all |
|---|---|---|---|---|---|---|---|
| 36 | 0.0880 | 80 | 50 | 11 | 5 | 0.001647 | 0.01032 |
| 41 | 0.0876 | 79 | 48 | 12 | 8 | 0.001012 | 0.01008 |
| 45 | 0.0872 | 74 | 45 | 27 | 5 | 0.001396 | 0.00601 |
| 44 | 0.0871 | 80 | 34 | 25 | 6 | 0.001132 | 0.005437 |
| 39 | 0.0869 | 79 | 42 | 21 | 7 | 0.001555 | 0.01225 |
| 37 | 0.0865 | 76 | 47 | 21 | 6 | 0.002541 | 0.009704 |
| 43 | 0.0865 | 77 | 49 | 17 | 7 | 0.002251 | 0.01326 |
| 49 | 0.0864 | 78 | 28 | 20 | 7 | 0.001042 | 0.008465 |
| 46 | 0.0862 | 79 | 34 | 16 | 11 | 0.0013 | 0.005868 |
| 38 | 0.0862 | 73 | 40 | 20 | 5 | 0.001808 | 0.01516 |
