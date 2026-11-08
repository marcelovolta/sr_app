# naive

Optimizado el 2026-11-08T12:14 con 50 trials de Optuna (TPE), maximizando NDCG@20.

## Resultado

| Métrica | Valor |
|---|---|
| NDCG@20 (test) | **0.0878** |
| NDCG@20 de popularidad (referencia) | 0.0889 |
| Lectores del test cubiertos por el modelo | 13.8% |
| Libros del modelo (en la evaluación) | 632 |
| Tiempo por trial | 0 s |
| Modelo final: interacciones / lectores / libros | 91757 / 699 / 776 |

## Hiperparámetros elegidos

| Hiperparámetro | Valor |
|---|---|
| min_lector | 80 |
| min_libro | 47 |
| reg_i | 140.7 |
| reg_u | 3.467 |
| n_epochs | 9 |

## Importancia de cada hiperparámetro (fANOVA)

| Hiperparámetro | Importancia |
|---|---|
| min_lector | 0.41 |
| reg_i | 0.31 |
| n_epochs | 0.23 |
| min_libro | 0.04 |
| reg_u | 0.01 |

## Mejores trials

| # | NDCG@20 | min_lector | min_libro | reg_i | reg_u | n_epochs |
|---|---|---|---|---|---|---|
| 49 | 0.0878 | 80 | 47 | 140.7 | 3.467 | 9 |
| 46 | 0.0875 | 79 | 40 | 194.6 | 0.8346 | 3 |
| 48 | 0.0875 | 75 | 39 | 191 | 0.2922 | 3 |
| 47 | 0.0874 | 77 | 49 | 124.9 | 2.303 | 4 |
| 30 | 0.0874 | 80 | 40 | 131.7 | 0.342 | 3 |
| 31 | 0.0871 | 80 | 39 | 118.5 | 0.7938 | 3 |
| 43 | 0.0870 | 80 | 49 | 24.55 | 1.022 | 5 |
| 45 | 0.0870 | 79 | 38 | 107.2 | 0.9751 | 6 |
| 29 | 0.0869 | 71 | 48 | 103.1 | 0.5034 | 3 |
| 41 | 0.0868 | 77 | 29 | 190.5 | 3.67 | 5 |
