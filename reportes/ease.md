# ease

Optimizado el 2026-11-11T19:39 con 60 trials de Optuna (TPE), maximizando NDCG@20 en validación.

## Resultado

| Métrica | Valor |
|---|---|
| NDCG@20 en test | **0.0981** |
| NDCG@20 en validación (el que maximizó Optuna) | 0.1332 |
| NDCG@20 de popularidad en test (referencia) | 0.0504 |
| Lectores del universo que el modelo conoce (test) | 99.9% |
| Libros del modelo (en test) | 9087 |
| Tiempo por trial | 13 s |
| Modelo final: interacciones / lectores / libros | 377321 / 5228 / 10174 |

## Hiperparámetros elegidos

| Hiperparámetro | Valor |
|---|---|
| min_libro | 6 |
| lambda | 756.8 |
| peso_autor | 0.001302 |
| peso_genero | 0.0012 |
| peso_editorial | 0.001762 |
| peso_decada | 0.001147 |

## Importancia de cada hiperparámetro (fANOVA)

| Hiperparámetro | Importancia |
|---|---|
| lambda | 0.83 |
| peso_genero | 0.07 |
| peso_editorial | 0.06 |
| peso_autor | 0.02 |
| min_libro | 0.02 |
| peso_decada | 0.00 |

## Mejores trials

| # | NDCG@20 (validación) | min_libro | lambda | peso_autor | peso_genero | peso_editorial | peso_decada |
|---|---|---|---|---|---|---|---|
| 58 | 0.1332 | 6 | 756.8 | 0.001302 | 0.0012 | 0.001762 | 0.001147 |
| 0 | 0.1332 | 5 | 771.5 | 0.001 | 0.001 | 0.001 | 0.001 |
| 37 | 0.1331 | 6 | 761.8 | 0.001819 | 0.002091 | 0.001287 | 0.009554 |
| 59 | 0.1330 | 7 | 798.5 | 0.001098 | 0.002581 | 0.00182 | 0.002053 |
| 50 | 0.1327 | 6 | 1024 | 0.007752 | 0.001623 | 0.005811 | 0.007592 |
| 47 | 0.1326 | 6 | 662.5 | 0.0042 | 0.003546 | 0.001585 | 0.002677 |
| 42 | 0.1326 | 5 | 608.8 | 0.001438 | 0.001144 | 0.002389 | 0.001672 |
| 23 | 0.1325 | 5 | 618.2 | 0.02202 | 0.01143 | 0.007648 | 0.001089 |
| 53 | 0.1325 | 5 | 585.1 | 0.003042 | 0.006178 | 0.007653 | 0.00955 |
| 57 | 0.1325 | 8 | 630.4 | 0.00224 | 0.001714 | 0.001867 | 0.004299 |
