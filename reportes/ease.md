# ease

Optimizado el 2026-11-08T12:20 con 50 trials de Optuna (TPE), maximizando NDCG@20.

## Resultado

| Métrica | Valor |
|---|---|
| NDCG@20 (test) | **0.1211** |
| NDCG@20 de popularidad (referencia) | 0.0889 |
| Lectores del test cubiertos por el modelo | 69.2% |
| Libros del modelo (en la evaluación) | 9788 |
| Tiempo por trial | 16 s |
| Modelo final: interacciones / lectores / libros | 364589 / 3704 / 11453 |

## Hiperparámetros elegidos

| Hiperparámetro | Valor |
|---|---|
| min_lector | 20 |
| min_libro | 5 |
| lambda | 298.8 |

## Importancia de cada hiperparámetro (fANOVA)

| Hiperparámetro | Importancia |
|---|---|
| min_lector | 0.72 |
| lambda | 0.23 |
| min_libro | 0.05 |

## Mejores trials

| # | NDCG@20 | min_lector | min_libro | lambda |
|---|---|---|---|---|
| 37 | 0.1211 | 20 | 5 | 298.8 |
| 46 | 0.1202 | 20 | 8 | 540.8 |
| 33 | 0.1201 | 20 | 8 | 518.9 |
| 22 | 0.1197 | 20 | 10 | 367.7 |
| 45 | 0.1196 | 21 | 8 | 656.9 |
| 36 | 0.1194 | 20 | 5 | 1908 |
| 38 | 0.1192 | 21 | 5 | 244.4 |
| 28 | 0.1182 | 20 | 13 | 289.8 |
| 41 | 0.1178 | 23 | 6 | 449.8 |
| 44 | 0.1173 | 22 | 6 | 161.1 |
