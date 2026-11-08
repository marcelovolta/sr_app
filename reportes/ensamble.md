# Ensamble (LightGBM)

Optimizado el 2026-11-08T15:25 con 30 trials de Optuna (TPE), maximizando NDCG@20.

## Cómo funciona

- Candidatos: los 50 primeros libros de cada modelo base (svd, ease, dos_torres, gran_torre), unidos.
- Variables: el puntaje y la posición de cada candidato en cada modelo.
- LightGBM con objetivo LambdaRank reordena los candidatos.
- Sin fuga de datos: LightGBM se entrena con modelos base que no vieron las respuestas (train sin sus últimas 20 interacciones por lector → esas 20) y se evalúa en test.

## Resultado (NDCG@20 en test)

| Modelo | NDCG@20 |
|---|---|
| **Ensamble LightGBM** | **0.1251** |
| promedio simple | 0.1252 |
| ease | 0.1211 |
| dos_torres | 0.1178 |
| gran_torre | 0.1070 |
| svd | 0.0880 |
| popularidad | 0.0889 |

Lectores del test cubiertos por algún modelo base: 69%.

## Importancia de las variables (ganancia)

| Variable | Importancia |
|---|---|
| ease_puntaje | 31% |
| ease_posicion | 21% |
| dos_torres_puntaje | 19% |
| dos_torres_posicion | 12% |
| gran_torre_puntaje | 8% |
| svd_puntaje | 4% |
| gran_torre_posicion | 4% |
| svd_posicion | 1% |

## Hiperparámetros de LightGBM

| Hiperparámetro | Valor |
|---|---|
| num_leaves | 9 |
| learning_rate | 0.01491 |
| n_estimators | 282 |
| min_child_samples | 32 |
| colsample_bytree | 0.5504 |
| reg_lambda | 0.3 |
