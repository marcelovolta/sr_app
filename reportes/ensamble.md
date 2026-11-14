# Ensamble (LightGBM)

Optimizado el 2026-11-11T20:27 con 30 trials de Optuna (TPE), maximizando NDCG@20 en validación (lectores apartados de la etapa A).

## Cómo funciona

- Candidatos: los 50 primeros libros de cada modelo base (svd, ease, dos_torres, gran_torre), unidos.
- Variables: el puntaje y la posición de cada candidato en cada modelo, y de contenido: qué parte de lo que leyó el lector comparte autor, género o editorial con el candidato, el año de edición y la popularidad.
- LightGBM con objetivo LambdaRank reordena los candidatos.
- Sin fuga de datos: LightGBM aprende a predecir la validación con modelos base entrenados con train, y se evalúa en test con modelos base entrenados con train + validación.

## Resultado (NDCG@20 en test)

| Modelo | NDCG@20 |
|---|---|
| **Ensamble LightGBM** | **0.1074** |
| ease | 0.0981 |
| dos_torres | 0.0977 |
| promedio simple | 0.0976 |
| gran_torre | 0.0787 |
| svd | 0.0210 |
| popularidad | 0.0504 |

NDCG@20 en validación (el que maximizó Optuna): 0.1566.

Lectores del universo que conoce algún modelo base: 100%.

## Importancia de las variables (ganancia)

| Variable | Importancia |
|---|---|
| ease_posicion | 22% |
| dos_torres_posicion | 11% |
| afinidad_autor | 9% |
| anio_edicion | 9% |
| popularidad | 8% |
| ease_puntaje | 7% |
| svd_posicion | 7% |
| gran_torre_puntaje | 6% |
| afinidad_genero | 5% |
| dos_torres_puntaje | 5% |
| svd_puntaje | 4% |
| afinidad_editorial | 4% |
| gran_torre_posicion | 4% |

## Hiperparámetros de LightGBM

| Hiperparámetro | Valor |
|---|---|
| num_leaves | 35 |
| learning_rate | 0.0174 |
| n_estimators | 487 |
| min_child_samples | 138 |
| colsample_bytree | 0.9697 |
| reg_lambda | 3.796 |
