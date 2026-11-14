# Optimización de los recomendadores

Generado el 2026-11-11 20:30 con una copia de `data/data.db`.

## Datos

- Interacciones (rating > 0): 461075, de 10667 lectores y 48063 libros.
- Universo de los modelos: los 3203 lectores con al menos 30 interacciones (408136 interacciones).
- Partición cronológica por lector: las últimas 10 interacciones son test, las 10 anteriores validación y el resto train.
- Train: 344076 interacciones. Validación: 32030. Test: 32030.

## Método

- Cada modelo se ajusta con Optuna (TPE): entrena con train y maximiza NDCG@20 en validación.
- Con los mejores hiperparámetros se reentrena con train + validación y se mide en test: es el NDCG que se informa (el test no participó de ninguna elección).
- El modelo final se entrena con todos los lectores con al menos 10 interacciones (todas sus interacciones), con los hiperparámetros elegidos.
- Todos los lectores del universo se entrenan y se evalúan, siempre los mismos.
- El mínimo de interacciones por libro es un hiperparámetro, contado sólo sobre los datos de entrenamiento.
- Relevante = cualquier libro leído, sin importar el rating.

## Resultados

Referencia: recomendar a todos los más leídos da NDCG@20 = 0.0504 en test.

| Modelo | NDCG@20 (test) | vs popularidad | NDCG@20 (validación) | Lectores del universo que conoce | min_libro | Lectores (final) | Libros (final) | Trials |
|---|---|---|---|---|---|---|---|---|
| [naive](naive.md) | **0.0316** | 0.63× | 0.0465 | 100% | 8 | 5224 | 7910 | 50 |
| [svd](svd.md) | **0.0210** | 0.42× | 0.0323 | 100% | 49 | 5205 | 1453 | 50 |
| [ease](ease.md) | **0.0981** | 1.95× | 0.1332 | 100% | 6 | 5228 | 10174 | 60 |
| [dos_torres](dos_torres.md) | **0.0977** | 1.94× | 0.1366 | 100% | 9 | 5224 | 7121 | 50 |
| [gran_torre](gran_torre.md) | **0.0787** | 1.56× | 0.1231 | 100% | 22 | 5218 | 3180 | 30 |
| [ensamble](ensamble.md) | **0.1074** | 2.13× | 0.1566 | 100% | — | — | — | 30 |

## Cascadas

Hasta 2 valoraciones: top_n. Si el modelo no conoce al lector (menos de 10 interacciones): visión de túnel. El resto: el modelo.

Reparto de los 832 lectores de `prediction_base.csv`:

| Cascada | top_n | visión de túnel | modelo |
|---|---|---|---|
| cascada_naive | 43 | 61 | 728 |
| cascada_svd | 43 | 62 | 727 |
| cascada_ease | 43 | 61 | 728 |
| cascada_dos_torres | 43 | 61 | 728 |
| cascada_gran_torre | 43 | 61 | 728 |
| cascada_ensamble | 43 | 61 | 728 |
