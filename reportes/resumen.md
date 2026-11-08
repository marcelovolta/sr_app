# Optimización de los recomendadores

Generado el 2026-11-08 15:26 con una copia de `data/data.db`.

## Datos

- Interacciones (rating > 0): 461075, de 10667 lectores y 48063 libros.
- Partición cronológica por lector: las últimas 20 interacciones son test y las anteriores train.
- Train: 385055 interacciones. Test: 76020 (3801 lectores con más de 20 interacciones).

## Método

- Cada modelo se ajusta con Optuna (TPE) maximizando NDCG@20 en el test.
- El filtro de la base es un hiperparámetro más: mínimo de interacciones por lector (≥ 20) y por libro, aplicado de forma recursiva y sólo sobre train.
- Se evalúa siempre a todos los lectores del test; los que el filtro deja afuera reciben los más leídos, así un filtro exigente no se ve favorecido por evaluar sólo a lectores fáciles.
- Relevante = cualquier libro leído en el test, sin importar el rating.
- Con los mejores hiperparámetros se entrena el modelo final con todos los datos.
- El NDCG se mide en el mismo test con el que se eligieron los hiperparámetros: es algo optimista.

## Resultados

Referencia: recomendar a todos los más leídos da NDCG@20 = 0.0889.

| Modelo | NDCG@20 | vs popularidad | Lectores del test cubiertos | min_lector | min_libro | Lectores (final) | Libros (final) | Trials |
|---|---|---|---|---|---|---|---|---|
| [naive](naive.md) | **0.0878** | 0.99× | 14% | 80 | 47 | 699 | 776 | 50 |
| [svd](svd.md) | **0.0880** | 0.99× | 13% | 80 | 50 | 653 | 688 | 50 |
| [ease](ease.md) | **0.1211** | 1.36× | 69% | 20 | 5 | 3704 | 11453 | 50 |
| [dos_torres](dos_torres.md) | **0.1178** | 1.33× | 64% | 21 | 18 | 3390 | 3645 | 50 |
| [gran_torre](gran_torre.md) | **0.1070** | 1.20× | 64% | 20 | 23 | 3406 | 2880 | 30 |
| [ensamble](ensamble.md) | **0.1251** | 1.41× | 69% | — | — | — | — | 30 |

## Cascadas

Hasta 2 valoraciones: top_n. Si el modelo no conoce al lector (quedó afuera del filtro): visión de túnel. El resto: el modelo.

Reparto de los 832 lectores de `prediction_base.csv`:

| Cascada | top_n | visión de túnel | modelo |
|---|---|---|---|
| cascada_naive | 43 | 581 | 208 |
| cascada_svd | 43 | 592 | 197 |
| cascada_ease | 43 | 131 | 658 |
| cascada_dos_torres | 43 | 158 | 631 |
| cascada_gran_torre | 43 | 161 | 628 |
| cascada_ensamble | 43 | 131 | 658 |
