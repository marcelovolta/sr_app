# Arquitecturas de redes para recomendación: qué usa la literatura

## Dos torres (recuperación)

Una red por lado: la **torre del usuario** y la **torre del ítem** producen cada una un
embedding, y el puntaje es su producto escalar (o la similitud coseno). Como los
embeddings de los ítems se calculan una sola vez, la búsqueda entre millones de ítems es
rápida.

- **Entrenamiento con negativos del batch y corrección logQ** (Yi et al., 2019, YouTube).
  Cada par (usuario, ítem elegido) usa como negativos los ítems de las demás filas del
  batch. Como un ítem popular aparece más seguido como negativo, se corrige restando el
  log de su frecuencia. Es exactamente lo que hace nuestro `dos_torres`. Un trabajo de 2025
  revisa la derivación de esa corrección.
- **Torres con atributos y capas ocultas**: en la práctica cada torre no es sólo una tabla
  de embeddings por id, sino una MLP sobre el id y los atributos. Lo probamos (autor,
  género, editorial y antigüedad del libro; género, nacimiento y España del lector) y no
  mejoró: ver [intento_complejidad.md](intento_complejidad.md). Yang et al. (2020) entrenan torres de ese tipo, y además mezclan los negativos del batch
  con negativos sorteados uniformemente del catálogo (*mixed negative sampling*), para que
  los ítems poco populares también aparezcan como negativos.
- **Producto escalar vs. MLP como combinador**: Rendle et al. (2020) muestran que, bien
  ajustado, el producto escalar supera a combinar los embeddings con una MLP (como en NCF),
  y que a una MLP le cuesta aprender un producto escalar. Por eso la complejidad conviene
  ponerla *dentro* de cada torre y no en cómo se combinan.

## Una gran torre (ranking)

Una sola red que recibe usuario e ítem juntos (los embeddings de todos sus rasgos
concatenados) y estima la probabilidad de interacción. Es más expresiva que dos torres,
porque puede cruzar rasgos del usuario con rasgos del ítem, pero hay que pasar cada par por
la red, así que se usa para ordenar pocos candidatos.

- **Ranking de YouTube** (Covington et al., 2016): embeddings de rasgos categóricos más
  rasgos numéricos normalizados, y una torre de capas ReLU que se angosta. El paper compara
  profundidades de 1 a 4 capas (por ejemplo 1024 → 512 → 256).
- **Wide & Deep** (Cheng et al., 2016, Google Play): la parte *deep* es una red de este
  tipo (generaliza a combinaciones nuevas), y la parte *wide* es un modelo lineal sobre
  cruces de rasgos (memoriza combinaciones frecuentes). Se entrenan juntas.
- **NCF** (He et al., 2017): popularizó la "torre" en pirámide: cada capa con la mitad de
  unidades que la anterior. Probamos esa estructura en `gran_torre` (1 a 4 capas, ancho
  y dropout elegidos por Optuna) y eligió la red más chica; se volvió a 3 capas de 256.
- **DCN-V2** (Wang et al., 2021): agrega capas de cruce explícitas, que modelan
  interacciones entre rasgos de forma más eficiente que una MLP sola. Es el siguiente paso
  natural si la gran torre necesita más capacidad.

## Una advertencia de la literatura

- Ferrari Dacrema et al. (2019) intentaron reproducir 18 métodos neuronales de top-N de
  conferencias importantes: sólo 7 se pudieron reproducir, y 6 de esos perdían contra
  métodos simples bien ajustados (vecinos cercanos, modelos lineales).
- Steck (2019) propone EASE, un modelo lineal libro a libro con solución cerrada, que en
  sus experimentos compite con modelos profundos.
- Conclusión para este trabajo: comparar siempre contra baselines bien ajustados (EASE,
  popularidad). En nuestros resultados EASE supera a las redes.

## Referencias

- Cheng, H.-T. et al. (2016). *Wide & Deep Learning for Recommender Systems*. DLRS @ RecSys. [arXiv:1606.07792](https://arxiv.org/abs/1606.07792)
- Covington, P., Adams, J., Sargin, E. (2016). *Deep Neural Networks for YouTube Recommendations*. RecSys. [research.google/pubs/pub45530](https://research.google/pubs/pub45530)
- He, X., Liao, L., Zhang, H. et al. (2017). *Neural Collaborative Filtering*. WWW. [arXiv:1708.05031](https://arxiv.org/abs/1708.05031)
- Yi, X. et al. (2019). *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations*. RecSys. [research.google](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)
- Steck, H. (2019). *Embarrassingly Shallow Autoencoders for Sparse Data*. WWW. [arXiv:1905.03375](https://arxiv.org/abs/1905.03375)
- Ferrari Dacrema, M., Cremonesi, P., Jannach, D. (2019). *Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches*. RecSys. [arXiv:1907.06902](https://arxiv.org/abs/1907.06902)
- Yang, J. et al. (2020). *Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations*. WWW Companion. [research.google](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/)
- Rendle, S., Krichene, W., Zhang, L., Anderson, J. (2020). *Neural Collaborative Filtering vs. Matrix Factorization Revisited*. RecSys. [arXiv:2005.09683](https://arxiv.org/abs/2005.09683)
- Wang, R. et al. (2021). *DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems*. WWW. [arXiv:2008.13535](https://arxiv.org/abs/2008.13535)
- *Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval* (2025). [arXiv:2507.09331](https://arxiv.org/abs/2507.09331)
