# MD-TP3
UFMG DCC057, trabalho prático 3

Este projeto consistiu na exploração de um banco de dados de imagens de câncer de pele, divididas pelas classes "benign" e "malignant" [banco](https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign). O objetivo do projeto foi conseguir um modelo com F-SCORE no conjunto de teste maior ou igual a 95%. Todavia, isto se mostrou impraticável com o conjunto de dados escolhidos.

No final, o modelo implementado foi uma simples rede neural convolutiva, cuja implementação se baseia largamente [neste notebook](https://github.com/google/eng-edu/blob/main/ml/pc/exercises/image_classification_part1.ipynb).

Os notebooks phs1, phs2, phs3, phs4 correspondem ao desenvolvimento das fases 1, 2, etc. Porém, no final do projeto, devido a vários empecilhos, somente o notebook phs4 (fase de modelagem), e o código python para separação do conjunto de validação em "pysrc/loading/load.py" foram utilizados.
