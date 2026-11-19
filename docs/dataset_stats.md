# Estatísticas do dataset (gerado por scripts/prepare_dataset.py)

| Split | Imagens | Sem caixa (negativas) | rua_alagada | bueiro_entupido | veiculo_submerso |
|---|---:|---:|---:|---:|---:|
| train | 1055 | 243 | 898 | 34 | 508 |
| val | 309 | 71 | 270 | 5 | 137 |
| test | 154 | 35 | 128 | 5 | 63 |
| **total** | **1518** | **349** | **1296** | **44** | **708** |

Valores das colunas de classe = número de caixas (instâncias).

## Fontes

| Fonte | Imagens lidas | Classes originais (instâncias) | Mapeamento | URL |
|---|---:|---|---|---|
| cvfd | 4491 | mascara (725), veiculo_pseudo_rotulo (342) | mascara → rua_alagada, veiculo_pseudo_rotulo → veiculo_submerso | https://www.kaggle.com/datasets/sunnyshabanali/close-view-flood-dataset-cvfd |
| roadway_flooding | 440 | mascara (453), veiculo_pseudo_rotulo (366) | mascara → rua_alagada, veiculo_pseudo_rotulo → veiculo_submerso | https://data.mendeley.com/datasets/t395bwcvbw/1 |
| flood_area_seg | 74 | mascara (122) | mascara → rua_alagada | https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation |
| bueiros_commons | 177 | bueiro_entupido (44) | bueiro_entupido → bueiro_entupido | https://commons.wikimedia.org/wiki/Category:Storm_drains |

Contagens = imagens únicas. No treino, as fotos positivas destas fontes são gravadas mais de uma vez para compensar a classe rara: bueiros_commons ×3.
