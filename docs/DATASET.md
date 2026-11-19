# Dataset — Alaga Alerta

Dataset de **detecção de objetos** (formato YOLO: uma caixa por objeto, coordenadas normalizadas) montado só com
fontes **públicas e baixáveis sem login**: três datasets de segmentação de água do Kaggle e fotos de bueiros do
Wikimedia Commons. Tudo é reproduzível a partir de [training/datasets.yaml](../training/datasets.yaml):

```bash
python scripts/coletar_bueiros.py      # fotos de bueiro (lista revisada em training/bueiros_commons.csv)
python scripts/prepare_dataset.py      # baixa o Kaggle, converte, gera pseudo-rótulos e divide
```

> As quantidades exatas (imagens e caixas por classe e por split) são geradas a cada execução em
> [dataset_stats.md](dataset_stats.md) e entram automaticamente no PDF.

## Classes e critério de anotação

| Id | Classe | Conta como positivo | Não conta |
|---|---|---|---|
| 0 | `rua_alagada` | água acumulada cobrindo rua, calçada ou área urbana (caixa sobre a mancha de água) | asfalto só molhado, poças pequenas (< 2% da foto) |
| 1 | `bueiro_entupido` | boca de lobo, grelha ou bueiro com obstrução visível (lixo, folhas, terra, mato) | bueiro limpo e tampas de inspeção (viram exemplos negativos) |
| 2 | `veiculo_submerso` | carro, moto, ônibus ou caminhão com água pelo menos na altura das rodas | veículo em chão seco ou rua só molhada |

## Origem

| Fonte | O que tem | Como vira caixa | Licença | Link |
|---|---|---|---|---|
| `cvfd` — Close-View Flood Dataset | ~3,7 mil fotos alagadas ao nível do chão com máscara de água + ~4,2 mil sem alagamento | máscara → `rua_alagada`; veículos → `veiculo_submerso` (pseudo-rótulo); fotos sem alagamento → negativas | CC BY-SA 4.0 | [kaggle](https://www.kaggle.com/datasets/sunnyshabanali/close-view-flood-dataset-cvfd) |
| `roadway_flooding` — Roadway Flooding Image Dataset (Sazara, Cetin, Iftekharuddin, 2019) | 441 fotos de ruas/estradas alagadas com máscara | máscara → `rua_alagada`; veículos → `veiculo_submerso` (pseudo-rótulo) | CC BY 4.0 | [Mendeley Data](https://data.mendeley.com/datasets/t395bwcvbw/1) / [kaggle](https://www.kaggle.com/datasets/saurabhshahane/roadway-flooding-image-dataset) |
| `flood_area_seg` — Flood Area Segmentation | 290 fotos de áreas alagadas com máscara (maioria aérea) | máscara → `rua_alagada` (só as com contexto de rua) | CC0 | [kaggle](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation) |
| `bueiros_commons` — Wikimedia Commons | fotos de bocas de lobo/grelhas de rua, revisadas uma a uma | caixa desenhada manualmente → `bueiro_entupido`; bueiro limpo → negativa | CC0 / CC BY / CC BY-SA (por foto, em `creditos.csv`) | [Commons: Storm drains](https://commons.wikimedia.org/wiki/Category:Storm_drains) |
| `blocked_drain`, `drowning_cars` (opcionais) | datasets do Roboflow Universe | regras de `mapa` | CC BY 4.0 | exigem `ROBOFLOW_API_KEY` (conta grátis); sem a chave são pulados |

Os créditos de cada foto do Commons (autor, licença, página) ficam em `data/raw/bueiros_commons/creditos.csv`,
gerado pelo `coletar_bueiros.py`.

## Pré-processamento

1. **Download**: Kaggle pela API pública (sem login); Commons pela API do MediaWiki (miniaturas de 500 px).
2. **Máscara → caixas** (`rua_alagada`): pixel de água = maior canal acima do limiar da fonte; fechamento morfológico
   7×7; componentes conexos; manchas menores que 2% da imagem são descartadas. Foto marcada como alagada mas sem
   mancha relevante é descartada (ambígua).
3. **Filtro de contexto urbano** (CVFD e Flood Area): o CVFD mistura enchentes de rio, vistas aéreas e paisagens. Um
   YOLO26s pré-treinado no COCO procura objetos de rua (carro, moto, ônibus, caminhão, bicicleta, semáforo, placa de
   pare, hidrante, banco); fotos alagadas sem nenhum deles não entram como positivas. A revisão visual de uma amostra
   do que foi descartado mostrou quase só rios e fotos aéreas.
4. **Pseudo-rótulo de veículo submerso**: o mesmo YOLO26s acha veículos (confiança ≥ 0,40). Mede-se a fração de água
   na faixa logo abaixo da base da caixa (de 10% acima a 15% abaixo da borda inferior, 10% mais larga): se ≥ 0,40, o
   veículo vira `veiculo_submerso`. Medir *abaixo* da caixa funciona porque, num carro submerso, a caixa do detector
   termina na linha d'água; num carro em chão seco, embaixo há asfalto. O critério foi calibrado em folhas de
   conferência com as caixas e a fração desenhadas (carros em água: 0,5–0,9; carros em chão seco: 0,0–0,2).
5. **Bueiros**: 880 fotos candidatas (categorias de bueiros e buscas como "storm drain litter", "clogged drain",
   "boca de lobo"), filtradas por licença livre e revisadas manualmente em folhas de contato: *entupido* recebe caixa
   sobre a grelha/abertura obstruída; *limpo* (grelha desobstruída, tampas de inspeção) entra como negativa; o resto
   (paisagens, canais, produtos) é descartado. Propostas de um detector de vocabulário aberto (YOLOE) ajudaram, mas a
   maioria das caixas foi ajustada à mão.
6. **Deduplicação** por hash MD5.
7. **Classe rara**: as fotos de treino com `bueiro_entupido` são gravadas 3 vezes (`repetir_treino: 3`); o
   mosaic, os flips e a variação de cor do treino fazem cada cópia sair diferente. Validação e teste não são repetidos.
8. **Negativos**: imagens sem caixa (ruas e rios sem alagamento, bueiros limpos) até 30% do número de positivas, para
   o modelo aprender o que *não* é alerta — inclusive rios dentro do leito, que câmeras como as da Marginal veem o tempo todo.

## Divisão treino / validação / teste

- **70% / 20% / 10%**, semente fixa (42).
- A divisão é feita **por fonte**, para que todas as fontes apareçam nos três conjuntos.
- A divisão é feita **por imagem de origem**: cópias aumentadas da mesma foto (padrão do Roboflow,
  `foto_jpg.rf.<hash>.jpg`) ficam no mesmo split, e validação/teste recebem uma única cópia por foto.
- O **teste** não é usado no treino nem na escolha do melhor checkpoint (que usa a validação); as métricas do PDF vêm
  dele (`scripts/evaluate.py`).

## Limitações conhecidas

- `veiculo_submerso` vem de pseudo-rótulos: veículos que o YOLO-COCO não detectou ficam sem caixa (ruído de rótulo) e
  a métrica dessa classe mede em parte a concordância com o próprio critério.
- `bueiro_entupido` tem poucas dezenas de exemplos, a maioria de bueiros europeus fotografados de cima; é a classe mais
  fraca. Fotos próprias de bocas de lobo de SP (fonte `proprias`) ou os datasets do Roboflow são o próximo passo.
- As caixas de `rua_alagada` vêm de máscaras e cobrem manchas inteiras de água; o modelo aprende "região alagada",
  não um objeto de contorno bem definido, o que derruba o mAP50-95 dessa classe.

## Aumento de dados no treino

Padrões do Ultralytics (mosaic, flip horizontal, variação HSV, escala/translação) + rotação de até 5° e leve
perspectiva, porque fotos de cidadão chegam inclinadas e de ângulos variados.

## Como adicionar fotos próprias

1. Fotografe ruas, bueiros e veículos (com e sem alagamento) e anote no Roboflow/CVAT/Label Studio com os nomes das 3 classes.
2. Exporte em formato **YOLO** para `data/raw/proprias/` (pasta com `data.yaml`, `images/`, `labels/`).
3. Em `training/datasets.yaml`, mude `ativo: false` para `true` na fonte `proprias` e rode o `prepare_dataset.py`.
