# Métricas do detector (split: test)

| Classe | Precisão | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|
| rua_alagada | 0.903 | 0.758 | 0.824 | 0.886 | 0.647 |
| bueiro_entupido | 0.555 | 0.600 | 0.576 | 0.572 | 0.286 |
| veiculo_submerso | 0.780 | 0.619 | 0.690 | 0.722 | 0.565 |
| **todas** | **0.746** | **0.659** | **0.700** | **0.727** | **0.499** |

Tempo de inferência: 23.1 ms/imagem (pré 0.8 ms, pós 0.6 ms).

Por foto (confiança ≥ 0.40, como na API): 108 de 119 fotos com alagamento/bueiro viraram ocorrência; 4 de 35 fotos sem nada geraram alarme falso.
