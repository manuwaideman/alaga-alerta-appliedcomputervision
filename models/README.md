# models/

Pesos usados pela API e pela preparação do dataset. Os quatro primeiros já vêm no repositório, então a API roda
logo depois do clone; os `yolo26*` da Ultralytics são baixados automaticamente quando necessários.

| Arquivo | Origem | Como obter |
|---|---|---|
| `alaga_yolo.pt` | detector treinado pela equipe (YOLO26n, 3 classes, 448 px, ~5 MB) | incluído no repositório; para retreinar: `python scripts/train.py ...` |
| `alaga_yolo.json` | metadados do treino (modelo base, run, data) | gerado junto com o `alaga_yolo.pt` |
| `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo — YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT) | incluído no repositório (ou `python scripts/download_models.py`) |
| `license-plate-finetune-v1n.onnx` | [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) (AGPL-3.0) | incluído no repositório (ou `python scripts/download_models.py`) |
| `yolo26n.pt` | Ultralytics, pré-treinado no COCO: ponto de partida do treino | baixado automaticamente pelo `train.py` |
| `yolo26s.pt` | Ultralytics, pré-treinado no COCO: filtro urbano e pseudo-rótulos de veículo (só na preparação do dataset) | baixado automaticamente pelo `prepare_dataset.py` |

Os modelos de terceiros usados pela API rodam em ONNX, que não executa código ao ser carregado
(diferente de `.pt`, que é um pickle). Os `.pt` acima são da Ultralytics ou treinados pela própria equipe.
