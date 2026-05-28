# Installation serveur — PDF Reader v2

## Prérequis
- Python 3.10+
- CUDA 12.x (pilotes NVIDIA installés)
- pip

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Lancement

```bash
python main.py
# ou avec plusieurs workers :
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

L'interface est accessible sur http://localhost:8000

## Accès réseau local
Depuis un autre poste du réseau : http://<IP_SERVEUR>:8000
