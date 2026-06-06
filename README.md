# Vet Eye AI USG Navigation — FURASSIST

AI-asystowana nawigacja ultrasonograficzna AFAST dla psów i kotów.
Projekt końcowy BiznesAI 15, Akademia Leona Koźmińskiego.

> **Zastrzeżenie:** system trenowany na danych syntetycznych — nie do użytku klinicznego.

## Linki

| Zasób | URL |
|---|---|
| 🤗 Dataset | [koscielnamarta/synthetic-usg-afast-vet](https://huggingface.co/datasets/koscielnamarta/synthetic-usg-afast-vet) |
| 🤗 Model | [koscielnamarta/synthetic-usg-afast-vet-classifier](https://huggingface.co/koscielnamarta/synthetic-usg-afast-vet-classifier) |
| 🤗 Demo (HF Spaces) | [koscielnamarta/vet-eye-usg-demo](https://huggingface.co/spaces/koscielnamarta/vet-eye-usg-demo) |

## Jak czytać to portfolio

1. [`notebooks/02b_finetune_synthetic_usg.ipynb`](notebooks/02b_finetune_synthetic_usg.ipynb) — trening: fine-tuning TinyUSFM na syntetycznym datasecie AFAST (transfer learning, 2 etapy: linear probing → partial unfreezing)
2. [`notebooks/03_furassist_architecture.ipynb`](notebooks/03_furassist_architecture.ipynb) — wdrożenie: pełny pipeline od obrazu do instrukcji nawigacyjnej, metryki na 3 zbiorach, stress test
3. [`archive/`](archive/) — wcześniejsze iteracje: porównanie domenowe FETAL_PLANES, kalibracja progu pewności

## Struktura repozytorium

```
notebooks/
  02b_finetune_synthetic_usg.ipynb   ← trening (artefakt treningowy)
  03_furassist_architecture.ipynb    ← pipeline wdrożeniowy (reference implementation)
archive/
  02_finetune_fetal_planes.ipynb     ← porównanie domenowe (FETAL_PLANES_DB, 6 klas)
  04_confidence_threshold_calibration.ipynb  ← kalibracja progu pewności
src/
  confidence_thresholder.py          ← post-processing (próg 0.85)
  scripted_instructions.py           ← 16 instrukcji nawigacyjnych AFAST
app.py                               ← Gradio app (FURASSIST, HF Spaces)
requirements.txt
README.md
```

## Architektura systemu

```
Obraz USG (PIL)
     │
     ▼
eval_transform   Grayscale(3) → Resize(224) → ToTensor → Normalize(ImageNet)
     │
     ▼
TinyUSFM         ViT-Tiny, 5.5M params, fine-tuned na synthetic-usg-afast-vet
                 head: Linear(192 → 4 klas: CC / DH / HR / SR)
     │
     ▼
softmax          [P(CC), P(DH), P(HR), P(SR)]
     │
     ▼
ConfidenceThresholder   max(softmax) ≥ 0.85 → klasa
                        max(softmax) < 0.85 → "niepewne"
     │
     ▼
AFASTNavigator   (current_view, views_done) → instrukcja nawigacyjna (PL/EN)
```

## Wyniki

| Zbiór | Acc (accepted) | Acc (overall) | Abstain rate | Macro F1 |
|---|---|---|---|---|
| Val (czysty syntetyk) | 1.0000 | 0.9525 | 0.0475 | 0.9749 |
| Holdout easy | 1.0000 | 0.9263 | 0.0737 | 0.9593 |
| Holdout hard (OOD-like) | 0.8056 | 0.4713 | 0.4150 | 0.5324 |

*Threshold = 0.85, model: `checkpoints/best_i2.pt` (I2 partial unfreezing, 1 blok).*

## Quick start (Colab)

```python
# 1. Klonuj TinyUSFM
import subprocess, sys
subprocess.run(["git", "clone", "https://github.com/MacDunno/TinyUSFM", "TinyUSFM"])
sys.path.insert(0, "TinyUSFM")

# 2. Załaduj model
import torch, torch.nn as nn
from model.tinyusfm import TinyUSFM
from huggingface_hub import hf_hub_download

model = TinyUSFM()
model.model.head = nn.Linear(192, 4)
ckpt = hf_hub_download("koscielnamarta/synthetic-usg-afast-vet-classifier",
                        "checkpoints/best_i2.pt")
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

# 3. Predykcja
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Grayscale(3), transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
img = Image.open("usg.png")
with torch.no_grad():
    probs = torch.softmax(model(transform(img).unsqueeze(0)), dim=1)
print(["CC", "DH", "HR", "SR"][probs.argmax()])
```

Pełna dokumentacja pipeline'u: [`notebooks/03_furassist_architecture.ipynb`](notebooks/03_furassist_architecture.ipynb)

## Autorzy i źródła

- Marta Kościelna / BiznesAI 15 / Akademia Leona Koźmińskiego, 2026
- Backbone: [TinyUSFM](https://github.com/MacDunno/TinyUSFM) (MacDunno, MIT)
- AFAST protocol: Lisciandro 2011; Boysen & Lisciandro 2013
