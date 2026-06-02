# Vet Eye AI Navigation USG — demo

[![HF Dataset](https://img.shields.io/badge/🤗_Dataset-synthetic--usg--afast--vet-yellow)](https://huggingface.co/datasets/koscielnamarta/synthetic-usg-afast-vet)
[![HF Model](https://img.shields.io/badge/🤗_Model-synthetic--usg--afast--vet--classifier-yellow)](https://huggingface.co/koscielnamarta/synthetic-usg-afast-vet-classifier)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)

> AI-asystowana nawigacja USG dla weterynarii — proof of concept zbudowany w ramach projektu końcowego programu **BiznesAI 15** (Akademia Leona Koźmińskiego, edycja 2025/2026).
>
> **Zespół projektu (kapitał intelektualny):** Wiktoria Mikołajów, Małgorzata Polaczuk, Aneta Szurmak, Marta Kościelna, Tomasz Dębowski, Tomasz Fic, Marcin Leszczyński, Artur Charles.
> **Implementacja techniczna POC:** Marta Kościelna.
>
> Termin obrony: 4–6 lipca 2026.

🇵🇱 [Polski](#-polski) · 🇬🇧 [English](#-english)

---

## 🇵🇱 Polski

### Czym jest ten projekt

To proof-of-concept (PoC) systemu, który w czasie rzeczywistym analizuje obraz z głowicy ultrasonografu i podpowiada operatorowi, jak ją przesunąć, aby uzyskać prawidłową projekcję narządu. Bezpośrednie odpowiedniki w medycynie ludzkiej: **Caption AI** (GE HealthCare) w kardiologii i **UltraSight** w POCUS sercowym.

Klient docelowy rozwiązania: **Vet Eye S.A.** — polski producent ultrasonografów weterynaryjnych (linie *vet pro-key 75/76*, *vet portable 15*, *vet pro 70*).

### Status projektu

| Tydzień | Daty | Cel | Status |
| :--- | :--- | :--- | :---: |
| 1 | 4–10.05.2026 | Setup środowiska + pretrained inference | 🟢 gotowe |
| 2 | 11–17.05.2026 | DataLoader + pierwszy fine-tuning (FETAL_PLANES val_acc 0.8583) + detour syntetyczny | 🟢 gotowe |
| 3 | 18–24.05.2026 | Notebook 02b finalny + 3-way split + domain randomization + HF Model | 🟢 gotowe |
| 4 | 25–31.05.2026 | Confidence threshold + scripted instructions + notebook 03 (pipeline) + Gradio + HF Spaces | ⚪ w toku |
| 5 | 1–7.06.2026 | Sekcje techniczne pracy końcowej (kosztorys, legal, POC, demo) | ⚪ |
| 6 | 8–14.06.2026 | Sekcje zmiany (Kotter + kompetencje) + samodzielny QA | ⚪ |
| 7 | 15–20.06.2026 | Bufor + finalna korekta + złożenie pracy | ⚪ |

### Wyniki dotychczas (po Tygodniu 3)

#### Notebook 02 — FETAL_PLANES_DB (humanowe USG, jako proxy historyczne)

| Metryka | Wartość |
| :--- | ---: |
| Backbone | TinyUSFM (5,5 M params, ViT-Tiny) |
| Dataset | FETAL_PLANES_DB (Zenodo), 12 400 obrazów, 6 klas |
| I1 — linear probing (5 ep, lr=1e-3, 1 542 wag) | val_acc 0.6691 |
| I2 — partial unfreezing 2 bloków (8 ep, lr=1e-4, 893k wag) | **val_acc 0.8583** |
| Macro F1 / Weighted F1 | 0.85 / 0.86 |

#### Notebook 02b — syntetyczny AFAST (weterynaryjny, **primary path do demo**)

| Metryka | Wartość |
| :--- | ---: |
| Backbone | TinyUSFM (5,5 M params) |
| Dataset | Syntetyczny AFAST 4 klasy (CC, DH, HR, SR), 4 800 obrazów total (3-way split) |
| Konfiguracja | I1 linear probing (5 ep, lr=1e-3 + CosineAnnealing) + I2 unfreeze 1 blok (8 ep, lr=1e-4, 446k wag = 8%) |
| Augmentations | 10 augmentations (geometric + photometric + USG-specific) — domain randomization |
| **Val accuracy** | 0.9975 |
| **Easy holdout** (in-distribution, seed=123) | 0.9962 |
| **Hard holdout** (OOD, rozszerzony zakres parametrów) | **0.7700** |
| Stress test (heavy noise+blur) | 0.79 (easy) / 0.56 (hard) |

**Kluczowy wynik:** dorzucenie domain randomization w `train_transform` zwiększyło accuracy na hard holdout z 0.38 do 0.77 (**+39 pp**). To empiryczna walidacja wartości domain randomization dla syntetycznych datasetów ML w medical imaging.

### Architektura — Wariant E (hybrydowy)

Pipeline 4–6 modeli AI ze wspólnym backbone (foundation model dla USG) z trzema głowicami zadaniowymi:

- **View classifier** — która projekcja jest aktualnie widoczna (CC/DH/HR/SR dla AFAST) — ✅ zaimplementowane w 02b
- **Quality head** — czy bieżąca klatka jest „diagnostic-quality" (auto-capture trigger) — 🚧 confidence threshold jako MVP, multitask quality head jako roadmap Fazy 2
- **Action head** — jaki ruch głowicy zalecić operatorowi — 🚧 scripted (tabela 4×4 mapowań current→target) w MVP, behavioral cloning + RL jako roadmap

Plus orkiestrator (reguły + opcjonalnie mały LLM) generujący tekst instrukcji oraz UI/TTS z overlay graficznym.

### Demo na obronę — Poziom 2

Świadome uproszczenie dla wykonalności w 7 tygodni:

- **Backbone:** TinyUSFM (5,5 M parametrów, GitHub `MacDunno/TinyUSFM`) — pretrained na 2 mln obrazów USG
- **Dataset (primary):** **syntetyczny AFAST 4 klasy** — proceduralnie wygenerowany (numpy/PIL + szum Rayleigha + prymitywne struktury anatomiczne). Decyzja regulacyjna: POC trenowany wyłącznie na syntetyku weterynaryjnym **nie jest klasyfikowany jako „AI system wysokiego ryzyka"** wg AI Act Art. 6 + Aneks III pkt 5(b).
- **Dataset (porównanie domenowe):** FETAL_PLANES_DB jako historyczny proxy — dyskusja w pracy o domain transfer między medical (human) a vet.
- **Quality head:** confidence thresholding (max(softmax) z parametrycznym progiem) jako MVP
- **Action head:** scripted (tabela 4×4=16 par mapowań current→target dla AFAST)
- **UI:** Gradio (jeden plik `app.py`)
- **Hosting:** Hugging Face Spaces (free tier)

### Stack techniczny

| Warstwa | Narzędzie |
| :--- | :--- |
| IDE | VS Code |
| Compute | Google Colab Pro (GPU T4) |
| ML framework | PyTorch + Hugging Face Transformers/Datasets |
| Backbone | TinyUSFM (ViT-Tiny, 5,5 M params) |
| Dataset hosting | 🤗 Hugging Face Datasets |
| Model hosting | 🤗 Hugging Face Model Hub |
| UI | Gradio |
| Deploy | Hugging Face Spaces |
| Wersjonowanie | GitHub (to repo) |
| Backup demo | OBS Studio (MP4) |

**Świadoma decyzja "no Drive":** cały pipeline (kod / dane / model / demo) hostowany publicznie na GitHub + Hugging Face. Brak prywatnych zależności (Google Drive itp.), pełna reproducybilność dla każdego klonującego repo.

### Resources

| Co | Gdzie | URL |
| :--- | :--- | :--- |
| Kod | GitHub (to repo) | [koscielnamarta/vet-eye-ai-usg-demo](https://github.com/koscielnamarta/vet-eye-ai-usg-demo) |
| Dataset (4 800 obrazów, 3-way split + hard) | HF Datasets | [koscielnamarta/synthetic-usg-afast-vet](https://huggingface.co/datasets/koscielnamarta/synthetic-usg-afast-vet) |
| Fine-tuned model (best_i2 + artefakty) | HF Model Hub | [koscielnamarta/synthetic-usg-afast-vet-classifier](https://huggingface.co/koscielnamarta/synthetic-usg-afast-vet-classifier) |

### Quick start

```python
# 1. Pobierz dataset (4 800 obrazów: train/val/holdout/hard)
from huggingface_hub import snapshot_download
data_root = snapshot_download(
    repo_id="koscielnamarta/synthetic-usg-afast-vet",
    repo_type="dataset"
)

# 2. Pobierz wytrenowany model
from huggingface_hub import snapshot_download
model_root = snapshot_download(
    repo_id="koscielnamarta/synthetic-usg-afast-vet-classifier",
    repo_type="model"
)
# checkpoint: model_root/checkpoints/best_i2.pt
# artefakty:  model_root/outputs/  (confusion matrices, classification reports, stress test)

# 3. Zobacz notebooks w repo:
#    notebooks/02_finetune_fetal_planes.ipynb  - baseline na FETAL_PLANES_DB
#    notebooks/02b_finetune_synthetic_usg.ipynb - PRIMARY: syntetyk AFAST 3-way split
#    notebooks/03_full_pipeline.ipynb           - W TOKU: full pipeline (confidence + scripted + temporal)
```

### Struktura repo

```
vet-eye-ai-usg-demo/
├── notebooks/                      # Jupyter/Colab notebooks (numerowane: 01_, 02_, 02b_, 03_)
├── synthetic_usg_dataset/
│   ├── generate_synthetic_usg.py   # generator (4 klasy AFAST, --holdout, --verify, --repair)
│   └── README.md                   # opis datasetu + ograniczenia
├── outputs/                        # artefakty treningu (confusion matrix, learning curves)
├── app/                            # aplikacja Gradio (app.py) — w toku, T4
├── docs/                           # diagramy, screenshoty, fragmenty pracy
├── README.md                       # ten plik
├── LICENSE                         # MIT
└── .gitignore                      # ignorowane pliki (cache, raw data, syntetyk PNG)
```

### ⚠️ Ważne zastrzeżenie (disclaimer)

Ten projekt to **demonstracja akademicka**, nie wyrób medyczny. W szczególności:

- Model jest trenowany na **w pełni syntetycznych** obrazach (proceduralna generacja numpy/PIL) — co ogranicza zastosowanie do testu pipeline'u, nie diagnostyki klinicznej
- Brak walidacji na realnych obrazach USG zwierząt, brak walidacji klinicznej, brak certyfikacji MDR / FDA / AI Act
- Konsultacja kliniczna z lekarzem weterynarii specjalizującym się w USG jest konieczna przed jakimkolwiek wnioskowaniem o przydatności klinicznej
- **NIE używać do podejmowania decyzji diagnostycznych** u zwierząt ani u ludzi
- Wszelkie wyniki służą wyłącznie do weryfikacji wykonalności technicznej (proof of concept) i argumentacji biznesowej w pracy dyplomowej BiznesAI 15

### Licencja

[MIT](LICENSE) — kod jest open source. Możesz go używać, modyfikować i dystrybuować, pod warunkiem zachowania informacji o autorach i licencji.

---

## 🇬🇧 English

### About the project

A proof-of-concept (PoC) system that analyzes the live feed from an ultrasound probe in real time and instructs the operator how to move the probe to obtain a correct organ view. Direct human-medicine equivalents: **Caption AI** (GE HealthCare) in cardiology and **UltraSight** in cardiac POCUS.

Target customer: **Vet Eye S.A.** — Polish manufacturer of veterinary ultrasound systems (product lines *vet pro-key 75/76*, *vet portable 15*, *vet pro 70*).

### Project status

| Week | Dates | Goal | Status |
| :--- | :--- | :--- | :---: |
| 1 | May 4–10, 2026 | Environment setup + pretrained inference | 🟢 done |
| 2 | May 11–17, 2026 | DataLoader + first fine-tuning (FETAL_PLANES val_acc 0.8583) + synthetic detour | 🟢 done |
| 3 | May 18–24, 2026 | Final 02b notebook + 3-way split + domain randomization + HF Model | 🟢 done |
| 4 | May 25–31, 2026 | Confidence threshold + scripted instructions + pipeline notebook + Gradio + HF Spaces | ⚪ in progress |
| 5 | Jun 1–7, 2026 | Thesis technical sections (costs, legal, POC, demo) | ⚪ |
| 6 | Jun 8–14, 2026 | Change-management sections (Kotter + competencies) + self QA | ⚪ |
| 7 | Jun 15–20, 2026 | Buffer + proofreading + submission | ⚪ |

### Current results (end of Week 3)

#### Notebook 02 — FETAL_PLANES_DB (human ultrasound, historical proxy)

| Metric | Value |
| :--- | ---: |
| Backbone | TinyUSFM (5.5M params, ViT-Tiny) |
| Dataset | FETAL_PLANES_DB (Zenodo), 12,400 images, 6 classes |
| I1 — linear probing (5 ep, lr=1e-3) | val_acc 0.6691 |
| I2 — partial unfreezing of 2 blocks (8 ep, lr=1e-4) | **val_acc 0.8583** |
| Macro F1 / Weighted F1 | 0.85 / 0.86 |

#### Notebook 02b — Synthetic AFAST (veterinary, **primary demo path**)

| Metric | Value |
| :--- | ---: |
| Backbone | TinyUSFM (5.5M params) |
| Dataset | Synthetic AFAST 4 classes (CC, DH, HR, SR), 4,800 images total (3-way split) |
| Configuration | I1 linear probing (5 ep, lr=1e-3 + CosineAnnealing) + I2 unfreeze 1 block (8 ep, lr=1e-4, 446k weights = 8%) |
| Augmentations | 10 augmentations (geometric + photometric + USG-specific) — domain randomization |
| **Validation accuracy** | 0.9975 |
| **Easy holdout** (in-distribution, seed=123) | 0.9962 |
| **Hard holdout** (OOD, extended parameter range) | **0.7700** |
| Stress test (heavy noise+blur) | 0.79 (easy) / 0.56 (hard) |

**Key finding:** adding domain randomization to `train_transform` raised hard holdout accuracy from 0.38 to 0.77 (**+39 pp**). Empirical validation of the value of domain randomization for synthetic ML datasets in medical imaging.

### Architecture — Variant E (hybrid)

A pipeline of 4–6 AI models with a shared backbone (USG foundation model) and three task-specific heads:

- **View classifier** — which view is currently visible (CC/DH/HR/SR for AFAST) — ✅ implemented in 02b
- **Quality head** — is the current frame diagnostic-quality (auto-capture trigger) — 🚧 confidence threshold as MVP, multitask quality head as Phase-2 roadmap
- **Action head** — what probe movement to recommend — 🚧 scripted (4×4 mapping table current→target) in MVP, behavioral cloning + RL as roadmap

Plus an orchestrator (rules + optional small LLM) generating instruction text, and a UI/TTS layer with graphical overlay.

### Defence demo — Level 2

A deliberate simplification for solo feasibility in 7 weeks:

- **Backbone:** TinyUSFM (5.5M params, GitHub `MacDunno/TinyUSFM`) — pretrained on 2M ultrasound images
- **Dataset (primary):** **synthetic AFAST 4 classes** — procedurally generated (numpy/PIL + Rayleigh speckle + primitive anatomical structures). Regulatory rationale: a PoC trained exclusively on synthetic veterinary data **is not classified as a "high-risk AI system"** per EU AI Act Art. 6 + Annex III pt 5(b).
- **Dataset (domain comparison):** FETAL_PLANES_DB as a historical proxy — domain transfer between medical (human) and vet discussed in the thesis.
- **Quality head:** confidence thresholding (max(softmax) with parametric threshold) as MVP
- **Action head:** scripted (4×4=16 pair mapping table for AFAST current→target)
- **UI:** Gradio (single `app.py`)
- **Hosting:** Hugging Face Spaces (free tier)

### Tech stack

| Layer | Tool |
| :--- | :--- |
| IDE | VS Code |
| Compute | Google Colab Pro (T4 GPU) |
| ML framework | PyTorch + Hugging Face Transformers/Datasets |
| Backbone | TinyUSFM (ViT-Tiny, 5.5M params) |
| Dataset hosting | 🤗 Hugging Face Datasets |
| Model hosting | 🤗 Hugging Face Model Hub |
| UI | Gradio |
| Deploy | Hugging Face Spaces |
| Versioning | GitHub (this repo) |
| Demo backup | OBS Studio (MP4) |

**Deliberate "no Drive" decision:** the entire pipeline (code / data / model / demo) is hosted publicly on GitHub + Hugging Face. No private dependencies (Google Drive etc.), full reproducibility for anyone cloning the repo.

### Resources

| What | Where | URL |
| :--- | :--- | :--- |
| Code | GitHub (this repo) | [koscielnamarta/vet-eye-ai-usg-demo](https://github.com/koscielnamarta/vet-eye-ai-usg-demo) |
| Dataset (4,800 images, 3-way split + hard) | HF Datasets | [koscielnamarta/synthetic-usg-afast-vet](https://huggingface.co/datasets/koscielnamarta/synthetic-usg-afast-vet) |
| Fine-tuned model (best_i2 + artifacts) | HF Model Hub | [koscielnamarta/synthetic-usg-afast-vet-classifier](https://huggingface.co/koscielnamarta/synthetic-usg-afast-vet-classifier) |

### Quick start

```python
# 1. Get the dataset (4,800 images: train/val/holdout/hard)
from huggingface_hub import snapshot_download
data_root = snapshot_download(
    repo_id="koscielnamarta/synthetic-usg-afast-vet",
    repo_type="dataset"
)

# 2. Get the trained model
from huggingface_hub import snapshot_download
model_root = snapshot_download(
    repo_id="koscielnamarta/synthetic-usg-afast-vet-classifier",
    repo_type="model"
)
# checkpoint: model_root/checkpoints/best_i2.pt
# artifacts:  model_root/outputs/  (confusion matrices, classification reports, stress test)

# 3. See the notebooks in this repo:
#    notebooks/02_finetune_fetal_planes.ipynb  - FETAL_PLANES_DB baseline
#    notebooks/02b_finetune_synthetic_usg.ipynb - PRIMARY: synthetic AFAST 3-way split
#    notebooks/03_full_pipeline.ipynb           - WIP: full pipeline (confidence + scripted + temporal)
```

### Repo structure

```
vet-eye-ai-usg-demo/
├── notebooks/                      # Jupyter/Colab notebooks (numbered: 01_, 02_, 02b_, 03_)
├── synthetic_usg_dataset/
│   ├── generate_synthetic_usg.py   # generator (4 AFAST classes, --holdout, --verify, --repair)
│   └── README.md                   # dataset description + limitations
├── outputs/                        # training artifacts (confusion matrix, learning curves)
├── app/                            # Gradio application (app.py) — WIP, week 4
├── docs/                           # diagrams, screenshots, thesis excerpts
├── README.md                       # this file
├── LICENSE                         # MIT
└── .gitignore                      # ignored files (cache, raw data, synthetic PNG)
```

### ⚠️ Important disclaimer

This project is an **academic demonstration**, not a medical device. In particular:

- The model is trained on **entirely synthetic** images (procedural numpy/PIL generation) — limiting application to pipeline testing, not clinical diagnosis
- No validation on real animal ultrasound images, no clinical validation, no MDR / FDA / EU AI Act certification
- Clinical consultation with a veterinarian specialising in ultrasound is necessary before drawing any conclusions about clinical utility
- **Do not use for diagnostic decisions** in animals or humans
- All results serve solely to verify technical feasibility (proof of concept) and business reasoning in the BiznesAI 15 thesis

### License

[MIT](LICENSE) — open source. Free to use, modify and distribute, provided the authors and license notice are preserved.
