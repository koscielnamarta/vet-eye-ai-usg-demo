# Syntetyczny dataset USG - protokół AFAST/POCUS (psy + koty)

**Eksperymentalny zbiór syntetycznych obrazów USG** wygenerowanych proceduralnie do
testowania pipeline'u treningowego modelu klasyfikującego okna AFAST (Abdominal
Focused Assessment with Sonography for Trauma) w weterynarii.

## ⚠️ WAŻNE OGRANICZENIA — przeczytaj zanim użyjesz do treningu

To **NIE są prawdziwe obrazy USG zwierząt**. Zostały wygenerowane algorytmicznie
przy pomocy numpy/PIL — stylizowane geometrycznie struktury + szum speckle (Rayleigh).
Cel: sanity check pipeline'u, test architektury, augmentacja syntetycznego boosta
dla małego prawdziwego zbioru. **Model wytrenowany wyłącznie na tych danych nie
będzie działał klinicznie.**

Konkretnie czego brakuje vs realne USG:
- brak prawdziwej anatomii zwierząt (różnice między psem a kotem, rasy, wiek, BCS)
- uproszczone artefakty (cienie/wzmocnienia akustyczne są schematyczne)
- brak ruchu (oddech, perystaltyka), brak Dopplera, brak pomiarów/etykiet w obrazie
- brak typowych nakładek z urządzenia (logo, MI/TI, marker pozycji głowicy)
- brak patologii (wolny płyn, krwiak — typowe znaleziska AFAST nie są reprezentowane)
- brak różnorodności sond (curvilinear vs phased array vs linear)

**Konsultacja kliniczna z lek. wet. specjalizującym się w USG jest konieczna**
przed jakimkolwiek wnioskowaniem o klinicznej przydatności modelu.

## Struktura

```
dataset/
├── train/
│   ├── DH/   (800 obrazów)
│   ├── SR/   (800 obrazów)
│   ├── CC/   (800 obrazów)
│   └── HR/   (800 obrazów)
├── test/
│   ├── DH/   (200 obrazów)
│   ├── SR/   (200 obrazów)
│   ├── CC/   (200 obrazów)
│   └── HR/   (200 obrazów)
└── split.csv  (4000 wierszy: filename, class, split)
```

**Łącznie: 4000 obrazów PNG 256×256 grayscale, ~126 MB.**
**Split: 80/20 train/test, zbalansowany po klasach.**

## Klasy (okna AFAST)

| Kod | Pełna nazwa                | Co reprezentuje                                  |
|-----|----------------------------|--------------------------------------------------|
| DH  | Diaphragmatico-Hepatic     | okno przeponowo-wątrobowe (subxiphoid)           |
| SR  | Spleno-Renal               | okno śledzionowo-nerkowe (lewe brzuszne)         |
| CC  | Cysto-Colic                | okno pęcherz-okrężnica (nadłonowe)               |
| HR  | Hepato-Renal               | okno wątrobowo-nerkowe (prawe brzuszne)          |

## Sposób użycia

### PyTorch (ImageFolder)

```python
from torchvision.datasets import ImageFolder
from torchvision import transforms

tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # dla modeli ImageNet
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_ds = ImageFolder("dataset/train", transform=tfm)
test_ds  = ImageFolder("dataset/test",  transform=tfm)
# klasy automatycznie: CC=0, DH=1, HR=2, SR=3 (alfabetycznie)
```

### HuggingFace `datasets`

```python
from datasets import load_dataset
ds = load_dataset("imagefolder", data_dir="dataset")
# ds["train"], ds["test"]
```

### Z CSV (dla custom Dataset)

```python
import pandas as pd
df = pd.read_csv("dataset/split.csv")
train_df = df[df["split"] == "train"]
test_df  = df[df["split"] == "test"]
```

## Regenerowanie / rozszerzanie

Wszystko deterministyczne (`SEED=42` w `generate_synthetic_usg.py`).

```bash
# pełny rerun (skip jeśli plik istnieje)
python generate_synthetic_usg.py

# tylko jedna klasa
python generate_synthetic_usg.py CC

# wiele klas
python generate_synthetic_usg.py DH SR

# regeneracja samego CSV (po dodaniu/usunięciu plików)
python generate_synthetic_usg.py --csv
```

Zmiana liczby obrazów / proporcji split: edytuj stałe `PER_CLASS`, `TRAIN_RATIO`
u góry skryptu.

## Pipeline techniczny generacji (skrótowo)

1. Pusty canvas (mid-dark ~40, typowe tło tkanki miękkiej)
2. Anatomia specyficzna dla klasy (elipsy z gradientem + krzywe)
3. Lekkie rozmycie Gaussa (imitacja point spread function wiązki USG)
4. Multiplikatywny szum **Rayleigh** (klasyczny model speckle dla USG)
5. Addytywny szum Gaussa (elektronika urządzenia)
6. Gamma correction (kontrast typowy dla USG)
7. Maska sektora (stożek głowicy sektorowej)
8. Znaczniki głębokości po bokach (jak na realnym aparacie)

## Co dalej

Sugerowane następne kroki (od najtańszego do najdroższego eksperymentu):

1. **Sanity baseline** — wytrenować prosty CNN (ResNet18 pretrained) na tym zbiorze,
   oczekiwać >95% test_acc (klasy są wizualnie bardzo odróżnialne). Jeśli model
   się nie uczy — bug w pipeline'ie, nie w danych.
2. **Domain adaptation** — pretrain na syntetyce, fine-tune na małym prawdziwym zbiorze.
3. **Augmentacja prawdziwego zbioru** — mieszać syntetykę z realnymi obrazami w batchach.
4. **Realistic upgrade** — zastąpić strukturalne elipsy małymi crop'ami z realnych USG
   (jeśli będziemy mieli choćby kilka prawdziwych obrazów per klasa).
