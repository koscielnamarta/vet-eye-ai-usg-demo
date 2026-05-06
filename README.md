# Vet Eye AI Navigation USG — demo

> AI-asystowana nawigacja USG dla weterynarii — proof of concept zbudowany w ramach projektu końcowego programu **BiznesAI 15** (Akademia Leona Koźmińskiego).
> Praca dyplomowa: Wiktoria Mikołajów, Małgorzata Polaczuk, Aneta Szurmak, Marta Kościelna, Tomasz Dębowski, Tomasz Fic, Marcin Leszczyński, Artur Charles
Termin obrony: 4–6 lipca 2026.

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
| 2 | 11–17.05.2026 | DataLoader + pierwszy fine-tuning | ⚪ |
| 3 | 18–24.05.2026 | Quality head + scripted instructions | ⚪ |
| 4 | 25–31.05.2026 | UI Gradio + deploy na HF Spaces | ⚪ |
| 5 | 1–7.06.2026 | Sekcje techniczne pracy końcowej | ⚪ |
| 6 | 8–14.06.2026 | Sekcje zmiany + QA | ⚪ |
| 7 | 15–20.06.2026 | Bufor + finalna korekta + złożenie | ⚪ |

### Architektura — Wariant E (hybrydowy)

Pipeline 4–6 modeli AI ze wspólnym backbone (foundation model dla USG) z trzema głowicami zadaniowymi:

- **View classifier** — która projekcja jest aktualnie widoczna na ekranie
- **Quality head** — czy bieżąca klatka jest „diagnostic-quality" (auto-capture trigger)
- **Action head** — jaki ruch głowicy zalecić operatorowi (behavioral cloning + RL)

Plus orkiestrator (reguły + opcjonalnie mały LLM) generujący tekst instrukcji oraz UI/TTS z overlay graficznym.

### Demo na obronę — Poziom 2

Świadome uproszczenie dla wykonalności w 7 tygodni solo:

- **Backbone:** TinyUSFM (5,5 M parametrów, GitHub `MacDunno/TinyUSFM`) — pretrained na 2 mln obrazów USG
- **Dataset:** FETAL_PLANES_DB (Zenodo, 12 400 obrazów, 6 klas) — **proxy dla weterynarii**, bo brak otwartego datasetu vet
- **Quality head:** heurystyka oparta na pewności klasyfikatora (nie trenowana osobno)
- **Action head:** scripted (tabela 6×6 mapowań current→target)
- **UI:** Gradio (jeden plik `app.py`)
- **Hosting:** Hugging Face Spaces (free tier)

### Stack techniczny

| Warstwa | Narzędzie |
| :--- | :--- |
| IDE | VS Code |
| Compute | Google Colab Pro (GPU T4) |
| ML framework | PyTorch + Hugging Face Transformers/Datasets |
| Backbone | TinyUSFM |
| UI | Gradio |
| Deploy | Hugging Face Spaces |
| Wersjonowanie | GitHub (to repo) |
| Backup demo | OBS Studio (MP4) |

### Struktura repo

```
vet-eye-ai-usg-demo/
├── notebooks/        # Jupyter/Colab notebooks (numerowane: 01_, 02_, ...)
├── data/             # placeholder na dane (raw/ ignorowany przez git)
├── app/              # aplikacja Gradio (app.py)
├── docs/             # diagramy, screenshoty, fragmenty pracy
├── README.md         # ten plik
├── LICENSE           # MIT
└── .gitignore        # ignorowane pliki (cache, dane, modele)
```

### Jak uruchomić

> ⚠️ **Status:** kod zacznie działać dopiero od końca Tygodnia 1. Sekcja zostanie zaktualizowana, gdy notebook `01_first_inference.ipynb` zostanie zacommitowany.

### ⚠️ Ważne zastrzeżenie (disclaimer)

Ten projekt to **demonstracja akademicka**, nie wyrób medyczny. W szczególności:

- Model jest trenowany na danych z **medycyny ludzkiej** (FETAL_PLANES_DB) jako proxy — co ogranicza zgeneralizowanie wniosków na weterynarię
- Brak walidacji klinicznej, brak certyfikacji MDR / FDA / AI Act
- **NIE używać do podejmowania decyzji diagnostycznych** u zwierząt ani u ludzi
- Wszelkie wyniki służą wyłącznie do weryfikacji wykonalności technicznej (proof of concept)

### Autor

Marta Kościelna · uczestniczka programu BiznesAI 15 (Akademia Leona Koźmińskiego, edycja 2025/2026)
Kontakt: [GitHub](https://github.com/koscielnamarta)

### Licencja

[MIT](LICENSE) — kod jest open source. Możesz go używać, modyfikować i dystrybuować, pod warunkiem zachowania informacji o autorze i licencji.

---

## 🇬🇧 English

### About the project

A proof-of-concept (PoC) system that analyzes the live feed from an ultrasound probe in real time and instructs the operator how to move the probe to obtain a correct organ view. Direct human-medicine equivalents: **Caption AI** (GE HealthCare) in cardiology and **UltraSight** in cardiac POCUS.

Target customer: **Vet Eye S.A.** — Polish manufacturer of veterinary ultrasound systems (product lines *vet pro-key 75/76*, *vet portable 15*, *vet pro 70*).

### Project status

| Week | Dates | Goal | Status |
| :--- | :--- | :--- | :---: |
| 1 | May 4–10, 2026 | Environment setup + pretrained inference | 🟢 done |
| 2 | May 11–17, 2026 | DataLoader + first fine-tuning | ⚪ |
| 3 | May 18–24, 2026 | Quality head + scripted instructions | ⚪ |
| 4 | May 25–31, 2026 | Gradio UI + HF Spaces deploy | ⚪ |
| 5 | Jun 1–7, 2026 | Thesis technical sections | ⚪ |
| 6 | Jun 8–14, 2026 | Change-management sections + QA | ⚪ |
| 7 | Jun 15–20, 2026 | Buffer + proofreading + submission | ⚪ |

### Architecture — Variant E (hybrid)

A pipeline of 4–6 AI models with a shared backbone (USG foundation model) and three task-specific heads:

- **View classifier** — which view is currently visible on screen
- **Quality head** — is the current frame diagnostic-quality (auto-capture trigger)
- **Action head** — what probe movement to recommend (behavioral cloning + RL)

Plus an orchestrator (rules + optional small LLM) generating instruction text, and a UI/TTS layer with graphical overlay.

### Defence demo — Level 2

A deliberate simplification for solo feasibility in 7 weeks:

- **Backbone:** TinyUSFM (5.5 M params, GitHub `MacDunno/TinyUSFM`) — pretrained on 2 M ultrasound images
- **Dataset:** FETAL_PLANES_DB (Zenodo, 12 400 images, 6 classes) — used as a **proxy for veterinary data**, due to lack of open vet ultrasound datasets
- **Quality head:** heuristic based on classifier confidence (not separately trained)
- **Action head:** scripted (6×6 mapping table for current→target)
- **UI:** Gradio (single `app.py`)
- **Hosting:** Hugging Face Spaces (free tier)

### Tech stack

| Layer | Tool |
| :--- | :--- |
| IDE | VS Code |
| Compute | Google Colab Pro (T4 GPU) |
| ML framework | PyTorch + Hugging Face Transformers/Datasets |
| Backbone | TinyUSFM |
| UI | Gradio |
| Deploy | Hugging Face Spaces |
| Versioning | GitHub (this repo) |
| Demo backup | OBS Studio (MP4) |

### Repo structure

```
vet-eye-ai-usg-demo/
├── notebooks/        # Jupyter/Colab notebooks (numbered: 01_, 02_, ...)
├── data/             # data placeholder (raw/ git-ignored)
├── app/              # Gradio application (app.py)
├── docs/             # diagrams, screenshots, thesis excerpts
├── README.md         # this file
├── LICENSE           # MIT
└── .gitignore        # ignored files (cache, data, models)
```

### How to run

> ⚠️ **Status:** the code becomes runnable at the end of Week 1. This section will be updated once `notebooks/01_first_inference.ipynb` is committed.

### ⚠️ Important disclaimer

This project is an **academic demonstration**, not a medical device. In particular:

- The model is trained on **human medical data** (FETAL_PLANES_DB) as a proxy — which limits the generalizability of conclusions to veterinary medicine
- No clinical validation, no MDR / FDA / EU AI Act certification
- **Do not use for diagnostic decisions** in animals or humans
- All results serve solely to verify technical feasibility (proof of concept)

### Author

Marta Kościelna · participant of the BiznesAI 15 program (Kozminski University, 2025/2026 cohort)
Contact: [GitHub](https://github.com/koscielnamarta)

### License

[MIT](LICENSE) — open source. Free to use, modify and distribute, provided the author and license notice are preserved.
