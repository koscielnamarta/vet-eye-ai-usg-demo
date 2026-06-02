"""
scripted_instructions.py
========================
Tabela instrukcji nawigacyjnych dla protokolu AFAST (4 widoki, psy/koty).

CO TO JEST:
    AFAST (Abdominal Focused Assessment with Sonography for Trauma) to
    protokol ultrasonograficzny uzywany w weterynaryjnych stanach naglych.
    Operator skanuje 4 standardowe okna (widoki) w ustalonym protokole.

    Ten modul zawiera:
    - 16 par instrukcji nawigacyjnych (4 widoki × 4 cele = kazda kombinacja)
    - Instrukcje w dwoch jezykach (PL + EN)
    - Klase AFASTNavigator do uzycia w pipeline i Gradio app

WIDOKI AFAST (psy/koty):
    CC = Cysto-Colic         -- pęcherz moczowy + okrężnica (ogon)
    DH = Diaphragmatico-Hepatic -- przepona + wątroba (czaszkowy)
    HR = Hepato-Renal        -- wątroba + nerka prawa (prawy bok, czaszkowy)
    SR = Spleno-Renal        -- śledziona + nerka lewa (lewy bok, czaszkowy)

ANATOMIA POLOZENIA (potrzebna do weryfikacji instrukcji):
    DH -- najbardziej kranialnie (ku klatce piersiowej)
    HR -- kranialnie, prawy bok
    SR -- kranialnie, lewy bok
    CC -- najbardziej kaudalnie (ku ogonowi), linia srodkowa

STANDARDOWA KOLEJNOSC SKANOWANIA: DH → SR → HR → CC

Autor:  Marta Koscielna / BiznesAI 15 / ALK
Data:   Tydzien 3, 2026-06
Zrodla: Lisciandro 2011 (AFAST protocol), Boysen & Lisciandro 2013
"""

from __future__ import annotations
from typing import Literal

# ---------------------------------------------------------------------------
# Stale
# ---------------------------------------------------------------------------

CLASSES = ["CC", "DH", "HR", "SR"]

Lang = Literal["pl", "en"]

# Opisy widokow (do UI i dokumentacji)
VIEW_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "pl": {
        "CC": "Cysto-Colic (pęcherz moczowy + okrężnica)",
        "DH": "Diaphragmatico-Hepatic (przepona + wątroba)",
        "HR": "Hepato-Renal (wątroba + nerka prawa)",
        "SR": "Spleno-Renal (śledziona + nerka lewa)",
    },
    "en": {
        "CC": "Cysto-Colic (urinary bladder + colon)",
        "DH": "Diaphragmatico-Hepatic (diaphragm + liver)",
        "HR": "Hepato-Renal (liver + right kidney)",
        "SR": "Spleno-Renal (spleen + left kidney)",
    },
}

# Standardowy protokol AFAST — zalecana kolejnosc skanowania
AFAST_SCAN_ORDER = ["DH", "SR", "HR", "CC"]

# ---------------------------------------------------------------------------
# 16 par instrukcji nawigacyjnych: (current, target) -> {pl, en}
# ---------------------------------------------------------------------------
# Logika anatomiczna:
#   - "kranialnie"  = w strone glowy (ku przeponie)
#   - "kaudalnie"   = w strone ogona (ku pecherzowi)
#   - "prawy bok"   = strona prawa pacjenta (leżącego na grzbiecie)
#   - "lewy bok"    = strona lewa pacjenta

NAVIGATION_INSTRUCTIONS: dict[tuple[str, str], dict[str, str]] = {

    # ---- Z CC ----
    ("CC", "CC"): {
        "pl": "Utrzymaj pozycję — widok CC (pęcherz + okrężnica) jest prawidłowy.",
        "en": "Hold position — CC view (bladder + colon) is correct.",
    },
    ("CC", "DH"): {
        "pl": "Przesuń głowicę kranialnie wzdłuż linii środkowej ku przeponie i wątrobie.",
        "en": "Slide the probe cranially along the midline toward the diaphragm and liver.",
    },
    ("CC", "HR"): {
        "pl": "Przesuń głowicę kranialnie i nieznacznie w prawo ku wątrobie i prawej nerce.",
        "en": "Slide the probe cranially and slightly right toward the liver and right kidney.",
    },
    ("CC", "SR"): {
        "pl": "Przesuń głowicę kranialnie i nieznacznie w lewo ku śledzionie i lewej nerce.",
        "en": "Slide the probe cranially and slightly left toward the spleen and left kidney.",
    },

    # ---- Z DH ----
    ("DH", "CC"): {
        "pl": "Przesuń głowicę kaudalnie wzdłuż linii środkowej ku pęcherzowi moczowemu.",
        "en": "Slide the probe caudally along the midline toward the urinary bladder.",
    },
    ("DH", "DH"): {
        "pl": "Utrzymaj pozycję — widok DH (przepona + wątroba) jest prawidłowy.",
        "en": "Hold position — DH view (diaphragm + liver) is correct.",
    },
    ("DH", "HR"): {
        "pl": "Przesuń głowicę kaudalnie i w prawo ku wątrobie i prawej nerce.",
        "en": "Slide the probe caudally and to the right toward the liver and right kidney.",
    },
    ("DH", "SR"): {
        "pl": "Przesuń głowicę kaudalnie i w lewo ku śledzionie i lewej nerce.",
        "en": "Slide the probe caudally and to the left toward the spleen and left kidney.",
    },

    # ---- Z HR ----
    ("HR", "CC"): {
        "pl": "Przesuń głowicę kaudalnie i ku linii środkowej ku pęcherzowi moczowemu.",
        "en": "Slide the probe caudally and toward the midline to reach the urinary bladder.",
    },
    ("HR", "DH"): {
        "pl": "Przesuń głowicę kranialnie i nieznacznie w lewo ku przeponie i wątrobie.",
        "en": "Slide the probe cranially and slightly left toward the diaphragm and liver.",
    },
    ("HR", "HR"): {
        "pl": "Utrzymaj pozycję — widok HR (wątroba + nerka prawa) jest prawidłowy.",
        "en": "Hold position — HR view (liver + right kidney) is correct.",
    },
    ("HR", "SR"): {
        "pl": "Przenieś głowicę na lewy bok brzucha ku śledzionie i lewej nerce.",
        "en": "Move the probe to the left flank toward the spleen and left kidney.",
    },

    # ---- Z SR ----
    ("SR", "CC"): {
        "pl": "Przesuń głowicę kaudalnie i ku linii środkowej ku pęcherzowi moczowemu.",
        "en": "Slide the probe caudally and toward the midline to reach the urinary bladder.",
    },
    ("SR", "DH"): {
        "pl": "Przesuń głowicę kranialnie i nieznacznie w prawo ku przeponie i wątrobie.",
        "en": "Slide the probe cranially and slightly right toward the diaphragm and liver.",
    },
    ("SR", "HR"): {
        "pl": "Przenieś głowicę na prawy bok brzucha ku wątrobie i prawej nerce.",
        "en": "Move the probe to the right flank toward the liver and right kidney.",
    },
    ("SR", "SR"): {
        "pl": "Utrzymaj pozycję — widok SR (śledziona + nerka lewa) jest prawidłowy.",
        "en": "Hold position — SR view (spleen + left kidney) is correct.",
    },
}

# Instrukcja gdy model jest niepewny (max_prob < threshold)
ABSTAIN_INSTRUCTION: dict[str, str] = {
    "pl": "Obraz niepewny — wycentruj głowicę i ustabilizuj pozycję sondy.",
    "en": "Uncertain image — center the probe and stabilize its position.",
}

# ---------------------------------------------------------------------------
# Klasa nawigacyjna
# ---------------------------------------------------------------------------

class AFASTNavigator:
    """
    Orkiestrator tekstowy dla pipeline AFAST.

    Laczy predykcje klasyfikatora (current_view) z celem nawigacyjnym
    (target_view) i zwraca gotowy tekst instrukcji dla operatora.

    Parametry
    ----------
    lang : "pl" lub "en"
        Jezyk instrukcji wyjsciowych.
    scan_order : list[str], opcjonalny
        Kolejnosc skanowania AFAST. Domyslnie: ["DH", "SR", "HR", "CC"].

    Przyklad
    --------
    >>> nav = AFASTNavigator(lang="pl")
    >>> nav.get_instruction(current_view="CC", target_view="DH")
    'Przesuń głowicę kranialnie wzdłuż linii środkowej ku przeponie i wątrobie.'
    >>> nav.get_instruction_auto(current_view="SR", views_done=["DH", "SR"])
    'Przenieś głowicę na prawy bok brzucha ku wątrobie i prawej nerce.'
    """

    def __init__(
        self,
        lang: Lang = "pl",
        scan_order: list[str] | None = None,
    ) -> None:
        if lang not in ("pl", "en"):
            raise ValueError(f"lang musi byc 'pl' lub 'en', dostalam: {lang!r}")
        self.lang = lang
        self.scan_order = scan_order or list(AFAST_SCAN_ORDER)

    # ------------------------------------------------------------------
    # Instrukcja dla konkretnej pary
    # ------------------------------------------------------------------

    def get_instruction(self, current_view: str, target_view: str) -> str:
        """
        Instrukcja dla pary (current_view, target_view).

        Parametry
        ----------
        current_view : str
            Aktualnie wykryty widok ("CC", "DH", "HR" lub "SR").
        target_view : str
            Docelowy widok.

        Zwraca
        -------
        str
            Tekst instrukcji w wybranym jezyku.
        """
        key = (current_view, target_view)
        if key not in NAVIGATION_INSTRUCTIONS:
            raise ValueError(
                f"Nieznana para widokow: {key}. "
                f"Dostepne widoki: {CLASSES}"
            )
        return NAVIGATION_INSTRUCTIONS[key][self.lang]

    def get_abstain_instruction(self) -> str:
        """Instrukcja gdy model jest niepewny (ponizej progu pewnosci)."""
        return ABSTAIN_INSTRUCTION[self.lang]

    # ------------------------------------------------------------------
    # Instrukcja automatyczna (protokol-guided)
    # ------------------------------------------------------------------

    def get_instruction_auto(
        self,
        current_view: str,
        views_done: list[str] | None = None,
    ) -> str:
        """
        Automatyczny wybor celu na podstawie protokolu AFAST.

        Wybiera pierwszy widok z scan_order ktory nie zostal jeszcze
        wykonany. Jesli wszystkie widoki sa zrobione — wraca do poczatku
        (restart pelnego badania AFAST).

        Parametry
        ----------
        current_view : str
            Aktualnie wykryty widok.
        views_done : list[str], opcjonalny
            Lista widokow juz wykonanych w biezacym badaniu.
            Jesli None — zaklada ze nic nie jest zrobione.

        Zwraca
        -------
        str
            Tekst instrukcji.
        """
        views_done = views_done or []
        remaining = [v for v in self.scan_order if v not in views_done]

        if not remaining:
            # Pelny protokol zakonczony — restart
            target = self.scan_order[0]
        elif current_view in remaining:
            # Jestesmy juz na niezrobionym widoku — zostajemy
            target = current_view
        else:
            # Idziemy do pierwszego brakujacego
            target = remaining[0]

        return self.get_instruction(current_view, target)

    def next_view(self, current_view: str) -> str:
        """Nastepny widok w standardowym protokole AFAST."""
        if current_view not in self.scan_order:
            return self.scan_order[0]
        idx = self.scan_order.index(current_view)
        return self.scan_order[(idx + 1) % len(self.scan_order)]

    def get_view_description(self, view: str) -> str:
        """Opis widoku (do UI)."""
        return VIEW_DESCRIPTIONS[self.lang].get(view, view)

    def __repr__(self) -> str:
        return f"AFASTNavigator(lang='{self.lang}', order={self.scan_order})"


# ---------------------------------------------------------------------------
# Szybki podglad tablicy (python src/scripted_instructions.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Tabela instrukcji AFAST 4x4 ===\n")
    for lang in ("pl", "en"):
        nav = AFASTNavigator(lang=lang)
        print(f"--- [{lang.upper()}] ---")
        for current in CLASSES:
            for target in CLASSES:
                instr = nav.get_instruction(current, target)
                marker = " (*)" if current == target else ""
                print(f"  {current} -> {target}{marker}: {instr}")
        print()
    print("Abstain:")
    print(f"  [PL] {ABSTAIN_INSTRUCTION['pl']}")
    print(f"  [EN] {ABSTAIN_INSTRUCTION['en']}")
