"""
confidence_thresholder.py
=========================
Moduł post-processingu dla klasyfikatora widoków AFAST (Vet Eye AI USG Navigation).

CO TO ROBI I DLACZEGO:
    Model TinyUSFM fine-tuned na syntetyku (02b) ma dokładność 99.75% na czystych
    obrazach, ale jest "agresywny" — klasyfikuje każdy obraz, nawet kiedy jest
    niepewny. W warunkach realnych (szum, artefakty, zła orientacja sondy) bez
    thresholdingu system dawałby błędne instrukcje nawigacyjne zamiast powiedzieć
    "nie wiem, wycentruj sondę".

    ConfidenceThresholder przyjmuje wartości softmax z sieci i sprawdza:
        max(softmax) >= threshold  →  zwróć nazwę klasy (np. "HR")
        max(softmax) <  threshold  →  zwróć abstain_label (domyślnie "niepewne")

JAK UŻYWAĆ (minimum):
    >>> thresholder = ConfidenceThresholder(threshold=0.85)
    >>> probs = np.array([0.05, 0.06, 0.87, 0.02])  # pewne HR
    >>> thresholder.predict(probs)
    'HR'
    >>> thresholder.predict(np.array([0.28, 0.30, 0.22, 0.20]))
    'niepewne'

KLASY:
    ConfidenceThresholder  – główna klasa post-processingu

UWAGI:
    - Moduł jest NIEZALEŻNY od modelu i datasetu — można testować bez GPU.
    - Klasy AFAST: CC=0, DH=1, HR=2, SR=3 (kolejność z HF Dataset).
    - Do kalibracji progu użyj metody sweep_thresholds() lub notebooka
      04_confidence_threshold_calibration.ipynb.

Autor: Marta Kościelna / BiznesAI 15 / Akademia Leona Koźmińskiego
Data:  Tydzień 3, 2026-06
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


# ===========================================================================
# Główna klasa
# ===========================================================================

class ConfidenceThresholder:
    """
    Post-processing dla predykcji klasyfikatora widoków USG AFAST.

    Parametry
    ----------
    threshold : float
        Globalny próg pewności (0.0–1.0).
        Jeśli max(softmax) < threshold → predykcja = abstain_label.
        Domyślnie 0.85 (wartość z oryginalnego Plan_Prac).
        Do kalibracji użyj sweep_thresholds() lub find_optimal_threshold().
    per_class_thresholds : dict, opcjonalny
        Per-klasowe progi nadpisujące globalny threshold dla danej klasy.
        Przykład: {"CC": 0.80, "SR": 0.90}
        Klasy bez wpisu używają globalnego threshold.
    abstain_label : str
        Etykieta zwracana gdy model jest niepewny.
        Domyślnie "niepewne" (wyświetlana w UI Gradio).
    label_names : list[str], opcjonalny
        Nazwy klas w kolejności odpowiadającej indeksom softmax.
        Domyślnie ["CC", "DH", "HR", "SR"] — kolejność z HF Dataset
        koscielnamarta/synthetic-usg-afast-vet.

    Przykłady
    ---------
    # Podstawowe użycie z globalnym progiem
    >>> t = ConfidenceThresholder(threshold=0.85)
    >>> t.predict(np.array([0.05, 0.06, 0.87, 0.02]))
    'HR'

    # Per-klasowe progi (np. CC dostaje niższy próg bo jest łatwiejsze)
    >>> t2 = ConfidenceThresholder(
    ...     threshold=0.90,
    ...     per_class_thresholds={"CC": 0.75}
    ... )

    # Kalibracja na zestawie walidacyjnym
    >>> results = t.sweep_thresholds(val_probs, val_labels)
    >>> best_t, metrics = t.find_optimal_threshold(val_probs, val_labels)
    """

    # Domyślna kolejność klas — musi zgadzać się z HF Dataset label_names
    DEFAULT_LABEL_NAMES: List[str] = ["CC", "DH", "HR", "SR"]

    def __init__(
        self,
        threshold: float = 0.85,
        per_class_thresholds: Optional[Dict[str, float]] = None,
        abstain_label: str = "niepewne",
        label_names: Optional[List[str]] = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"threshold musi być w zakresie [0.0, 1.0], dostałem: {threshold}"
            )

        self.threshold = threshold
        self.per_class_thresholds: Dict[str, float] = per_class_thresholds or {}
        self.abstain_label = abstain_label
        self.label_names: List[str] = label_names or list(self.DEFAULT_LABEL_NAMES)

        # Walidacja per_class_thresholds — sprawdzamy czy klasy są znane
        unknown_classes = set(self.per_class_thresholds) - set(self.label_names)
        if unknown_classes:
            raise ValueError(
                f"per_class_thresholds zawiera nieznane klasy: {unknown_classes}. "
                f"Znane klasy: {self.label_names}"
            )

        # Walidacja per-klasowych progów
        for cls, thr in self.per_class_thresholds.items():
            if not 0.0 <= thr <= 1.0:
                raise ValueError(
                    f"per_class_threshold dla '{cls}' musi być w [0, 1], "
                    f"dostałem: {thr}"
                )

    # -----------------------------------------------------------------------
    # Predykcja — pojedynczy obraz
    # -----------------------------------------------------------------------

    def predict(self, probs: np.ndarray) -> str:
        """
        Predykcja dla jednego obrazu.

        LOGIKA:
            1. Znajdź klasę z najwyższą wartością softmax (argmax).
            2. Sprawdź próg dla tej klasy (per-klasowy lub globalny).
            3. Jeśli max_prob >= próg → zwróć nazwę klasy.
               Jeśli max_prob <  próg → zwróć abstain_label ("niepewne").

        Parametry
        ----------
        probs : np.ndarray
            Wartości softmax dla jednego obrazu.
            Kształt: (num_classes,) lub (1, num_classes).
            Wartości powinny sumować się do ~1.0, każda w [0, 1].

        Zwraca
        -------
        str
            Nazwa klasy (np. "HR") lub abstain_label (np. "niepewne").
        """
        probs = np.asarray(probs, dtype=float).flatten()

        if probs.shape[0] != len(self.label_names):
            raise ValueError(
                f"Oczekiwano {len(self.label_names)} wartości softmax "
                f"(dla klas {self.label_names}), "
                f"dostałem wektor o długości {probs.shape[0]}"
            )

        best_idx: int = int(np.argmax(probs))
        best_prob: float = float(probs[best_idx])
        predicted_class: str = self.label_names[best_idx]

        # Per-klasowy próg (jeśli zdefiniowany) nadpisuje globalny
        effective_threshold = self.per_class_thresholds.get(
            predicted_class, self.threshold
        )

        if best_prob >= effective_threshold:
            return predicted_class
        else:
            return self.abstain_label

    # -----------------------------------------------------------------------
    # Predykcja wsadowa — N obrazów naraz
    # -----------------------------------------------------------------------

    def predict_batch(self, probs: np.ndarray) -> List[str]:
        """
        Predykcja dla N obrazów (batch processing).

        Parametry
        ----------
        probs : np.ndarray
            Macierz wartości softmax.
            Kształt: (N, num_classes) lub (num_classes,) dla jednego obrazu.

        Zwraca
        -------
        list[str]
            Lista N predykcji. Każda to nazwa klasy lub abstain_label.
        """
        probs = np.asarray(probs, dtype=float)

        # Jeśli podano 1D (pojedynczy obraz) — owijamy w batch
        if probs.ndim == 1:
            probs = probs[np.newaxis, :]

        return [self.predict(row) for row in probs]

    # -----------------------------------------------------------------------
    # Ewaluacja przy bieżącym progu
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        probs: np.ndarray,
        true_labels: Union[np.ndarray, List[int]],
    ) -> Dict:
        """
        Ewaluacja klasyfikatora z bieżącym progiem na zbiorze testowym.

        CO LICZYMY I DLACZEGO:
            - abstain_rate: jak często model odmawia predykcji.
              Zbyt wysoki (>30%) = próg za restrykcyjny (system bezużyteczny).
              Zbyt niski (0%) = próg za liberalny (model klasyfikuje bzdury).
            - accuracy_accepted: poprawność wśród zaakceptowanych predykcji.
              Pokazuje "kiedy model się odzywa, czy ma rację?"
            - accuracy_overall: poprawność liczona z abstain jako błąd.
              Bardziej pesymistyczna metryka (baseline comparison).
            - per_class precision/recall/F1: diagnoza per-klasowa.

        Parametry
        ----------
        probs : np.ndarray, kształt (N, num_classes)
            Macierz softmax dla N obrazów.
        true_labels : array-like[int], kształt (N,)
            Prawdziwe etykiety jako indeksy: 0=CC, 1=DH, 2=HR, 3=SR.

        Zwraca
        -------
        dict z kluczami:
            threshold         : aktualny próg (float)
            n_total           : łączna liczba obrazów (int)
            n_abstained       : ile razy model powiedział "niepewne" (int)
            abstain_rate      : n_abstained / n_total (float, 4 miejsca)
            n_accepted        : n_total - n_abstained (int)
            accuracy_accepted : poprawność na zaakceptowanych (float)
            accuracy_overall  : poprawność ogólna, abstain = błąd (float)
            per_class         : dict {class_name: {tp, fp, fn, precision, recall, f1}}
        """
        probs = np.asarray(probs, dtype=float)
        true_labels = np.asarray(true_labels, dtype=int)
        n_total = len(true_labels)

        if n_total == 0:
            raise ValueError("true_labels jest pusty — brak obrazów do ewaluacji")

        predictions: List[str] = self.predict_batch(probs)
        true_class_names: List[str] = [self.label_names[i] for i in true_labels]

        # ---- Abstain stats ----
        n_abstained = sum(1 for p in predictions if p == self.abstain_label)
        n_accepted = n_total - n_abstained

        # ---- Accuracy (accepted) ----
        correct_accepted = sum(
            1
            for pred, true in zip(predictions, true_class_names)
            if pred != self.abstain_label and pred == true
        )
        accuracy_accepted = correct_accepted / n_accepted if n_accepted > 0 else 0.0

        # ---- Accuracy (overall, abstain = błąd) ----
        correct_overall = sum(
            1 for pred, true in zip(predictions, true_class_names) if pred == true
        )
        accuracy_overall = correct_overall / n_total

        # ---- Per-klasowe precision/recall/F1 ----
        # Abstain jest liczony jako FN dla danej klasy (model nie dał predykcji
        # dla obrazu który należał do tej klasy)
        per_class: Dict[str, Dict] = {}
        for cls in self.label_names:
            tp = sum(
                1 for p, t in zip(predictions, true_class_names)
                if p == cls and t == cls
            )
            fp = sum(
                1 for p, t in zip(predictions, true_class_names)
                if p == cls and t != cls
            )
            fn = sum(
                1 for p, t in zip(predictions, true_class_names)
                if p != cls and t == cls
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )

            per_class[cls] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall":    round(recall, 4),
                "f1":        round(f1, 4),
            }

        return {
            "threshold":          self.threshold,
            "n_total":            n_total,
            "n_abstained":        n_abstained,
            "abstain_rate":       round(n_abstained / n_total, 4),
            "n_accepted":         n_accepted,
            "accuracy_accepted":  round(accuracy_accepted, 4),
            "accuracy_overall":   round(accuracy_overall, 4),
            "per_class":          per_class,
        }

    # -----------------------------------------------------------------------
    # Sweep progów — kalibracja
    # -----------------------------------------------------------------------

    def sweep_thresholds(
        self,
        probs: np.ndarray,
        true_labels: Union[np.ndarray, List[int]],
        thresholds: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Ewaluacja dla całego zakresu progów — do kalibracji.

        Użyj wyników do narysowania krzywej precision/recall/abstain_rate
        jako funkcji progu, żeby wybrać najlepszy threshold dla Twojego
        przypadku użycia (notebook 04_confidence_threshold_calibration.ipynb).

        Parametry
        ----------
        probs : np.ndarray, kształt (N, num_classes)
        true_labels : array-like[int], kształt (N,)
        thresholds : np.ndarray, opcjonalny
            Zakres progów do sprawdzenia.
            Domyślnie: np.arange(0.50, 1.00, 0.01) — 50 punktów.

        Zwraca
        -------
        list[dict]
            Lista słowników (jeden na próg), każdy zawiera:
            threshold, abstain_rate, accuracy_accepted, accuracy_overall,
            macro_f1 (średnia F1 po wszystkich klasach).
        """
        if thresholds is None:
            thresholds = np.arange(0.50, 1.00, 0.01)

        # Zapisujemy oryginalny próg — przywrócimy go po sweepie
        original_threshold = self.threshold
        results = []

        for t in thresholds:
            self.threshold = float(t)
            metrics = self.evaluate(probs, true_labels)
            macro_f1 = float(np.mean([v["f1"] for v in metrics["per_class"].values()]))

            results.append({
                "threshold":         round(float(t), 4),
                "abstain_rate":      metrics["abstain_rate"],
                "accuracy_accepted": metrics["accuracy_accepted"],
                "accuracy_overall":  metrics["accuracy_overall"],
                "macro_f1":          round(macro_f1, 4),
            })

        # Przywracamy oryginalny próg
        self.threshold = original_threshold
        return results

    # -----------------------------------------------------------------------
    # Automatyczny wybór optymalnego progu
    # -----------------------------------------------------------------------

    def find_optimal_threshold(
        self,
        probs: np.ndarray,
        true_labels: Union[np.ndarray, List[int]],
        metric: str = "macro_f1",
        thresholds: Optional[np.ndarray] = None,
        max_abstain_rate: float = 0.20,
    ) -> Tuple[float, Dict]:
        """
        Automatyczny wybór optymalnego progu na zbiorze walidacyjnym.

        STRATEGIA:
            1. Przeiteruj po zakresie progów (sweep_thresholds).
            2. Odfiltruj progi gdzie abstain_rate > max_abstain_rate
               (zbyt restrykcyjny próg = system zbyt często odmawia — niebezpieczny
               dla UX nawigacji USG).
            3. Spośród pozostałych wybierz próg z najlepszą metryką.

        Parametry
        ----------
        probs : np.ndarray, kształt (N, num_classes)
        true_labels : array-like[int], kształt (N,)
        metric : str
            Metryka do maksymalizacji: "macro_f1" lub "accuracy_accepted".
            Domyślnie "macro_f1".
        thresholds : np.ndarray, opcjonalny
        max_abstain_rate : float
            Maksymalny dopuszczalny procent abstained. Domyślnie 0.20 (20%).
            Uzasadnienie: w aplikacji nawigacji USG jeśli model odmawia
            predykcji dla >20% klatek, nawigacja jest bezużyteczna.

        Zwraca
        -------
        Tuple[float, dict]
            (best_threshold, słownik_z_metrykami_dla_tego_progu)
        """
        valid_metrics = {"macro_f1", "accuracy_accepted", "accuracy_overall"}
        if metric not in valid_metrics:
            raise ValueError(
                f"metric musi być jedną z: {valid_metrics}, dostałem: {metric}"
            )

        sweep_results = self.sweep_thresholds(probs, true_labels, thresholds)

        # Filtrujemy po max_abstain_rate
        valid = [r for r in sweep_results if r["abstain_rate"] <= max_abstain_rate]

        if not valid:
            # Żaden próg nie spełnia ograniczenia — bierzemy najniższy próg
            # (= najbardziej liberalny = najmniej abstain)
            valid = [sweep_results[0]]

        # Wybieramy próg z najlepszą wartością metryki
        best = max(valid, key=lambda r: r[metric])
        return best["threshold"], best

    # -----------------------------------------------------------------------
    # Pomocnicze
    # -----------------------------------------------------------------------

    def set_threshold(self, threshold: float) -> None:
        """Zmień globalny próg (np. po kalibracji na val)."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold musi być w [0, 1], dostałem: {threshold}")
        self.threshold = threshold

    def set_per_class_threshold(self, class_name: str, threshold: float) -> None:
        """Ustaw per-klasowy próg dla jednej klasy."""
        if class_name not in self.label_names:
            raise ValueError(
                f"Nieznana klasa: '{class_name}'. Znane: {self.label_names}"
            )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold musi być w [0, 1], dostałem: {threshold}")
        self.per_class_thresholds[class_name] = threshold

    def __repr__(self) -> str:
        per_class_str = (
            f", per_class={self.per_class_thresholds}"
            if self.per_class_thresholds
            else ""
        )
        return (
            f"ConfidenceThresholder("
            f"threshold={self.threshold}"
            f"{per_class_str}, "
            f"labels={self.label_names}, "
            f"abstain='{self.abstain_label}'"
            f")"
        )
