"""
test_confidence_thresholder.py
================================
Testy jednostkowe dla klasy ConfidenceThresholder.

WAŻNE: Testy NIE wymagają modelu, GPU ani datasetu.
Testują wyłącznie logikę klasy — działają natychmiast na każdej maszynie.

Uruchomienie:
    # Opcja 1: przez pytest (jeśli zainstalowany)
    pytest tests/test_confidence_thresholder.py -v

    # Opcja 2: bezpośrednio przez Python (bez zewnętrznych zależności)
    python tests/test_confidence_thresholder.py

    # Opcja 3: z katalogu głównego repo
    python -m pytest tests/ -v

Pokrycie testami (18 testów):
    - predict()              → 7 testów (podstawowe scenariusze)
    - per-class thresholds   → 2 testy
    - predict_batch()        → 2 testy
    - evaluate()             → 3 testy
    - sweep_thresholds()     → 2 testy
    - find_optimal_threshold → 2 testy
"""

import sys
import os
import numpy as np

# Dodajemy src/ do path żeby importować bez instalacji pakietu
# Działa zarówno z katalogu głównego repo jak i z katalogu tests/
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_root, "src"))

from confidence_thresholder import ConfidenceThresholder


# ===========================================================================
# Helper — tworzenie obiektu z domyślnymi ustawieniami
# ===========================================================================

def make_t(**kwargs) -> ConfidenceThresholder:
    """Shortcut: ConfidenceThresholder z label_names dla AFAST."""
    return ConfidenceThresholder(
        label_names=["CC", "DH", "HR", "SR"],
        **kwargs
    )


# ===========================================================================
# predict() — 7 testów
# ===========================================================================

def test_predict_pewny_zwraca_klase():
    """Gdy max_prob > threshold → zwraca nazwę klasy."""
    t = make_t(threshold=0.85)
    # max = 0.87 (indeks 2 = HR) > 0.85 → HR
    probs = np.array([0.05, 0.06, 0.87, 0.02])
    assert t.predict(probs) == "HR", "Pewna predykcja powinna zwrócić 'HR'"


def test_predict_niepewny_zwraca_abstain():
    """Gdy max_prob < threshold → zwraca abstain_label."""
    t = make_t(threshold=0.85)
    # max = 0.30 < 0.85 → "niepewne"
    probs = np.array([0.28, 0.30, 0.22, 0.20])
    assert t.predict(probs) == "niepewne", \
        "Niska pewność powinna zwrócić 'niepewne'"


def test_predict_dokladnie_na_progu_przechodzi():
    """max_prob == threshold → zaakceptowany (operacja >=, nie >)."""
    t = make_t(threshold=0.85)
    # max = 0.85 (CC) == 0.85 → CC (>= threshold)
    probs = np.array([0.85, 0.05, 0.05, 0.05])
    assert t.predict(probs) == "CC", \
        "Dokładnie na progu powinno przejść (operator >=)"


def test_predict_tuz_ponizej_progu_abstain():
    """max_prob tuż poniżej threshold → abstain."""
    t = make_t(threshold=0.85)
    # max = 0.8499 < 0.85 → niepewne
    probs = np.array([0.8499, 0.05, 0.05, 0.0501])
    assert t.predict(probs) == "niepewne", \
        "Tuż poniżej progu powinno abstain"


def test_predict_niestandardowy_abstain_label():
    """Niestandardowa etykieta abstain jest zwracana poprawnie."""
    t = make_t(threshold=0.90, abstain_label="NISKA_PEWNOSC")
    probs = np.array([0.50, 0.20, 0.20, 0.10])  # max=0.50 < 0.90
    assert t.predict(probs) == "NISKA_PEWNOSC"


def test_predict_prog_zero_zawsze_klasyfikuje():
    """Próg = 0.0 → model zawsze daje predykcję, nigdy abstain."""
    t = make_t(threshold=0.0)
    # Nawet prawie uniform distribution → powinno zwrócić klasę z max
    probs = np.array([0.001, 0.001, 0.001, 0.997])
    result = t.predict(probs)
    assert result == "SR", f"Przy threshold=0 powinno zwrócić 'SR', dostałem '{result}'"
    assert result != "niepewne", "Przy threshold=0 nigdy nie powinno abstain"


def test_predict_zly_ksztalt_rzuca_valueerror():
    """Niepoprawna liczba wartości softmax → ValueError."""
    t = make_t()
    # Podajemy 2 wartości zamiast 4 (mamy 4 klasy AFAST)
    try:
        t.predict(np.array([0.5, 0.5]))
        assert False, "Powinien rzucić ValueError"
    except ValueError as e:
        assert "4" in str(e) or "length" in str(e).lower() or "długości" in str(e), \
            f"Komunikat błędu powinien wspominać o liczbie klas, dostałem: {e}"


# ===========================================================================
# per_class_thresholds — 2 testy
# ===========================================================================

def test_per_class_prog_nizszy_niz_globalny():
    """
    Per-klasowy próg CC=0.70 < globalny 0.90.
    → CC z max_prob=0.75 przechodzi, DH z max_prob=0.75 abstain.
    """
    t = make_t(threshold=0.90, per_class_thresholds={"CC": 0.70})

    probs_cc = np.array([0.75, 0.10, 0.10, 0.05])  # max=0.75 (CC)
    assert t.predict(probs_cc) == "CC", \
        "0.75 >= per-class threshold CC=0.70, powinno zwrócić CC"

    probs_dh = np.array([0.05, 0.75, 0.10, 0.10])  # max=0.75 (DH)
    assert t.predict(probs_dh) == "niepewne", \
        "0.75 < globalny threshold 0.90, DH powinno abstain"


def test_per_class_nieznana_klasa_rzuca_valueerror():
    """Nieznana klasa w per_class_thresholds → ValueError przy inicjalizacji."""
    try:
        ConfidenceThresholder(
            threshold=0.85,
            per_class_thresholds={"NIEZNANA_KLASA": 0.80},
            label_names=["CC", "DH", "HR", "SR"]
        )
        assert False, "Powinien rzucić ValueError"
    except ValueError as e:
        assert "NIEZNANA_KLASA" in str(e), \
            f"Komunikat powinien wspominać nieznana klasę, dostałem: {e}"


# ===========================================================================
# predict_batch() — 2 testy
# ===========================================================================

def test_predict_batch_miesza_klasy_i_abstain():
    """Batch 3 obrazów: pewny CC, abstain, pewny SR."""
    t = make_t(threshold=0.85)
    probs = np.array([
        [0.90, 0.04, 0.04, 0.02],   # CC, max=0.90 > 0.85 → CC
        [0.30, 0.25, 0.25, 0.20],   # max=0.30 < 0.85 → niepewne
        [0.01, 0.01, 0.01, 0.97],   # SR, max=0.97 > 0.85 → SR
    ])
    result = t.predict_batch(probs)
    assert result == ["CC", "niepewne", "SR"], \
        f"Oczekiwano ['CC', 'niepewne', 'SR'], dostałem {result}"


def test_predict_batch_akceptuje_1d_input():
    """predict_batch powinno działać z 1D input (pojedynczy obraz)."""
    t = make_t(threshold=0.85)
    probs_1d = np.array([0.90, 0.04, 0.04, 0.02])
    result = t.predict_batch(probs_1d)
    assert isinstance(result, list) and len(result) == 1, \
        "1D input powinien dać listę z jednym elementem"
    assert result[0] == "CC"


# ===========================================================================
# evaluate() — 3 testy
# ===========================================================================

def test_evaluate_wszystko_poprawne_bez_abstain():
    """Wszystkie obrazy powyżej progu i poprawne → abstain_rate=0, acc=1.0."""
    t = make_t(threshold=0.80)
    # 4 obrazy, każdy pewny i poprawny
    probs = np.array([
        [0.90, 0.04, 0.04, 0.02],   # CC
        [0.02, 0.91, 0.04, 0.03],   # DH
        [0.02, 0.04, 0.92, 0.02],   # HR
        [0.02, 0.03, 0.04, 0.91],   # SR
    ])
    true_labels = [0, 1, 2, 3]  # CC, DH, HR, SR

    result = t.evaluate(probs, true_labels)

    assert result["abstain_rate"] == 0.0, "Brak abstain przy pewnych predykcjach"
    assert result["accuracy_accepted"] == 1.0, "Wszystkie poprawne → acc=1.0"
    assert result["n_abstained"] == 0


def test_evaluate_abstain_liczy_sie_jako_blad_w_overall():
    """
    Abstained → accuracy_overall go liczy jako błąd,
    ale accuracy_accepted ignoruje.
    """
    t = make_t(threshold=0.85)
    probs = np.array([
        [0.90, 0.04, 0.04, 0.02],   # CC, pewny, poprawny → zaakceptowany
        [0.30, 0.25, 0.25, 0.20],   # niepewny → abstain (prawdziwa klasa DH)
    ])
    true_labels = [0, 1]  # CC, DH

    result = t.evaluate(probs, true_labels)

    assert result["n_abstained"] == 1,       "1 obraz poniżej progu"
    assert result["n_accepted"] == 1,        "1 obraz zaakceptowany"
    assert result["accuracy_accepted"] == 1.0, "Zaakceptowany był poprawny"
    assert result["accuracy_overall"] == 0.5,  "Ogółem: 1/2 (abstain = błąd)"


def test_evaluate_zawiera_per_class_metryki():
    """evaluate() musi zwracać per_class z precision/recall/f1."""
    t = make_t(threshold=0.0)  # threshold=0 → nigdy nie abstain
    probs = np.array([
        [0.90, 0.04, 0.04, 0.02],   # CC → CC
        [0.02, 0.91, 0.04, 0.03],   # DH → DH
    ])
    result = t.evaluate(probs, [0, 1])

    # Sprawdzamy strukturę
    assert "per_class" in result
    for cls in ["CC", "DH", "HR", "SR"]:
        assert cls in result["per_class"], f"Brak klasy '{cls}' w per_class"
        for key in ("precision", "recall", "f1"):
            assert key in result["per_class"][cls], \
                f"Brak klucza '{key}' dla klasy '{cls}'"
            val = result["per_class"][cls][key]
            assert 0.0 <= val <= 1.0, \
                f"{key} dla {cls} powinno być w [0,1], dostałem {val}"


# ===========================================================================
# sweep_thresholds() — 2 testy
# ===========================================================================

def test_sweep_thresholds_zwraca_liste_slownikow():
    """sweep_thresholds() zwraca listę z jednym elementem na próg."""
    t = make_t()
    probs = np.tile(np.array([0.90, 0.04, 0.04, 0.02]), (10, 1))
    true_labels = [0] * 10

    # Testujemy 3 progi
    results = t.sweep_thresholds(
        probs, true_labels,
        thresholds=np.array([0.5, 0.7, 0.9])
    )

    assert len(results) == 3, f"3 progi → 3 wyniki, dostałem {len(results)}"
    for r in results:
        for key in ("threshold", "abstain_rate", "accuracy_accepted",
                    "accuracy_overall", "macro_f1"):
            assert key in r, f"Brak klucza '{key}' w wynikach sweep"


def test_sweep_nie_zmienia_oryginalnego_progu():
    """sweep_thresholds() przywraca self.threshold po zakończeniu."""
    t = make_t(threshold=0.85)
    probs = np.tile(np.array([0.90, 0.04, 0.04, 0.02]), (5, 1))

    t.sweep_thresholds(probs, [0] * 5, thresholds=np.array([0.5, 0.6, 0.7]))

    assert t.threshold == 0.85, \
        f"threshold powinien zostać 0.85 po sweep, jest {t.threshold}"


# ===========================================================================
# find_optimal_threshold() — 2 testy
# ===========================================================================

def test_find_optimal_threshold_zwraca_float_i_dict():
    """find_optimal_threshold() zwraca (float, dict)."""
    t = make_t()
    # 20 obrazów, wszystkie CC z high confidence
    probs = np.tile(np.array([0.88, 0.04, 0.04, 0.04]), (20, 1))
    true_labels = [0] * 20

    best_t, best_metrics = t.find_optimal_threshold(probs, true_labels)

    assert isinstance(best_t, float), "best_threshold powinien być float"
    assert isinstance(best_metrics, dict), "best_metrics powinien być dict"
    assert "threshold" in best_metrics, "dict powinien zawierać 'threshold'"


def test_find_optimal_respektuje_max_abstain_rate():
    """find_optimal nie przekracza max_abstain_rate."""
    t = make_t()

    # Połowa obrazów: max_prob=0.50 (abstain przy progach > 0.50)
    # Połowa obrazów: max_prob=0.95 (pewne CC)
    probs_low  = np.tile(np.array([0.50, 0.17, 0.17, 0.16]), (5, 1))
    probs_high = np.tile(np.array([0.95, 0.02, 0.02, 0.01]), (5, 1))
    probs = np.vstack([probs_low, probs_high])
    true_labels = [0] * 10

    best_t, best_metrics = t.find_optimal_threshold(
        probs, true_labels,
        max_abstain_rate=0.30   # max 30% abstain
    )

    assert best_metrics["abstain_rate"] <= 0.30, \
        (f"abstain_rate={best_metrics['abstain_rate']} przekracza "
         f"max_abstain_rate=0.30")


# ===========================================================================
# Runner — działa bez pytest
# ===========================================================================

if __name__ == "__main__":
    # Zbieramy wszystkie funkcje testowe (te zaczynające się od "test_")
    test_functions = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]

    passed = 0
    failed = 0
    errors = []

    print(f"\n{'='*60}")
    print(f"  Testy ConfidenceThresholder ({len(test_functions)} testów)")
    print(f"{'='*60}\n")

    for name, fn in test_functions:
        try:
            fn()
            print(f"  ✓  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗  {name}")
            print(f"       AssertionError: {e}")
            failed += 1
            errors.append((name, str(e)))
        except Exception as e:
            print(f"  ✗  {name}")
            print(f"       {type(e).__name__}: {e}")
            failed += 1
            errors.append((name, f"{type(e).__name__}: {e}"))

    print(f"\n{'='*60}")
    print(f"  Wynik: {passed}/{len(test_functions)} zaliczonych", end="")
    if failed == 0:
        print("  ✓ WSZYSTKIE PRZESZŁY")
    else:
        print(f"  ✗ {failed} NIEUDANYCH")
        print(f"\n  Szczegóły błędów:")
        for name, msg in errors:
            print(f"    - {name}: {msg}")
    print(f"{'='*60}\n")

    sys.exit(0 if failed == 0 else 1)
