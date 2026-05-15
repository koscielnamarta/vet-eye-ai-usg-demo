"""
Generator syntetycznych obrazow USG dla psow i kotow - protokol AFAST/POCUS.

Generuje 4 klasy okien AFAST (Abdominal Focused Assessment with Sonography for Trauma):
- DH: Diaphragmatico-Hepatic (okno przeponowo-watrobowe)
- SR: Spleno-Renal (okno sledzionowo-nerkowe)
- CC: Cysto-Colic (okno pecherz-okreznica)
- HR: Hepato-Renal (okno watrobowo-nerkowe)

Charakterystyka procedury:
- Stozek/sektor USG (typowy dla glowicy sektorowej)
- Multiplikatywny szum speckle (rozklad Rayleigha) - klasyczna cecha USG
- Hipo/hiperechogeniczne struktury jako wypelnione kszt(elipsy/wielokaty) z gradientem
- Cienie akustyczne za strukturami silnie odbijajacymi
- Wzmocnienie akustyczne za strukturami ciekymi (np. pecherz)

UWAGA: To sa SYNTETYCZNE dane proceduralne - wygladaja jak "stylizowane USG",
nie zastapia prawdziwych zdjec. Sluza do testu pipeline'u, augmentacji,
i sanity check architektury modelu.
"""

import os
import csv
import math
import random
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from pathlib import Path

# MIN_VALID_PNG_BYTES - prog "minimalnego sensownego pliku PNG".
# Nasze obrazki 256x256 grayscale po kompresji to zwykle 20-60 kB.
# Plik ponizej 1 kB to prawie na pewno truncated/zerobajtowy.
MIN_VALID_PNG_BYTES = 1000

# ------------------------------------------------------------------
# KONFIGURACJA
# ------------------------------------------------------------------
IMG_SIZE = 256
PER_CLASS = 1000           # obrazow na klase
TRAIN_RATIO = 0.8          # 80/20 split
CLASSES = ["DH", "SR", "CC", "HR"]
SEED = 42

OUTPUT_DIR = Path(__file__).parent / "dataset"

random.seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------------
# MASKA SEKTORA (stozek USG)
# ------------------------------------------------------------------
def make_sector_mask(size=IMG_SIZE, apex_y=10, half_angle_deg=32, depth_ratio=0.95):
    """Tworzy maske stozka USG (binarna) - poza maska piksele beda czarne."""
    yy, xx = np.mgrid[0:size, 0:size]
    cx = size / 2
    dx = xx - cx
    dy = yy - apex_y
    angle = np.degrees(np.arctan2(dx, dy))   # kat wzgledem osi pionowej
    r = np.sqrt(dx ** 2 + dy ** 2)
    in_angle = np.abs(angle) <= half_angle_deg
    in_depth = (r >= 5) & (r <= size * depth_ratio)
    return (in_angle & in_depth).astype(np.float32)


SECTOR_MASK = make_sector_mask()


# ------------------------------------------------------------------
# PODSTAWOWE PRYMITYWY ANATOMICZNE
# ------------------------------------------------------------------
def add_ellipse(canvas, cx, cy, rx, ry, angle_deg, intensity, soft=True):
    """Wpisuje wypelniona elipse (z miekkim brzegiem) o zadanej intensywnosci."""
    h, w = canvas.shape
    yy, xx = np.mgrid[0:h, 0:w]
    a = math.radians(angle_deg)
    xr = (xx - cx) * math.cos(a) + (yy - cy) * math.sin(a)
    yr = -(xx - cx) * math.sin(a) + (yy - cy) * math.cos(a)
    dist = (xr / rx) ** 2 + (yr / ry) ** 2
    if soft:
        # Gradient od centrum (1.0) do brzegu (0.0), z lekkim "wycieczeniem"
        weight = np.clip(1.2 - dist, 0, 1)
    else:
        weight = (dist <= 1.0).astype(np.float32)
    canvas[:] = canvas * (1 - weight) + intensity * weight


def add_curve_line(canvas, points, thickness, intensity):
    """Rysuje krzywa (sekwencje punktow) o danej grubosci - np. przepona, sciana naczynia."""
    img_pil = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img_pil)
    draw.line(points, fill=int(intensity), width=thickness)
    canvas[:] = np.array(img_pil).astype(np.float32)


def add_acoustic_shadow(canvas, x_center, x_half_width, y_start, intensity_factor=0.35):
    """Dodaje pionowy cien akustyczny (kolumna ciemniejsza) za struktura odbijajaca."""
    h, w = canvas.shape
    yy, xx = np.mgrid[0:h, 0:w]
    column = (np.abs(xx - x_center) <= x_half_width) & (yy > y_start)
    canvas[column] *= intensity_factor


def add_acoustic_enhancement(canvas, x_center, x_half_width, y_start, intensity_factor=1.6):
    """Wzmocnienie akustyczne - jasniejsza kolumna za struktura ciekla (np. pecherz)."""
    h, w = canvas.shape
    yy, xx = np.mgrid[0:h, 0:w]
    column = (np.abs(xx - x_center) <= x_half_width) & (yy > y_start)
    canvas[column] = np.clip(canvas[column] * intensity_factor, 0, 230)


# ------------------------------------------------------------------
# GENERATORY KLAS (kazda klasa ma swoja anatomie)
# ------------------------------------------------------------------
def gen_DH(canvas):
    """Diaphragmatico-Hepatic: przepona (jasny luk) + watroba (jednorodna szara)."""
    # Watroba - duzy obszar mid-grey w dolnej polowie
    add_ellipse(canvas, IMG_SIZE/2 + random.randint(-20, 20),
                IMG_SIZE*0.65 + random.randint(-15, 15),
                rx=random.randint(70, 100), ry=random.randint(55, 80),
                angle_deg=random.uniform(-15, 15),
                intensity=random.uniform(75, 105))
    # Drobne struktury naczyniowe w watrobie (hipoechogeniczne kropki)
    for _ in range(random.randint(2, 5)):
        add_ellipse(canvas,
                    IMG_SIZE/2 + random.randint(-50, 50),
                    IMG_SIZE*0.65 + random.randint(-30, 30),
                    rx=random.randint(3, 8), ry=random.randint(3, 8),
                    angle_deg=0, intensity=random.uniform(20, 45))
    # Przepona - jasny zakrzywiony luk u gory
    points = []
    y_base = random.randint(70, 95)
    for x in range(40, IMG_SIZE - 40, 4):
        y = y_base + int(8 * math.sin((x - 40) / 50))
        points.append((x, y))
    add_curve_line(canvas, points, thickness=random.randint(3, 5),
                   intensity=random.uniform(190, 230))
    # Lekkie odbicie lustrzane nad przepona (artefakt)
    add_ellipse(canvas, IMG_SIZE/2, y_base - 25,
                rx=60, ry=15, angle_deg=0, intensity=random.uniform(40, 60))


def gen_SR(canvas):
    """Spleno-Renal: sledziona (trojkat) + nerka (bean-shape z medulla)."""
    # Sledziona - elongowana, jasniejsza, po lewej
    add_ellipse(canvas,
                random.randint(70, 100), random.randint(120, 150),
                rx=random.randint(50, 70), ry=random.randint(20, 30),
                angle_deg=random.uniform(-25, -10),
                intensity=random.uniform(110, 140))
    # Nerka - bean shape (dwie elipsy) po prawej
    kid_cx = random.randint(155, 185)
    kid_cy = random.randint(140, 170)
    # Cortex (zewnetrzna warstwa, mid-grey)
    add_ellipse(canvas, kid_cx, kid_cy,
                rx=random.randint(35, 45), ry=random.randint(25, 35),
                angle_deg=random.uniform(-10, 10),
                intensity=random.uniform(85, 110))
    # Medulla - 2-3 ciemniejsze piramidy w srodku
    for i in range(random.randint(2, 4)):
        offset = (i - 1) * 12
        add_ellipse(canvas, kid_cx + offset, kid_cy + random.randint(-3, 3),
                    rx=5, ry=8, angle_deg=0,
                    intensity=random.uniform(40, 60))
    # Renal pelvis - mala jasna kropka w centrum nerki
    add_ellipse(canvas, kid_cx, kid_cy,
                rx=random.randint(3, 6), ry=random.randint(3, 6),
                angle_deg=0, intensity=random.uniform(180, 220))


def gen_CC(canvas):
    """Cysto-Colic: pecherz (duzy anechogenny owal) + wzmocnienie akustyczne + okreznica z gazem."""
    # Pecherz - duzy ciemny owal z jasna sciana
    bx = IMG_SIZE/2 + random.randint(-15, 15)
    by = random.randint(120, 150)
    brx = random.randint(55, 75)
    bry = random.randint(40, 55)
    # Sciana
    add_ellipse(canvas, bx, by, rx=brx + 3, ry=bry + 3,
                angle_deg=0, intensity=random.uniform(150, 190))
    # Wnetrze - prawie czarne (plyn)
    add_ellipse(canvas, bx, by, rx=brx, ry=bry,
                angle_deg=0, intensity=random.uniform(10, 25), soft=False)
    # Wzmocnienie akustyczne za pecherzem
    add_acoustic_enhancement(canvas, bx, brx * 0.85, by + bry,
                             intensity_factor=random.uniform(1.4, 1.8))
    # Okreznica z gazem - jasny luk z cieniem akustycznym
    if random.random() < 0.7:
        cx = random.choice([random.randint(50, 80), random.randint(180, 210)])
        cy = random.randint(90, 120)
        add_ellipse(canvas, cx, cy, rx=random.randint(15, 25), ry=8,
                    angle_deg=random.uniform(-20, 20),
                    intensity=random.uniform(200, 240))
        add_acoustic_shadow(canvas, cx, 12, cy + 5,
                            intensity_factor=random.uniform(0.25, 0.45))


def gen_HR(canvas):
    """Hepato-Renal: watroba przylegajaca do nerki (Morison's pouch w humanach, analogicznie u psow/kotow)."""
    # Watroba - dolna polowa, jednorodna szara
    add_ellipse(canvas,
                random.randint(80, 110), random.randint(140, 170),
                rx=random.randint(55, 75), ry=random.randint(45, 60),
                angle_deg=random.uniform(-20, 5),
                intensity=random.uniform(75, 100))
    # Naczynia watrobowe
    for _ in range(random.randint(1, 3)):
        add_ellipse(canvas,
                    random.randint(60, 130), random.randint(140, 170),
                    rx=random.randint(2, 6), ry=random.randint(2, 6),
                    angle_deg=0, intensity=random.uniform(20, 40))
    # Nerka po prawej, lekko nizej
    kid_cx = random.randint(160, 190)
    kid_cy = random.randint(150, 180)
    add_ellipse(canvas, kid_cx, kid_cy,
                rx=random.randint(30, 42), ry=random.randint(22, 32),
                angle_deg=random.uniform(0, 20),
                intensity=random.uniform(80, 105))
    # Medulla
    for i in range(random.randint(2, 3)):
        offset = (i - 1) * 12
        add_ellipse(canvas, kid_cx + offset, kid_cy,
                    rx=5, ry=7, angle_deg=0,
                    intensity=random.uniform(40, 60))
    # Renal pelvis
    add_ellipse(canvas, kid_cx, kid_cy,
                rx=random.randint(2, 5), ry=random.randint(2, 5),
                angle_deg=0, intensity=random.uniform(180, 220))
    # Granica watroba-nerka (cienka linia jasna)
    if random.random() < 0.5:
        y_line = (kid_cy + 140) // 2
        points = [(110, y_line + random.randint(-3, 3)),
                  (140, y_line + random.randint(-3, 3)),
                  (165, y_line + random.randint(-3, 3))]
        add_curve_line(canvas, points, thickness=2,
                       intensity=random.uniform(160, 200))


GENERATORS = {"DH": gen_DH, "SR": gen_SR, "CC": gen_CC, "HR": gen_HR}


# ------------------------------------------------------------------
# PIPELINE GENERACJI POJEDYNCZEGO OBRAZU
# ------------------------------------------------------------------
def generate_image(class_name):
    """Tworzy jeden obraz USG dla danej klasy."""
    # 1. Pusty canvas (mid-dark - typowe tlo tkanki miekkiej)
    base_bg = random.uniform(35, 55)
    canvas = np.full((IMG_SIZE, IMG_SIZE), base_bg, dtype=np.float32)

    # 2. Anatomia specyficzna dla klasy
    GENERATORS[class_name](canvas)

    # 3. Lekkie rozmycie - imituje rozmycie wiazki USG (point spread function)
    canvas_pil = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    canvas_pil = canvas_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.0)))
    canvas = np.array(canvas_pil).astype(np.float32)

    # 4. Speckle noise - Rayleigh, multiplikatywnie (klasyczny model USG)
    speckle = np.random.rayleigh(scale=0.7, size=canvas.shape)
    # Normalizacja zeby srednia speckle byla ~1.0
    speckle = speckle / np.mean(speckle)
    canvas = canvas * speckle

    # 5. Dodatkowy szum gaussa (elektronika urzadzenia)
    canvas += np.random.normal(0, random.uniform(2, 6), canvas.shape)

    # 6. Gamma correction (kontrast typowy dla USG)
    canvas = np.clip(canvas, 0, 255)
    gamma = random.uniform(0.85, 1.15)
    canvas = 255 * np.power(canvas / 255, gamma)

    # 7. Maska sektora - poza stozkiem czarne tlo
    sector = SECTOR_MASK
    canvas = canvas * sector

    # 8. Cienkie linie znacznikow glebokosci po bokach (czesto sa na USG)
    img_pil = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img_pil)
    for y in range(40, IMG_SIZE - 10, 25):
        draw.line([(4, y), (10, y)], fill=200, width=1)
        draw.line([(IMG_SIZE - 10, y), (IMG_SIZE - 4, y)], fill=200, width=1)

    return img_pil


# ------------------------------------------------------------------
# BEZPIECZNY ZAPIS PLIKU (atomic write + walidacja + retry)
# ------------------------------------------------------------------
def save_image_safely(img, out_path, retries=3, class_name=None):
    """
    Zapisuje obraz w sposob "bezpieczny":
    1. Pisze najpierw do pliku tymczasowego (out_path + ".tmp"), nie bezposrednio na docelowy.
    2. Po zapisaniu sprawdza, czy plik nie jest pusty/uciety - probuje go otworzyc i zweryfikowac.
    3. Dopiero gdy plik tymczasowy jest OK, robi atomowy rename na docelowa nazwe.
    4. Jezeli cos sie nie powiedzie - usuwa tmp i probuje jeszcze raz (do `retries` razy).

    To rozwiazuje bug, ktory wystapil w pierwszej generacji datasetu (SR_0089.png byl
    truncated/0-bajtowy) - przerwanie procesu w trakcie zapisu zostawialo uszkodzony plik.

    Parametry:
      img        - obiekt PIL.Image do zapisania
      out_path   - docelowa sciezka (Path lub str)
      retries    - ile razy probowac zanim si poddamy (default 3)
      class_name - opcjonalne, do regeneracji obrazu jezeli pierwsza proba sie nie powiodla

    Zwraca: True jezeli udalo sie zapisac, False jezeli wszystkie proby zawiodly.
    """
    out_path = Path(out_path)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    for attempt in range(1, retries + 1):
        try:
            # Krok 1: zapis do pliku tymczasowego.
            # Format="PNG" podajemy jawnie, bo nasz plik tmp ma rozszerzenie ".png.tmp"
            # i PIL by go nie rozpoznal automatycznie po rozszerzeniu.
            img.save(tmp_path, format="PNG", optimize=True)

            # Krok 2: walidacja - czy plik istnieje, ma sensowny rozmiar
            #         i da sie go otworzyc przez PIL bez bledu?
            if not tmp_path.exists():
                raise IOError("plik tymczasowy nie powstal")
            size = tmp_path.stat().st_size
            if size < MIN_VALID_PNG_BYTES:
                raise IOError(f"plik za maly: {size} bajtow (< {MIN_VALID_PNG_BYTES})")
            # verify() rzuci wyjatkiem dla truncated PNG
            with Image.open(tmp_path) as test_img:
                test_img.verify()

            # Krok 3: atomowy rename - po tym kroku albo cala operacja sie udala,
            # albo nie zmienilismy stanu docelowego folderu
            tmp_path.replace(out_path)
            return True

        except Exception as e:
            # Sprzatanie po nieudanej probie: usun tmp jezeli zostal
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            print(f"  [WARN] {out_path.name}: proba {attempt}/{retries} nieudana ({e})")
            # Jezeli mozemy regenerowac obraz - zrob to (moze byl bug w samym img)
            if attempt < retries and class_name is not None:
                img = generate_image(class_name)

    print(f"  [ERROR] {out_path.name}: nie udalo sie zapisac po {retries} probach")
    return False


def is_file_valid_png(path):
    """Szybki test: czy plik to poprawny, nieuciety PNG > MIN_VALID_PNG_BYTES bajtow?"""
    path = Path(path)
    if not path.exists():
        return False
    if path.stat().st_size < MIN_VALID_PNG_BYTES:
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def generate_for_class(class_name, force=False):
    """Generuje obrazy dla pojedynczej klasy. Pomija istniejace (resume-friendly).

    Bezpieczenstwo:
    - Kazdy zapis idzie przez save_image_safely (atomic write + walidacja + retry).
    - Pliki ktore istnieja ale sa uszkodzone (truncated PNG) traktujemy jak brakujace
      i regenerujemy. Bez tego po przerwaniu skryptu zostawalismy z 0-bajtowymi smieciami.
    """
    n_train = int(PER_CLASS * TRAIN_RATIO)
    n_test = PER_CLASS - n_train
    (OUTPUT_DIR / "train" / class_name).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "test" / class_name).mkdir(parents=True, exist_ok=True)

    print(f"[{class_name}] target: {n_train} train + {n_test} test")
    generated, skipped, repaired, failed = 0, 0, 0, 0
    for i in range(PER_CLASS):
        split = "train" if i < n_train else "test"
        idx_in_split = i if split == "train" else i - n_train
        fname = f"{class_name}_{idx_in_split:04d}.png"
        out_path = OUTPUT_DIR / split / class_name / fname

        if out_path.exists() and not force:
            # Plik istnieje - ale czy jest na pewno OK?
            if is_file_valid_png(out_path):
                skipped += 1
                continue
            # Plik istnieje ale jest uszkodzony - regenerujemy go (jak --repair)
            print(f"  [REPAIR] {fname} uszkodzony, regeneruje")
            repaired += 1

        img = generate_image(class_name)
        ok = save_image_safely(img, out_path, class_name=class_name)
        if ok:
            generated += 1
        else:
            failed += 1
        if generated > 0 and generated % 200 == 0:
            print(f"  [{class_name}] ... {generated} nowych")

    msg = (f"[{class_name}] DONE: wygenerowano {generated}, "
           f"pominieto {skipped}, naprawiono {repaired}, BLEDY: {failed}")
    print(msg)
    return failed


# ------------------------------------------------------------------
# WERYFIKACJA / NAPRAWA ISTNIEJACEGO DATASETU
# ------------------------------------------------------------------
def verify_dataset():
    """Skanuje caly dataset i raportuje uszkodzone pliki. NIE modyfikuje nic."""
    print(f"[VERIFY] Skanuje {OUTPUT_DIR}")
    broken = []
    total = 0
    for split in ["train", "test"]:
        for c in CLASSES:
            d = OUTPUT_DIR / split / c
            if not d.exists():
                continue
            for f in sorted(d.glob("*.png")):
                total += 1
                if not is_file_valid_png(f):
                    broken.append(f)
    print(f"[VERIFY] sprawdzono {total} plikow, uszkodzonych: {len(broken)}")
    for b in broken:
        size = b.stat().st_size if b.exists() else 0
        rel = b.relative_to(OUTPUT_DIR)
        print(f"  - {rel} ({size} bajtow)")
    # Sprawdz tez czy nie zostawione gdzies pliki .tmp
    tmps = list(OUTPUT_DIR.rglob("*.tmp"))
    if tmps:
        print(f"[VERIFY] znaleziono {len(tmps)} plikow .tmp (smieci po przerwanym zapisie):")
        for t in tmps:
            print(f"  - {t.relative_to(OUTPUT_DIR)}")
    return broken, tmps


def repair_dataset():
    """Skanuje dataset, regeneruje uszkodzone pliki, czysci pliki .tmp."""
    broken, tmps = verify_dataset()
    if not broken and not tmps:
        print("[REPAIR] Nic do naprawy - dataset czysty.")
        return 0

    # Czyscimy ewentualne pliki .tmp (smieci po przerwanym zapisie)
    for t in tmps:
        try:
            t.unlink()
            print(f"[REPAIR] usunieto: {t.relative_to(OUTPUT_DIR)}")
        except Exception as e:
            print(f"[REPAIR] nie udalo sie usunac {t}: {e}")

    if not broken:
        return 0

    # Regenerujemy uszkodzone pliki. Nazwa pliku zawiera klase i indeks
    # (np. SR_0089.png), wiec wiemy ktora klase generowac.
    print(f"[REPAIR] Regeneruje {len(broken)} uszkodzonych plikow")
    fixed, failed = 0, 0
    for path in broken:
        # Z nazwy CLASS_IDX.png wyciagamy klase
        stem = path.stem  # np. "SR_0089"
        class_name = stem.split("_")[0]
        if class_name not in CLASSES:
            print(f"  [REPAIR] {path.name}: nieznana klasa '{class_name}', pomijam")
            failed += 1
            continue
        img = generate_image(class_name)
        ok = save_image_safely(img, path, class_name=class_name)
        if ok:
            fixed += 1
            print(f"  [OK] naprawiono {path.relative_to(OUTPUT_DIR)}")
        else:
            failed += 1
    print(f"[REPAIR] DONE: naprawiono {fixed}, bledy: {failed}")
    return failed


def write_split_csv():
    """Zbiera istniejace pliki i zapisuje split.csv."""
    csv_path = OUTPUT_DIR / "split.csv"
    records = []
    for split in ["train", "test"]:
        for c in CLASSES:
            d = OUTPUT_DIR / split / c
            if not d.exists():
                continue
            for f in sorted(d.glob("*.png")):
                records.append({
                    "filename": f"{split}/{c}/{f.name}",
                    "class": c,
                    "split": split,
                })
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class", "split"])
        writer.writeheader()
        writer.writerows(records)
    print(f"[CSV] zapisano {len(records)} wpisow do {csv_path}")
    return len(records)


def main(only_class=None, write_csv=False, verify=False, repair=False):
    print(f"[INFO] Output: {OUTPUT_DIR}")
    if verify:
        verify_dataset()
        return
    if repair:
        repair_dataset()
        return
    if write_csv:
        write_split_csv()
        return
    classes_to_run = [only_class] if only_class else CLASSES
    for c in classes_to_run:
        generate_for_class(c)


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    # Flagi mozna laczyc np. --repair --csv (najpierw naprawa, potem CSV)
    if "--verify" in args:
        main(verify=True)
    elif "--repair" in args:
        main(repair=True)
        if "--csv" in args:
            main(write_csv=True)
    elif "--csv" in args:
        main(write_csv=True)
    elif args:
        # python generate_synthetic_usg.py DH SR ...
        for c in args:
            if c in CLASSES:
                main(only_class=c)
    else:
        main()
