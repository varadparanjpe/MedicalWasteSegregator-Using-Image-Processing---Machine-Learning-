"""
Biomedical Waste Dataset Preprocessing
=======================================
Merges TWO source datasets into 6 WHO-aligned biomedical waste categories:
  - /Users/varadparanjpe/Downloads/archive/dataset  (original 8 Indonesian classes)
  - /Users/varadparanjpe/Downloads/archive           (new archive folders)

Pipeline:
  1. Class mapping → 6 WHO categories
  2. Image resizing (224x224) & quality filtering
  3. Data augmentation on train set (rotation, flip, brightness, zoom)
  4. Stratified train / val / test split  (70 / 15 / 15)
  5. Class distribution chart + sample grid saved as PNGs
"""

import os, random, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────
# 1.  PATHS  — update OUT_ROOT if you want output elsewhere
# ──────────────────────────────────────────────────────────────
ARCHIVE_ROOT  = Path("/Users/varadparanjpe/Downloads/archive")
DATASET_ROOT  = ARCHIVE_ROOT / "dataset"          # Indonesian 8-class subfolder
OUT_ROOT      = Path("/Users/varadparanjpe/Downloads/processed_dataset")

IMG_SIZE   = (224, 224)
AUG_FACTOR = 6          # augmented copies per train image
SPLITS     = {"train": 0.70, "val": 0.15, "test": 0.15}

# ──────────────────────────────────────────────────────────────
# 2.  CLASS MAPPING  →  6 WHO CATEGORIES
# ──────────────────────────────────────────────────────────────
#
#  WHO Category 1 : sharps_waste          (White Bin  ⬜)
#  WHO Category 2 : infectious_waste      (Yellow Bin 🟡)
#  WHO Category 3 : pathological_waste    (Yellow Bin 🟡 — separate bag)
#  WHO Category 4 : plastic_recyclable    (Red Bin    🔴)
#  WHO Category 5 : pharmaceutical_waste  (Blue Bin   🔵)
#  WHO Category 6 : general_waste         (Black Bin  ⬛)
#
# ──────────────────────────────────────────────────────────────

# Folders inside  archive/dataset/  (Indonesian labels)
DATASET_MAPPING = {
    "suntik":   "sharps_waste",
    "skalpels": "sharps_waste",
    "ampul":    "pharmaceutical_waste",
    "IV-tube":  "infectious_waste",
    "kapas":    "infectious_waste",
    "perban":   "infectious_waste",
    "gloves":   "plastic_recyclable",
    "masker":   "general_waste",
}

# Folders directly inside  archive/  (new classes from screenshot)
ARCHIVE_MAPPING = {
    # Sharps
    "syringe":              "sharps_waste",
    "syringe_needle":       "sharps_waste",
    "tweezers":             "sharps_waste",

    # Infectious
    "gauze":                "infectious_waste",
    "urine_bag":            "infectious_waste",
    "test_tube":            "infectious_waste",

    # Pathological
    "body_tissue_or_organ": "pathological_waste",
    "organic_waste":        "pathological_waste",

    # Plastic recyclable
    "glove_pair_latex":     "plastic_recyclable",
    "glove_pair_nitrile":   "plastic_recyclable",
    "glove_pair_surgery":   "plastic_recyclable",
    "glove_single_latex":   "plastic_recyclable",
    "glove_single_nitrile": "plastic_recyclable",
    "glove_single_surgery": "plastic_recyclable",
    "shoe_cover_pair":      "plastic_recyclable",
    "shoe_cover_single":    "plastic_recyclable",
    "plastic_equipment_packaging": "plastic_recyclable",

    # Pharmaceutical / glass
    "glass_equipment_packaging":  "pharmaceutical_waste",
    "metal_equipment_packaging":  "pharmaceutical_waste",
    "paper_equipment_packaging":  "pharmaceutical_waste",

    # General
    "mask":                 "general_waste",
    "medical_cap":          "general_waste",
    "medical_glasses":      "general_waste",
}

BIN_COLOUR = {
    "sharps_waste":         "White Bin  ⬜",
    "infectious_waste":     "Yellow Bin 🟡",
    "pathological_waste":   "Yellow Bin 🟡 (separate bag)",
    "plastic_recyclable":   "Red Bin    🔴",
    "pharmaceutical_waste": "Blue Bin   🔵",
    "general_waste":        "Black Bin  ⬛",
}

ALL_CATEGORIES = list(BIN_COLOUR.keys())

# ──────────────────────────────────────────────────────────────
# 3.  CREATE OUTPUT DIRECTORIES
# ──────────────────────────────────────────────────────────────
for split in SPLITS:
    for cat in ALL_CATEGORIES:
        (OUT_ROOT / split / cat).mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 4.  HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────
def load_and_validate(path: Path, min_size=32):
    try:
        img = Image.open(path).convert("RGB")
        return None if min(img.size) < min_size else img
    except Exception:
        return None

def preprocess(img: Image.Image) -> Image.Image:
    return img.resize(IMG_SIZE, Image.LANCZOS)

def augment(img: Image.Image, n=AUG_FACTOR):
    results = []
    for _ in range(n):
        aug = img.copy()
        aug = aug.rotate(random.uniform(-30, 30), expand=False, fillcolor=(200, 200, 200))
        if random.random() > 0.5: aug = ImageOps.mirror(aug)
        if random.random() > 0.5: aug = ImageOps.flip(aug)
        aug = ImageEnhance.Brightness(aug).enhance(random.uniform(0.7, 1.3))
        w, h = aug.size
        p = random.uniform(0.05, 0.20)
        l, u = int(w*p), int(h*p)
        aug = aug.crop((l, u, w-l, h-u)).resize(IMG_SIZE, Image.LANCZOS)
        results.append(aug)
    return results

# ──────────────────────────────────────────────────────────────
# 5.  COLLECT ALL VALID IMAGES FROM BOTH SOURCES
# ──────────────────────────────────────────────────────────────
print("=" * 62)
print("  BIOMEDICAL WASTE PREPROCESSING  —  Merging 2 datasets")
print("=" * 62)

category_images = defaultdict(list)
skipped = 0

def collect(folder_root: Path, mapping: dict, source_tag: str):
    global skipped
    for raw_label, category in mapping.items():
        src_dir = folder_root / raw_label
        if not src_dir.exists():
            print(f"  [SKIP] Not found: {src_dir}")
            continue
        files = [f for f in src_dir.iterdir()
                 if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        for fpath in files:
            img = load_and_validate(fpath)
            if img is None:
                skipped += 1
            else:
                category_images[category].append((fpath, raw_label, source_tag, img))

collect(DATASET_ROOT, DATASET_MAPPING, "orig")
collect(ARCHIVE_ROOT, ARCHIVE_MAPPING, "arch")

total_valid = sum(len(v) for v in category_images.values())
print(f"\n  Valid images collected : {total_valid}")
print(f"  Corrupt / too-small   : {skipped}\n")

print("  Raw class distribution (before augmentation):")
for cat in ALL_CATEGORIES:
    n = len(category_images[cat])
    print(f"    {cat:<26} {n:>5} images   {BIN_COLOUR[cat]}")

# ──────────────────────────────────────────────────────────────
# 6.  STRATIFIED SPLIT
# ──────────────────────────────────────────────────────────────
print("\n  Splitting 70 / 15 / 15 (stratified per category) ...")
split_registry   = {"train": [], "val": [], "test": []}
split_counts     = defaultdict(lambda: defaultdict(int))

for cat, items in category_images.items():
    if len(items) < 3:
        for item in items:
            split_registry["train"].append((*item, cat))
            split_counts["train"][cat] += 1
        continue
    train_val, test = train_test_split(items, test_size=SPLITS["test"], random_state=42)
    val_ratio = SPLITS["val"] / (SPLITS["train"] + SPLITS["val"])
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=42)
    for split_name, subset in [("train", train), ("val", val), ("test", test)]:
        for item in subset:
            split_registry[split_name].append((*item, cat))
            split_counts[split_name][cat] += 1

for s in ("train", "val", "test"):
    print(f"    {s:<6}: {sum(split_counts[s].values())} images")

# ──────────────────────────────────────────────────────────────
# 7.  SAVE  (preprocess + augment train)
# ──────────────────────────────────────────────────────────────
print("\n  Saving preprocessed images ...")
saved_counts = defaultdict(lambda: defaultdict(int))
mean_list, std_list = [], []

for split_name, items in split_registry.items():
    for fpath, raw_label, source_tag, img, cat in items:
        proc = preprocess(img)
        arr  = np.array(proc).astype(np.float32) / 255.0
        mean_list.append(arr.mean())
        std_list.append(arr.std())

        out_dir = OUT_ROOT / split_name / cat
        stem    = f"{source_tag}_{raw_label}_{fpath.stem}"
        proc.save(out_dir / f"{stem}.jpg", "JPEG", quality=95)
        saved_counts[split_name][cat] += 1

        if split_name == "train":
            for i, aug_img in enumerate(augment(proc)):
                aug_img.save(out_dir / f"{stem}_aug{i+1}.jpg", "JPEG", quality=92)
                saved_counts[split_name][cat] += 1

# ──────────────────────────────────────────────────────────────
# 8.  SUMMARY TABLE
# ──────────────────────────────────────────────────────────────
dset_mean = float(np.mean(mean_list))
dset_std  = float(np.mean(std_list))

print(f"\n  Dataset pixel mean : {dset_mean:.4f}")
print(f"  Dataset pixel std  : {dset_std:.4f}")
print(f"  (use these values for normalisation during model training)\n")

header = f"  {'Category':<26}" + "".join(f"  {s:<8}" for s in ("Train","Val","Test","TOTAL"))
print(header)
print("  " + "-" * (len(header) - 2))
grand = {"train": 0, "val": 0, "test": 0}
for cat in ALL_CATEGORIES:
    row = f"  {cat:<26}"
    tot = 0
    for s in ("train", "val", "test"):
        n = saved_counts[s][cat]; row += f"  {n:<8}"; grand[s] += n; tot += n
    print(row + f"  {tot}")
print("  " + "-" * (len(header) - 2))
gt = sum(grand.values())
print(f"  {'TOTAL':<26}" + "".join(f"  {grand[s]:<8}" for s in ("train","val","test")) + f"  {gt}")

# ──────────────────────────────────────────────────────────────
# 9.  PLOTS
# ──────────────────────────────────────────────────────────────
COLORS = {
    "sharps_waste":         "#E74C3C",
    "infectious_waste":     "#F39C12",
    "pathological_waste":   "#8E44AD",
    "plastic_recyclable":   "#3498DB",
    "pharmaceutical_waste": "#1ABC9C",
    "general_waste":        "#95A5A6",
}

# Distribution bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Biomedical Waste Dataset — Class Distribution After Preprocessing",
             fontsize=13, fontweight="bold")

for ax, split_name in zip(axes, ("train", "val", "test")):
    labels = [c.replace("_", "\n") for c in ALL_CATEGORIES]
    counts = [saved_counts[split_name][c] for c in ALL_CATEGORIES]
    colors = [COLORS[c] for c in ALL_CATEGORIES]
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(f"{split_name.capitalize()} Set  (n={sum(counts)})",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Image count")
    ax.set_ylim(0, max(counts) * 1.28 if max(counts) > 0 else 10)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
dist_path = OUT_ROOT / "class_distribution.png"
plt.savefig(dist_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  [Saved] {dist_path}")

# Sample grid
fig, axes = plt.subplots(2, len(ALL_CATEGORIES), figsize=(22, 8))
fig.suptitle("Preprocessed Samples  (Top: Original  |  Bottom: Augmented)",
             fontsize=13, fontweight="bold")

for col, cat in enumerate(ALL_CATEGORIES):
    cat_dir   = OUT_ROOT / "train" / cat
    all_imgs  = sorted(cat_dir.glob("*.jpg"))
    originals = [f for f in all_imgs if "aug" not in f.name]
    augmented = [f for f in all_imgs if "aug"     in f.name]

    for row, pool in enumerate([originals, augmented]):
        if pool:
            axes[row][col].imshow(Image.open(random.choice(pool)))
        axes[row][col].axis("off")
        if row == 0:
            axes[row][col].set_title(cat.replace("_", "\n"),
                                     fontsize=8, fontweight="bold", color=COLORS[cat])

fig.text(0.01, 0.72, "Original",  va="center", rotation="vertical",
         fontsize=11, fontweight="bold", color="#2C3E50")
fig.text(0.01, 0.28, "Augmented", va="center", rotation="vertical",
         fontsize=11, fontweight="bold", color="#27AE60")
plt.tight_layout()
grid_path = OUT_ROOT / "sample_grid.png"
plt.savefig(grid_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [Saved] {grid_path}")

# Write log
log_path = OUT_ROOT / "preprocessing_log.txt"
with open(log_path, "w") as f:
    f.write("BIOMEDICAL WASTE PREPROCESSING LOG\n" + "="*60 + "\n\n")
    f.write("SOURCE 1: archive/dataset  (original Indonesian classes)\n")
    for k, v in DATASET_MAPPING.items(): f.write(f"  {k:<20} → {v}\n")
    f.write("\nSOURCE 2: archive/  (new archive folders)\n")
    for k, v in ARCHIVE_MAPPING.items(): f.write(f"  {k:<35} → {v}\n")
    f.write(f"\nImage size     : {IMG_SIZE}\n")
    f.write(f"Aug factor     : x{AUG_FACTOR} (train only)\n")
    f.write(f"Split          : train={SPLITS['train']} / val={SPLITS['val']} / test={SPLITS['test']}\n")
    f.write(f"Pixel mean     : {dset_mean:.4f}\n")
    f.write(f"Pixel std      : {dset_std:.4f}\n\n")
    f.write("FINAL COUNTS\n" + "-"*60 + "\n")
    for s in ("train", "val", "test"):
        f.write(f"\n{s.upper()}:\n")
        for cat in ALL_CATEGORIES:
            f.write(f"  {cat:<30} {saved_counts[s][cat]}\n")
print(f"  [Saved] {log_path}")

print("\n✅  All done!  Check your processed_dataset folder in Downloads.")