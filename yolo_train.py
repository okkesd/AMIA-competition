"""
YOLO11 Training Script
======================
Reads PNG images + metadata CSV → converts to YOLO format → trains & evaluates.

Metadata columns: image_id, class_name, class_id, x_min_new, y_min_new, x_max_new, y_max_new
"""

import os
import random
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import cv2
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit
from ultralytics import YOLO
import shutil


# ─────────────────────────── CONFIG ───────────────────────────────────────────
TRAIN_IMAGE_DIR = "train_preprocessed"               # folder containing .png images
METADATA_CSV    = "train_metadata_1024.csv"  # your metadata file
DATASET_DIR     = "yolo_dataset"        # will be created by this script
MODEL_WEIGHTS   = "yolo26s.pt"          # nano; change to yolo11s/m/l/x for larger
EPOCHS          = 80
IMG_SIZE        = 1024
BATCH_SIZE      = 2
VAL_SPLIT       = 0.2
SEED            = 42

# ── Test mode: use a small subset to verify everything works ──────────────────
TEST_RUN    = False    # set False for full training
TEST_IMAGES = 1000      # number of images to use in test mode
# ──────────────────────────────────────────────────────────────────────────────
print(">>> module loaded", flush=True)
if __name__ == "__main__":

    print(">>> entered main", flush=True)
    random.seed(SEED)

    if TEST_RUN:
        _epochs = 30
        _batch  = 2
        _imgsz  = 1024
        print("=" * 50, flush=True)
        print("TEST MODE — using limited images & epochs", flush=True)
        print("=" * 50, flush=True)
    else:
        _epochs = EPOCHS
        _batch  = BATCH_SIZE
        _imgsz  = IMG_SIZE

    # ── 1. Load metadata ──────────────────────────────────────────────────────
    df = pd.read_csv(METADATA_CSV)

    if TEST_RUN:
        sample_ids = df["image_id"].unique()[:TEST_IMAGES]
        df = df[df["image_id"].isin(sample_ids)]
        print(f"TEST MODE: using {df['image_id'].nunique()} images")

    print(f"Loaded {len(df)} annotations | {df['image_id'].nunique()} unique images")

    # ── 2. Build class map ────────────────────────────────────────────────────
    class_map = (
        df[["class_id", "class_name"]]
        .drop_duplicates()
        .sort_values("class_id")
        .set_index("class_id")["class_name"]
        .to_dict()
    )
    print(f"Classes ({len(class_map)}): {class_map}")
    
    # ── 3. Train / val split (grouped by image_id) ───────────────────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
    train_idx, val_idx = next(gss.split(df, groups=df["image_id"]))
    train_images = set(df.iloc[train_idx]["image_id"].unique())
    val_images   = set(df.iloc[val_idx]["image_id"].unique())
    print(f"Train images: {len(train_images)} | Val images: {len(val_images)}")
    
    if Path(DATASET_DIR).exists():
        shutil.rmtree(DATASET_DIR)

    # ── 4. Create YOLO directory structure ───────────────────────────────────
    for split in ("train", "val"):
        (Path(DATASET_DIR) / "images" / split).mkdir(parents=True, exist_ok=True)
        (Path(DATASET_DIR) / "labels" / split).mkdir(parents=True, exist_ok=True)

    def get_split(image_id):
        return "train" if image_id in train_images else "val"

    # ── 5. Convert annotations & create symlinks ──────────────────────────────
    missing_images = []

    for image_id, group in df.groupby("image_id"):
        img_path = Path(TRAIN_IMAGE_DIR) / f"{image_id}.png"

        if not img_path.exists():
            missing_images.append(image_id)
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            missing_images.append(image_id)
            continue
        h, w = img.shape[:2]

        split = get_split(image_id)

        # Symlink image (no disk space used)
        dst_img = Path(DATASET_DIR) / "images" / split / f"{image_id}.png"
        if not dst_img.exists():
            dst_img.symlink_to(img_path.resolve())

        # Write YOLO label file
        label_lines = []
        for _, row in group.iterrows():
            x_min = row["x_min_new"]
            y_min = row["y_min_new"]
            x_max = row["x_max_new"]
            y_max = row["y_max_new"]

            cx = ((x_min + x_max) / 2) / w
            cy = ((y_min + y_max) / 2) / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h

            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.0, min(1.0, bw))
            bh = max(0.0, min(1.0, bh))

            label_lines.append(
                f"{int(row['class_id'])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            )

        dst_lbl = Path(DATASET_DIR) / "labels" / split / f"{image_id}.txt"
        dst_lbl.write_text("\n".join(label_lines))

    if missing_images:
        print(f"Warning: {len(missing_images)} images not found and skipped.")
        print(f"  First 5: {missing_images[:5]}")

    # ── 6. Write data.yaml ────────────────────────────────────────────────────
    data_yaml = {
        "path"  : str(Path(DATASET_DIR).resolve()),
        "train" : "images/train",
        "val"   : "images/val",
        "nc"    : len(class_map),
        "names" : {int(k): v for k, v in class_map.items()},
    }
    yaml_path = Path(DATASET_DIR) / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"data.yaml written → {yaml_path}")

    # ── 7. Train ──────────────────────────────────────────────────────────────
    model = YOLO(MODEL_WEIGHTS)

    model.train(
        data     = str(yaml_path),
        epochs   = _epochs,
        imgsz    = _imgsz,
        batch    = _batch,
        seed     = SEED,
        project  = "runs/detect",
        name     = "yolo11_experiment",
        exist_ok = True,
        val      = True,
        plots    = True,
        device   = 0,       # 0 = first GPU; set "cpu" if no GPU
        workers = 8,
        cache = "disk",
        
    )

    # ── 8. Evaluate ───────────────────────────────────────────────────────────
    print("\n── Validation metrics ──")
    metrics = model.val()
    print(f"  mAP@50     : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95  : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")

    print("\nDone! Best weights → runs/detect/yolo11_experiment/weights/best.pt")