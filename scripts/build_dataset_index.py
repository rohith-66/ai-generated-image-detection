# scripts/build_dataset_index.py
# Drive-safe dataset indexing for:
# - COCO 2017 (real): enumerated via official annotation JSON (no folder listing)
# - GenImage subset (AI): enumerated via os.walk (works fine for ~10k)
#
# Optional: balance training set by sampling COCO train down to match AI count.

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import List, Iterator


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def load_coco_filenames(instances_json_path: Path) -> List[str]:
    if not instances_json_path.exists():
        raise FileNotFoundError(f"Missing COCO annotation JSON: {instances_json_path}")
    with open(instances_json_path, "r") as f:
        data = json.load(f)
    if "images" not in data:
        raise ValueError(f"Invalid COCO JSON (missing 'images'): {instances_json_path}")
    return [img["file_name"] for img in data["images"]]


def iter_images_walk(folder: Path) -> Iterator[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")
    for root, _, files in os.walk(str(folder)):
        for fn in files:
            if fn.lower().endswith(IMAGE_EXTS):
                yield Path(root) / fn


def ensure_coco_annotations_present(data_root: Path) -> None:
    """
    Checks that the expected annotation JSONs exist at:
      {data_root}/index/annotations/instances_train2017.json
      {data_root}/index/annotations/instances_val2017.json

    This script does NOT download them automatically (keeps it clean & reproducible).
    """
    ann_train = data_root / "index" / "annotations" / "instances_train2017.json"
    ann_val = data_root / "index" / "annotations" / "instances_val2017.json"
    if not ann_train.exists() or not ann_val.exists():
        raise FileNotFoundError(
            "COCO annotation JSON files not found.\n"
            "Please download & unzip annotations into:\n"
            f"  {data_root}/index/\n\n"
            "Expected files:\n"
            f"  {ann_train}\n"
            f"  {ann_val}\n\n"
            "In Colab, run:\n"
            f"  %cd {data_root}/index\n"
            "  !wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip\n"
            "  !unzip -q annotations_trainval2017.zip\n"
        )


def main():
    parser = argparse.ArgumentParser(description="Build dataset_index.csv (Drive-safe).")
    parser.add_argument("--data_root", required=True, type=str,
                        help="Path to AI_Image_Detection_Data (Drive mounted path).")

    # Balancing options
    parser.add_argument("--balance_train", action="store_true",
                        help="If set, sample COCO train down to match AI train size (or --coco_train_size).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used for sampling (reproducible).")
    parser.add_argument("--coco_train_size", type=int, default=0,
                        help="If >0, sample COCO train to this many images (overrides balance_train size).")
    parser.add_argument("--ai_train_size", type=int, default=0,
                        help="If >0, limit AI images to this many (useful if gen folder has more).")

    # Split control (simple: keep COCO val as val; AI defaults to train)
    parser.add_argument("--ai_split", type=str, default="train", choices=["train", "val", "test"],
                        help="Which split label to assign to AI rows (default: train).")

    args = parser.parse_args()
    data_root = Path(args.data_root)

    coco_train_dir = data_root / "coco" / "train2017"
    coco_val_dir = data_root / "coco" / "val2017"
    gen_dir = data_root / "genimage" / "stable_diffusion"

    index_dir = data_root / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    out_csv = index_dir / "dataset_index.csv"

    # Ensure annotation JSONs exist (Drive-safe enumeration)
    ensure_coco_annotations_present(data_root)
    ann_train = data_root / "index" / "annotations" / "instances_train2017.json"
    ann_val = data_root / "index" / "annotations" / "instances_val2017.json"

    # Load COCO filenames from JSON
    coco_train_files = load_coco_filenames(ann_train)
    coco_val_files = load_coco_filenames(ann_val)

    # Enumerate AI images (walk is stable)
    ai_paths = list(iter_images_walk(gen_dir))

    # Optionally limit AI size (if folder grows later)
    if args.ai_train_size and args.ai_train_size > 0:
        ai_paths = sorted(ai_paths)[: args.ai_train_size]

    ai_count = len(ai_paths)

    # Decide COCO train sampling size
    sample_coco_train = None
    if args.coco_train_size and args.coco_train_size > 0:
        sample_coco_train = args.coco_train_size
    elif args.balance_train:
        # balance COCO train to match AI count
        sample_coco_train = ai_count

    if sample_coco_train is not None:
        if sample_coco_train > len(coco_train_files):
            raise ValueError(
                f"Requested coco_train_size={sample_coco_train} "
                f"but only {len(coco_train_files)} COCO train images exist."
            )
        rng = random.Random(args.seed)
        rng.shuffle(coco_train_files)
        coco_train_files = coco_train_files[:sample_coco_train]

    rows = []

    # COCO train (real) - from JSON filenames
    for fn in coco_train_files:
        rows.append([str(coco_train_dir / fn), 0, "coco", "real", "train"])

    # COCO val (real) - from JSON filenames
    for fn in coco_val_files:
        rows.append([str(coco_val_dir / fn), 0, "coco", "real", "val"])

    # AI images
    for p in ai_paths:
        rows.append([str(p), 1, "genimage", "stable_diffusion", args.ai_split])

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "source", "generator", "split"])
        writer.writerows(rows)

    # Print summary
    coco_train_n = sum(1 for r in rows if r[2] == "coco" and r[4] == "train")
    coco_val_n = sum(1 for r in rows if r[2] == "coco" and r[4] == "val")
    ai_n = sum(1 for r in rows if r[2] == "genimage")

    print(f"✅ dataset_index.csv created at: {out_csv}")
    print(f"COCO train: {coco_train_n}")
    print(f"COCO val  : {coco_val_n}")
    print(f"GenImage  : {ai_n}")
    print(f"Total     : {len(rows)}")

    if sample_coco_train is not None:
        print(f"(Note) COCO train sampled to {sample_coco_train} using seed={args.seed}")
    if args.ai_train_size and args.ai_train_size > 0:
        print(f"(Note) AI limited to {args.ai_train_size}")


if __name__ == "__main__":
    main()
