import argparse
import csv
import json
import os
from pathlib import Path

def load_coco_filenames(instances_json_path: Path):
    with open(instances_json_path, "r") as f:
        data = json.load(f)
    return [img["file_name"] for img in data["images"]]

def iter_images(folder: Path):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    for root, _, files in os.walk(str(folder)):
        for f in files:
            if f.lower().endswith(exts):
                yield Path(root) / f

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)

    coco_train_dir = data_root / "coco" / "train2017"
    coco_val_dir   = data_root / "coco" / "val2017"
    gen_dir        = data_root / "genimage" / "stable_diffusion"

    ann_train = data_root / "index" / "annotations" / "instances_train2017.json"
    ann_val   = data_root / "index" / "annotations" / "instances_val2017.json"

    index_dir = data_root / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    out_csv = index_dir / "dataset_index.csv"

    rows = []

    # COCO train (from JSON, no directory listing)
    for fn in load_coco_filenames(ann_train):
        rows.append([str(coco_train_dir / fn), 0, "coco", "real", "train"])

    # COCO val (from JSON)
    for fn in load_coco_filenames(ann_val):
        rows.append([str(coco_val_dir / fn), 0, "coco", "real", "val"])

    # GenImage (AI)
    for p in iter_images(gen_dir):
        rows.append([str(p), 1, "genimage", "stable_diffusion", "train"])

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "source", "generator", "split"])
        writer.writerows(rows)

    print("✅ dataset_index.csv created at:", out_csv)
    print("COCO train:", len(load_coco_filenames(ann_train)))
    print("COCO val  :", len(load_coco_filenames(ann_val)))
    print("GenImage  :", sum(1 for r in rows if r[2] == "genimage"))
    print("Total     :", len(rows))

if __name__ == "__main__":
    main()
