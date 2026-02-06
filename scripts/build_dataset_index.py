import argparse
import csv
from pathlib import Path

def iter_images(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            yield p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to AI_Image_Detection_Data"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)

    coco_train = data_root / "coco" / "train2017"
    coco_val   = data_root / "coco" / "val2017"
    gen_sd     = data_root / "genimage" / "stable_diffusion"

    index_dir = data_root / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    out_csv = index_dir / "dataset_index.csv"

    rows = []

    # COCO train (real)
    for p in iter_images(coco_train):
        rows.append([str(p), 0, "coco", "real", "train"])

    # COCO val (real)
    for p in iter_images(coco_val):
        rows.append([str(p), 0, "coco", "real", "val"])

    # GenImage (AI)
    for p in iter_images(gen_sd):
        rows.append([str(p), 1, "genimage", "stable_diffusion", "train"])

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "source", "generator", "split"])
        writer.writerows(rows)

    print("dataset_index.csv created")
    print("Total samples:", len(rows))
    print("COCO train:", sum(1 for r in rows if r[2] == "coco" and r[4] == "train"))
    print("COCO val:",   sum(1 for r in rows if r[2] == "coco" and r[4] == "val"))
    print("GenImage:",   sum(1 for r in rows if r[2] == "genimage"))

if __name__ == "__main__":
    main()
