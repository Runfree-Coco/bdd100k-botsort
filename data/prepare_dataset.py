"""
Convert BDD100K detection annotations to YOLO format.

BDD100K detection label structure:
  <label_root>/{train,val}/<video_name>.json
  Each JSON: {"name": "video_name", "frames": [{"objects": [...]}]}

YOLO output structure:
  <out_dir>/images/{train,val}/<video_name>.jpg
  <out_dir>/labels/{train,val}/<video_name>.txt
  <out_dir>/bdd100k_det.yaml

Usage:
  python data/prepare_dataset.py \
    --img_root   /path/to/bdd100k_images_100k/100k \
    --label_root /path/to/bdd100k/100k \
    --out_dir    /path/to/bdd_yolo_dataset \
    --train_n    70000 \
    --val_n      -1
"""

import json
import shutil
import random
import argparse
from pathlib import Path

BDD_TO_YOLO = {"car": 0, "truck": 1, "bus": 2, "motorcycle": 3, "bicycle": 4}
YOLO_NAMES  = ["car", "truck", "bus", "motorcycle", "bicycle"]
IMG_W, IMG_H = 1280, 720


def convert_box_to_yolo(x1, y1, x2, y2):
    cx = max(0.0, min(1.0, (x1 + x2) / 2.0 / IMG_W))
    cy = max(0.0, min(1.0, (y1 + y2) / 2.0 / IMG_H))
    w  = max(0.001, min(1.0, (x2 - x1) / IMG_W))
    h  = max(0.001, min(1.0, (y2 - y1) / IMG_H))
    return cx, cy, w, h


def process_split(label_root, img_src, img_dst, lbl_dst, sample_n, split_name):
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    json_files = sorted(label_root.glob("*.json"))
    print(f"\n[{split_name}] Found {len(json_files)} JSON files")

    all_frames = []
    missing = 0
    for json_path in json_files:
        try:
            with open(json_path, encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        video_name = data.get("name")
        if not video_name:
            continue
        for frame in data.get("frames", []):
            img_name = f"{video_name}.jpg"
            if not (img_src / img_name).exists():
                missing += 1
                continue
            labels = []
            for obj in frame.get("objects", []):
                cat  = obj.get("category")
                bbox = obj.get("box2d")
                if cat not in BDD_TO_YOLO or not bbox:
                    continue
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                if x2 <= x1 or y2 <= y1 or (x2 - x1) < 2 or (y2 - y1) < 2:
                    continue
                labels.append((BDD_TO_YOLO[cat], x1, y1, x2, y2))
            if labels:
                all_frames.append((img_name, labels))

    print(f"  Valid frames: {len(all_frames)}  (missing images: {missing})")

    if sample_n > 0 and sample_n < len(all_frames):
        random.seed(42)
        all_frames = random.sample(all_frames, sample_n)
        print(f"  Sampled: {sample_n}")

    converted = total_boxes = 0
    for img_name, labels in all_frames:
        src = img_src / img_name
        if not src.exists():
            continue
        dst_lbl = lbl_dst / (Path(img_name).stem + ".txt")
        lines = []
        for cls_id, x1, y1, x2, y2 in labels:
            cx, cy, w, h = convert_box_to_yolo(x1, y1, x2, y2)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            total_boxes += 1
        if not lines:
            continue
        dst_lbl.write_text("\n".join(lines) + "\n")
        dst_img = img_dst / img_name
        if not dst_img.exists():
            shutil.copy2(src, dst_img)
        converted += 1

    print(f"  Converted: {converted} images, {total_boxes} boxes")
    return converted


def write_yaml(out_dir, train_n, val_n):
    yaml_path = out_dir / "bdd100k_det.yaml"
    content = (
        f"path: {out_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: {len(YOLO_NAMES)}\n"
        f"names: {YOLO_NAMES}\n"
    )
    yaml_path.write_text(content)
    print(f"\nDataset YAML -> {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="BDD100K detection -> YOLO format")
    parser.add_argument("--img_root",   required=True,
                        help="BDD100K 100k image root (contains train/ and val/)")
    parser.add_argument("--label_root", required=True,
                        help="BDD100K 100k label root (contains train/ and val/ JSON)")
    parser.add_argument("--out_dir",    required=True,
                        help="Output directory for YOLO dataset")
    parser.add_argument("--train_n", type=int, default=-1,
                        help="Training samples to keep (-1 = all, default: -1)")
    parser.add_argument("--val_n",   type=int, default=-1,
                        help="Validation samples to keep (-1 = all, default: -1)")
    args = parser.parse_args()

    img_root   = Path(args.img_root)
    label_root = Path(args.label_root)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BDD100K Detection -> YOLO")
    print(f"  Images:  {img_root}")
    print(f"  Labels:  {label_root}")
    print(f"  Output:  {out_dir}")
    print(f"  Classes: {YOLO_NAMES}")
    print("=" * 60)

    for split in ("train", "val"):
        n = args.train_n if split == "train" else args.val_n
        process_split(
            label_root = label_root / split,
            img_src    = img_root   / split,
            img_dst    = out_dir / "images" / split,
            lbl_dst    = out_dir / "labels" / split,
            sample_n   = n,
            split_name = split,
        )

    write_yaml(out_dir, args.train_n, args.val_n)

    n_train = len(list((out_dir / "images/train").glob("*.jpg")))
    n_val   = len(list((out_dir / "images/val").glob("*.jpg")))
    print(f"\nDone! Train: {n_train} images, Val: {n_val} images")
    print(f"Next: python detector/train.py --dataset_yaml {out_dir}/bdd100k_det.yaml")


if __name__ == "__main__":
    main()
