"""
Fine-tune YOLOv8 on BDD100K vehicle detection.

Prerequisite: run data/prepare_dataset.py first to generate the YOLO dataset.

Usage:
  python detector/train.py \
    --dataset_yaml /path/to/bdd_yolo_dataset/bdd100k_det.yaml \
    --model        yolov8l.pt \
    --epochs       50 \
    --batch        8 \
    --imgsz        1280 \
    --out_dir      runs/detect

GPU memory guide (YOLOv8l):
  imgsz=640,  batch=8  -> ~4 GB  (RTX 3050)
  imgsz=1280, batch=4  -> ~12 GB (RTX 3090/4090 recommended)
"""

import sys
import argparse
from pathlib import Path


AUGMENT_CFG = dict(
    hsv_h      = 0.015,
    hsv_s      = 0.7,
    hsv_v      = 0.4,
    degrees    = 0.0,    # no rotation for dashcam footage
    translate  = 0.1,
    scale      = 0.5,
    fliplr     = 0.5,
    mosaic     = 1.0,
    mixup      = 0.1,
    copy_paste = 0.0,
)

CLASS_NAMES = ["car", "truck", "bus", "motorcycle", "bicycle"]


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 fine-tuning on BDD100K")
    parser.add_argument("--dataset_yaml", required=True, help="Path to bdd100k_det.yaml")
    parser.add_argument("--model",   default="yolov8l.pt", help="Pretrained weights (default: yolov8l.pt)")
    parser.add_argument("--epochs",  type=int,   default=50)
    parser.add_argument("--batch",   type=int,   default=8)
    parser.add_argument("--imgsz",   type=int,   default=1280)
    parser.add_argument("--lr0",     type=float, default=0.001)
    parser.add_argument("--patience",type=int,   default=15)
    parser.add_argument("--workers", type=int,   default=4)
    parser.add_argument("--device",  default="0")
    parser.add_argument("--out_dir", default="runs/detect")
    parser.add_argument("--name",    default="bdd_finetune")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Please install: pip install ultralytics")
        sys.exit(1)

    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {name} ({vram:.1f} GB VRAM)")

    dataset_yaml = Path(args.dataset_yaml)
    if not dataset_yaml.exists():
        print(f"Dataset YAML not found: {dataset_yaml}")
        print("Run: python data/prepare_dataset.py first")
        sys.exit(1)

    print("=" * 60)
    print("YOLOv8 Fine-tuning - BDD100K Vehicle Detection")
    print(f"  Model:   {args.model}")
    print(f"  Dataset: {dataset_yaml}")
    print(f"  Epochs:  {args.epochs}  Batch: {args.batch}  imgsz: {args.imgsz}")
    print(f"  LR:      {args.lr0}")
    print("=" * 60)

    model = YOLO(args.model)
    model.train(
        data          = str(dataset_yaml),
        epochs        = args.epochs,
        batch         = args.batch,
        imgsz         = args.imgsz,
        device        = args.device,
        project       = args.out_dir,
        name          = args.name,
        exist_ok      = True,
        optimizer     = "AdamW",
        lr0           = args.lr0,
        lrf           = 0.1,
        warmup_epochs = 3,
        patience      = args.patience,
        workers       = args.workers,
        amp           = True,
        cache         = False,
        plots         = True,
        verbose       = True,
        **AUGMENT_CFG,
    )

    best_weights = Path(args.out_dir) / args.name / "weights/best.pt"
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Best weights: {best_weights}")
    print(f"\nNext: python detector/inference.py --model {best_weights} ...")
    print("=" * 60)


if __name__ == "__main__":
    main()
