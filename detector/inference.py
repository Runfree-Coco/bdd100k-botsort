"""
Run fine-tuned YOLOv8 on BDD100K tracking validation sequences.

Output: MOT-format detection .txt files, one per sequence.
Format: frame_id, det_id, x, y, w, h, conf, cls_id, -1, -1

Usage:
  python detector/inference.py \
    --model    /path/to/best.pt \
    --img_dir  /path/to/bdd100k/images/track/val \
    --out_dir  /path/to/det_results/yolo_cls \
    --conf     0.15 \
    --imgsz    1280
"""

import sys
import time
import argparse
from pathlib import Path

FINETUNE_CLASS_NAMES = {0: "car", 1: "truck", 2: "bus", 3: "motorcycle", 4: "bicycle"}
TARGET_CLASS_IDS     = set(FINETUNE_CLASS_NAMES.keys())


def run_on_sequence(model, seq_dir, out_file, conf, iou, imgsz):
    img_files = sorted(seq_dir.glob("*.jpg"))
    if not img_files:
        out_file.write_text("")
        return 0

    det_lines = []
    det_id    = 1
    for frame_idx, img_path in enumerate(img_files):
        frame_id = frame_idx + 1
        results  = model(str(img_path), conf=conf, iou=iou, imgsz=imgsz,
                         augment=True, verbose=False)
        result   = results[0]
        has_det  = False

        if result.boxes is not None and len(result.boxes) > 0:
            for j in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[j].item())
                if cls_id not in TARGET_CLASS_IDS:
                    continue
                score       = float(result.boxes.conf[j].item())
                x1,y1,x2,y2 = result.boxes.xyxy[j].tolist()
                w, h = x2 - x1, y2 - y1
                if w <= 2 or h <= 2:
                    continue
                det_lines.append(
                    f"{frame_id},{det_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                    f"{score:.4f},{cls_id},-1,-1\n"
                )
                det_id += 1
                has_det = True

        if not has_det:
            det_lines.append(f"{frame_id},0,0.0,0.0,1.0,1.0,0.0,0,-1,-1\n")

    with open(out_file, 'w') as f:
        f.writelines(det_lines)

    return sum(1 for l in det_lines if l.split(',')[1] != '0')


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 inference on BDD100K tracking val")
    parser.add_argument("--model",   required=True, help="Path to fine-tuned YOLOv8 weights (.pt)")
    parser.add_argument("--img_dir", required=True, help="BDD100K tracking val image directory")
    parser.add_argument("--out_dir", required=True, help="Output directory for detection .txt files")
    parser.add_argument("--conf",  type=float, default=0.15)
    parser.add_argument("--iou",   type=float, default=0.45)
    parser.add_argument("--imgsz", type=int,   default=1280)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Please install: pip install ultralytics")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.img_dir)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    model.to(f"cuda:{args.device}")
    print(f"Classes: {list(FINETUNE_CLASS_NAMES.values())}")

    seq_dirs   = sorted(d for d in img_dir.iterdir() if d.is_dir())
    total_time = 0.0
    print(f"\n{len(seq_dirs)} sequences, starting inference...\n")

    for i, seq_dir in enumerate(seq_dirs):
        out_file = out_dir / f"{seq_dir.name}.txt"
        t0 = time.time()
        try:
            n_dets = run_on_sequence(model, seq_dir, out_file, args.conf, args.iou, args.imgsz)
        except Exception as e:
            print(f"  [{i+1}] {seq_dir.name} failed: {e}")
            out_file.write_text("")
            continue
        elapsed     = time.time() - t0
        total_time += elapsed

        if (i + 1) % 20 == 0 or i == 0 or i == len(seq_dirs) - 1:
            avg    = total_time / (i + 1)
            remain = avg * (len(seq_dirs) - i - 1)
            n_imgs = len(list(seq_dir.glob("*.jpg")))
            fps    = n_imgs / elapsed if elapsed > 0 else 0
            print(f"  [{i+1:3d}/{len(seq_dirs)}] {seq_dir.name}: "
                  f"{n_dets} dets, {elapsed:.1f}s ({fps:.1f} fps)"
                  f" | ETA {remain/60:.1f} min")

    print(f"\nDone! Total time: {total_time/60:.1f} min")
    print(f"Detection files -> {out_dir}/")
    print(f"\nNext: python run_tracking.py --det_dir {out_dir} ...")


if __name__ == "__main__":
    main()
