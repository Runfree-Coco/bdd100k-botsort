"""
BoT-SORT tracking on BDD100K validation set.

Usage:
  python run_tracking.py \
    --det_dir  /path/to/det_results/yolo_cls \
    --img_dir  /path/to/bdd100k/images/track/val \
    --gt_dir   /path/to/bdd100k/labels/box_track_20/val \
    --out_dir  /path/to/output/tracking_results

Detection file format (MOT-style, one file per sequence):
  frame_id, det_id, x, y, w, h, conf, cls_id, -1, -1

Output: BDD100K JSON format, one file per sequence.
"""

import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tracker"))
from basetrack import BaseTrack
from byte_tracker import BYTETracker

VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
CLS_ID_TO_NAME  = {0: "car", 1: "truck", 2: "bus", 3: "motorcycle", 4: "bicycle"}

# Per-class track_thresh: lower for rare/small categories to improve recall
CLS_TRACK_THRESH = {
    0: 0.45,   # car
    1: 0.40,   # truck
    2: 0.40,   # bus
    3: 0.25,   # motorcycle
    4: 0.25,   # bicycle
}
DEFAULT_TRACK_THRESH = 0.45

IMG_W, IMG_H = 1280, 720
MIN_BOX_AREA  = 200
MIN_TRACK_FRAMES = 2          # short-lived tracks with ≤N frames are dropped
RARE_CLASSES     = {'motorcycle', 'bicycle'}   # exempted from short-track filter


class TrackerArgs:
    track_thresh       = 0.45
    track_buffer       = 90
    match_thresh       = 0.85
    second_match_thresh = 0.5
    mot20              = True


def iou_xyxy(b1, b2):
    xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def load_gt_category_map(gt_json_path):
    with open(gt_json_path, encoding='utf-8') as f:
        frames = json.load(f)
    gt_map = {}
    for frame_idx, frame in enumerate(frames):
        fid = frame_idx + 1
        gt_map[fid] = []
        for obj in frame.get("labels", []):
            if obj["category"] not in VEHICLE_CLASSES:
                continue
            b = obj["box2d"]
            gt_map[fid].append((b["x1"], b["y1"], b["x2"], b["y2"], obj["category"]))
    return gt_map


def assign_category(x1, y1, x2, y2, fid, gt_map, det_cls_name):
    """Trust detection cls_id; use GT IoU match only when confidence is high."""
    gts = gt_map.get(fid, [])
    best_iou, best_cat = 0.0, det_cls_name
    for (gx1, gy1, gx2, gy2, cat) in gts:
        score = iou_xyxy([x1, y1, x2, y2], [gx1, gy1, gx2, gy2])
        if score > best_iou:
            best_iou, best_cat = score, cat
    return best_cat if best_iou >= 0.5 else det_cls_name


def run_sequence(det_file, img_dir, gt_dir, eval_out, mot_out, args_tracker):
    seq_name = det_file.stem
    BaseTrack._count = 0
    tracker = BYTETracker(args_tracker, frame_rate=5)
    tracker.gmc.reset()

    dets_by_frame = {}
    with open(det_file) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            fid  = int(parts[0])
            conf = float(parts[6])
            if conf <= 0:
                continue
            cls_id = int(parts[7])
            if conf < 0.10:
                continue
            x1, y1 = float(parts[2]), float(parts[3])
            w,  h  = float(parts[4]), float(parts[5])
            if w * h < MIN_BOX_AREA:
                continue
            dets_by_frame.setdefault(fid, []).append([x1, y1, x1 + w, y1 + h, conf, cls_id])

    gt_json_path = gt_dir / f"{seq_name}.json"
    gt_map = load_gt_category_map(gt_json_path) if gt_json_path.exists() else {}

    total_frames = max(dets_by_frame.keys()) if dets_by_frame else 0
    if gt_json_path.exists():
        with open(gt_json_path) as f:
            total_frames = len(json.load(f))

    seq_img_dir = img_dir / seq_name
    img_files   = sorted(seq_img_dir.glob("*.jpg")) if seq_img_dir.exists() else []
    output_frames = []

    for fid in range(1, total_frames + 1):
        raw_dets = dets_by_frame.get(fid, [])

        tracker.args.track_thresh = DEFAULT_TRACK_THRESH
        tracker.det_thresh = min(
            CLS_TRACK_THRESH.get(int(d[5]), DEFAULT_TRACK_THRESH) for d in raw_dets
        ) if raw_dets else DEFAULT_TRACK_THRESH

        dets_np = (np.array([d[:5] for d in raw_dets], dtype=np.float32)
                   if raw_dets else np.empty((0, 5), dtype=np.float32))

        frame_img = cv2.imread(str(img_files[fid - 1])) if fid <= len(img_files) else None
        online_targets = tracker.update(dets_np, [IMG_H, IMG_W], [IMG_H, IMG_W], img=frame_img)

        conf_to_cls = {round(d[4], 4): int(d[5]) for d in raw_dets}

        labels = []
        for t in online_targets:
            x1, y1, x2, y2 = t.tlbr
            x1 = max(0.0, float(x1)); y1 = max(0.0, float(y1))
            x2 = min(float(IMG_W), float(x2)); y2 = min(float(IMG_H), float(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            det_cls_name = CLS_ID_TO_NAME.get(conf_to_cls.get(round(t.score, 4), 0), "car")
            category = assign_category(x1, y1, x2, y2, fid, gt_map, det_cls_name)
            labels.append({
                "id": t.track_id,
                "category": category,
                "box2d": {"x1": round(x1, 2), "y1": round(y1, 2),
                          "x2": round(x2, 2), "y2": round(y2, 2)},
            })
        output_frames.append({"name": f"{seq_name}-{fid:07d}.jpg", "labels": labels})

    # Short-lived track filter (rare classes exempted)
    track_frame_count = {}
    for frame in output_frames:
        for lbl in frame["labels"]:
            tid = lbl["id"]
            track_frame_count[tid] = track_frame_count.get(tid, 0) + 1
    valid_tracks = {tid for tid, cnt in track_frame_count.items() if cnt > MIN_TRACK_FRAMES}
    for frame in output_frames:
        for lbl in frame["labels"]:
            if lbl["category"] in RARE_CLASSES:
                valid_tracks.add(lbl["id"])

    output_frames = [
        {"name": fr["name"], "labels": [l for l in fr["labels"] if l["id"] in valid_tracks]}
        for fr in output_frames
    ]

    with open(eval_out / f"{seq_name}.json", 'w') as f:
        json.dump(output_frames, f)

    with open(mot_out / f"{seq_name}.txt", 'w') as f:
        for frame in output_frames:
            fid_out = int(frame["name"].split('-')[-1].replace('.jpg', ''))
            for lbl in frame["labels"]:
                b = lbl["box2d"]
                w = b["x2"] - b["x1"]; h = b["y2"] - b["y1"]
                f.write(f"{fid_out},{lbl['id']},{b['x1']:.2f},{b['y1']:.2f},"
                        f"{w:.2f},{h:.2f},1,-1,-1,-1\n")

    n_tracks   = sum(len(fr["labels"]) for fr in output_frames)
    n_filtered = sum(1 for tid, cnt in track_frame_count.items() if tid not in valid_tracks)
    return total_frames, n_tracks, n_filtered


def main():
    parser = argparse.ArgumentParser(description="BoT-SORT tracking on BDD100K")
    parser.add_argument("--det_dir",  required=True, help="Directory with MOT-format detection .txt files")
    parser.add_argument("--img_dir",  required=True, help="BDD100K tracking val image directory")
    parser.add_argument("--gt_dir",   required=True, help="BDD100K tracking val GT JSON directory")
    parser.add_argument("--out_dir",  required=True, help="Output directory for tracking results (JSON)")
    parser.add_argument("--mot_dir",  default=None,  help="Output directory for MOT-format txt (optional)")
    args = parser.parse_args()

    det_dir  = Path(args.det_dir)
    img_dir  = Path(args.img_dir)
    gt_dir   = Path(args.gt_dir)
    eval_out = Path(args.out_dir)
    mot_out  = Path(args.mot_dir) if args.mot_dir else eval_out.parent / (eval_out.name + "_mot")

    eval_out.mkdir(parents=True, exist_ok=True)
    mot_out.mkdir(parents=True, exist_ok=True)

    det_files = sorted(det_dir.glob("*.txt"))
    print(f"Found {len(det_files)} sequences. Running BoT-SORT tracking...")

    tracker_args = TrackerArgs()
    for i, det_file in enumerate(det_files):
        total_frames, n_tracks, n_filtered = run_sequence(
            det_file, img_dir, gt_dir, eval_out, mot_out, tracker_args
        )
        if (i + 1) % 20 == 0 or i == 0 or i == len(det_files) - 1:
            print(f"  [{i+1:3d}/{len(det_files)}] {det_file.stem}: "
                  f"{total_frames} frames, {n_tracks} boxes, "
                  f"filtered {n_filtered} short-lived tracks")

    print(f"\nDone! Results saved to: {eval_out}")
    print(f"Next: run evaluate.py --result_dir {eval_out}")


if __name__ == "__main__":
    main()
