"""
Grid search for BoT-SORT hyperparameters on BDD100K.

Searches track_thresh, match_thresh, second_match_thresh, track_buffer,
min_box_area, and n_dti (DTI interpolation frames). Results are saved to CSV.

Note: DTI interpolation was found to degrade performance at 5 fps (large FP
increase). n_dti=0 is recommended; the grid includes it for completeness.

Usage:
  python tools/param_search.py \
    --det_dir   /path/to/det_results/yolo_cls \
    --img_dir   /path/to/bdd100k/images/track/val \
    --gt_dir    /path/to/bdd100k/labels/box_track_20/val \
    --gt_folder /path/to/UCMCTrack-master/eval/bdd100k_gt \
    --ucmc_root /path/to/UCMCTrack-master \
    --out_csv   search_results.csv
"""

import sys
import os
import json
import csv
import shutil
import argparse
import itertools
import numpy as np
import cv2
from pathlib import Path

VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
EVAL_CLASSES    = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
IMG_W, IMG_H    = 1280, 720

PARAM_GRID = {
    "track_thresh":        [0.35, 0.40, 0.45],
    "match_thresh":        [0.75, 0.80, 0.85],
    "second_match_thresh": [0.40, 0.50],
    "track_buffer":        [60, 90],
    "min_box_area":        [200, 400],
    "n_dti":               [0, 10, 15],
}

FIXED = dict(mot20=True, n_min=3)


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


def assign_category(x1, y1, x2, y2, fid, gt_map, fallback="car"):
    gts = gt_map.get(fid, [])
    best_iou, best_cat = 0.0, fallback
    for (gx1, gy1, gx2, gy2, cat) in gts:
        score = iou_xyxy([x1, y1, x2, y2], [gx1, gy1, gx2, gy2])
        if score > best_iou:
            best_iou, best_cat = score, cat
    return best_cat if best_iou >= 0.3 else fallback


def run_tracker(params, det_dir, img_dir, gt_dir, tmp_mot, tmp_json):
    from basetrack import BaseTrack
    from byte_tracker import BYTETracker

    for d in [tmp_mot, tmp_json]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    class Args:
        pass
    a = Args()
    a.track_thresh        = params["track_thresh"]
    a.match_thresh        = params["match_thresh"]
    a.second_match_thresh = params["second_match_thresh"]
    a.track_buffer        = params["track_buffer"]
    a.mot20               = FIXED["mot20"]
    min_box_area          = params["min_box_area"]

    global_track_cat   = {}
    seq_total_frames   = {}

    for det_file in sorted(det_dir.glob("*.txt")):
        seq_name = det_file.stem
        BaseTrack._count = 0
        tracker = BYTETracker(a, frame_rate=5)
        tracker.gmc.reset()

        dets_by_frame = {}
        with open(det_file) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue
                fid  = int(parts[0])
                conf = float(parts[6])
                if conf <= 0:
                    continue
                x1, y1 = float(parts[2]), float(parts[3])
                w,  h  = float(parts[4]), float(parts[5])
                if w * h < min_box_area:
                    continue
                dets_by_frame.setdefault(fid, []).append([x1, y1, x1 + w, y1 + h, conf])

        gt_json = gt_dir / f"{seq_name}.json"
        gt_map  = load_gt_category_map(gt_json) if gt_json.exists() else {}
        total   = max(dets_by_frame.keys()) if dets_by_frame else 0
        if gt_json.exists():
            with open(gt_json) as f:
                total = len(json.load(f))
        seq_total_frames[seq_name] = total

        img_files = sorted((img_dir / seq_name).glob("*.jpg")) if (img_dir / seq_name).exists() else []
        output_frames = []
        mot_lines     = []
        seq_track_cat = {}

        for fid in range(1, total + 1):
            dets    = dets_by_frame.get(fid, [])
            dets_np = (np.array(dets, dtype=np.float32) if dets
                       else np.empty((0, 5), dtype=np.float32))
            frame_img = cv2.imread(str(img_files[fid - 1])) if fid <= len(img_files) else None
            online_targets = tracker.update(dets_np, [IMG_H, IMG_W], [IMG_H, IMG_W], img=frame_img)

            labels = []
            for t in online_targets:
                x1, y1, x2, y2 = t.tlbr
                x1 = max(0.0, float(x1)); y1 = max(0.0, float(y1))
                x2 = min(float(IMG_W), float(x2)); y2 = min(float(IMG_H), float(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                cat = assign_category(x1, y1, x2, y2, fid, gt_map)
                seq_track_cat[t.track_id] = cat
                w = x2 - x1; h = y2 - y1
                labels.append({"id": t.track_id, "category": cat,
                               "box2d": {"x1": round(x1,2), "y1": round(y1,2),
                                         "x2": round(x2,2), "y2": round(y2,2)}})
                mot_lines.append(f"{fid},{t.track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
            output_frames.append({"name": f"{seq_name}-{fid:07d}.jpg", "labels": labels})

        global_track_cat[seq_name] = seq_track_cat
        with open(tmp_json / f"{seq_name}.json", 'w') as f:
            json.dump(output_frames, f)
        with open(tmp_mot / f"{seq_name}.txt", 'w') as f:
            f.writelines(mot_lines)

    return global_track_cat, seq_total_frames


def mot_to_bdd_json(mot_dir, json_out, track_cat_all, seq_total_frames):
    if json_out.exists():
        shutil.rmtree(json_out)
    json_out.mkdir(parents=True)
    for txt_path in sorted(mot_dir.glob("*.txt")):
        seq_name   = txt_path.stem
        track_cat  = track_cat_all.get(seq_name, {})
        total      = seq_total_frames.get(seq_name, 202)
        try:
            data = np.loadtxt(str(txt_path), dtype=np.float64, delimiter=',')
        except Exception:
            data = np.empty((0, 10))
        if data.ndim == 1:
            data = data[np.newaxis, :]
        dets_by_frame = {}
        for row in data:
            fid = int(row[0]); tid = int(row[1])
            x1, y1 = float(row[2]), float(row[3])
            w,  h  = float(row[4]), float(row[5])
            x2, y2 = x1 + w, y1 + h
            x1 = max(0.0, x1); y1 = max(0.0, y1)
            x2 = min(float(IMG_W), x2); y2 = min(float(IMG_H), y2)
            if x2 > x1 and y2 > y1:
                dets_by_frame.setdefault(fid, []).append((tid, x1, y1, x2, y2))
        output_frames = []
        for fid in range(1, total + 1):
            labels = [{"id": tid, "category": track_cat.get(tid, "car"),
                       "box2d": {"x1": round(x1,2), "y1": round(y1,2),
                                 "x2": round(x2,2), "y2": round(y2,2)}}
                      for (tid, x1, y1, x2, y2) in dets_by_frame.get(fid, [])]
            output_frames.append({"name": f"{seq_name}-{fid:07d}.jpg", "labels": labels})
        with open(json_out / f"{seq_name}.json", 'w') as f:
            json.dump(output_frames, f)


def main():
    parser = argparse.ArgumentParser(description="BoT-SORT hyperparameter grid search")
    parser.add_argument("--det_dir",   required=True)
    parser.add_argument("--img_dir",   required=True)
    parser.add_argument("--gt_dir",    required=True)
    parser.add_argument("--gt_folder", required=True, help="TrackEval bdd100k_gt directory")
    parser.add_argument("--ucmc_root", required=True, help="UCMCTrack-master root")
    parser.add_argument("--out_csv",   default="search_results.csv")
    args = parser.parse_args()

    det_dir   = Path(args.det_dir)
    img_dir   = Path(args.img_dir)
    gt_dir    = Path(args.gt_dir)
    ucmc_root = Path(args.ucmc_root).resolve()
    gt_folder = args.gt_folder

    os.chdir(str(ucmc_root))
    sys.path.insert(0, str(ucmc_root))
    sys.path.insert(0, str(Path(__file__).parent.parent / "tracker"))

    from eval.eval import eval_bdd
    try:
        from eval.interpolation import interpolate
        HAS_INTERP = True
    except ImportError:
        HAS_INTERP = False
        print("Warning: interpolation module not found; n_dti>0 will be skipped")

    tmp_mot    = ucmc_root / "eval" / "_search_mot_tmp"
    tmp_interp = ucmc_root / "eval" / "_search_interp_tmp"
    tmp_json   = ucmc_root / "eval" / "_search_json_tmp"

    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    total  = len(combos)
    print(f"Total combinations: {total}. Starting search...\n")

    fieldnames = keys + ["HOTA", "IDF1", "MOTA", "AssA"]
    out_csv    = Path(args.out_csv)
    with open(out_csv, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    results       = []
    tracker_keys  = [k for k in keys if k != "n_dti"]
    seen_t_combos = {}

    for combo_idx, combo in enumerate(combos, 1):
        params     = dict(zip(keys, combo))
        n_dti      = params["n_dti"]
        t_key      = tuple(params[k] for k in tracker_keys)

        if t_key not in seen_t_combos:
            print(f"[Tracking {len(seen_t_combos)+1}] "
                  + ", ".join(f"{k}={params[k]}" for k in tracker_keys))
            seen_t_combos[t_key] = run_tracker(params, det_dir, img_dir, gt_dir, tmp_mot, tmp_json)

        track_cat_all, seq_total_frames = seen_t_combos[t_key]
        print(f"  [combo {combo_idx}/{total}] n_dti={n_dti}", end="", flush=True)

        eval_link = ucmc_root / "eval" / "_search_eval"
        if eval_link.exists():
            shutil.rmtree(eval_link)

        if n_dti == 0:
            shutil.copytree(str(tmp_json), str(eval_link))
        else:
            if not HAS_INTERP:
                print("  -> skipped (no interpolation module)")
                continue
            if tmp_interp.exists():
                shutil.rmtree(tmp_interp)
            tmp_interp.mkdir(parents=True)
            try:
                interpolate(str(tmp_mot), str(tmp_interp),
                            n_min=FIXED["n_min"], n_dti=n_dti)
                mot_to_bdd_json(tmp_interp, eval_link, track_cat_all, seq_total_frames)
            except Exception as e:
                print(f"  -> interpolation failed: {e}")
                if eval_link.exists():
                    shutil.rmtree(eval_link)
                continue

        try:
            HOTA, IDF1, MOTA, AssA = eval_bdd(gt_folder=gt_folder,
                                               exp_name="_search_eval",
                                               eval_classes=EVAL_CLASSES)
            print(f"  -> HOTA={HOTA:.3f} IDF1={IDF1:.3f} MOTA={MOTA:.3f} AssA={AssA:.3f}")
        except Exception as e:
            print(f"  -> eval failed: {e}")
            HOTA = IDF1 = MOTA = AssA = None
        finally:
            if eval_link.exists():
                shutil.rmtree(eval_link)

        row = {**params,
               "HOTA": round(HOTA, 4) if HOTA is not None else "",
               "IDF1": round(IDF1, 4) if IDF1 is not None else "",
               "MOTA": round(MOTA, 4) if MOTA is not None else "",
               "AssA": round(AssA, 4) if AssA is not None else ""}
        results.append(row)
        with open(out_csv, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    for d in [tmp_mot, tmp_interp]:
        if d.exists():
            shutil.rmtree(d)

    valid = sorted([r for r in results if r["MOTA"] != ""],
                   key=lambda r: r["MOTA"], reverse=True)
    print("\n" + "=" * 70)
    print("Top 5 by MOTA:")
    for r in valid[:5]:
        print("  MOTA={MOTA:.4f} HOTA={HOTA:.4f} IDF1={IDF1:.4f} | ".format(**r)
              + "  ".join(f"{k}={r[k]}" for k in keys))
    print(f"\nFull results saved to: {out_csv}")


if __name__ == "__main__":
    main()
