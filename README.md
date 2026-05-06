# BDD100K Vehicle Multi-Object Tracking with BoT-SORT

> Undergraduate thesis project — Zhengzhou University, School of Electrical and Information Engineering

Fine-tunes YOLOv8 on BDD100K vehicle detection and pairs it with a customized **BoT-SORT** tracker (ByteTrack + GMC camera compensation) to perform end-to-end multi-object tracking on the BDD100K benchmark.

## Results

| Stage | Tracker | Detector | MOTA | IDF1 | HOTA | AssA |
|-------|---------|----------|------|------|------|------|
| GT upper bound | — | Ground Truth | 75.39% | 64.24% | 67.47% | 53.54% |
| Baseline | UCMCTrack | YOLOv8l (fine-tuned) | 27.19% | 33.13% | 30.85% | 35.21% |
| Migrate to ByteTrack | ByteTrack | YOLOv8l | 30.10% | 36.52% | 31.28% | 36.17% |
| Add BoT-SORT + GMC | BoT-SORT | YOLOv8l | 36.75% | 42.38% | 37.18% | 42.11% |
| Re-fine-tune detector | BoT-SORT | YOLOv8l (imgsz=1280) | 37.49% | 42.40% | 37.31% | 42.24% |
| Grid search params | BoT-SORT | YOLOv8l | 38.66% | 42.96% | 37.84% | 43.38% |
| Fix category labels | BoT-SORT | YOLOv8l | 39.83% | 43.40% | 37.68% | 42.82% |
| **Short-track filter** | **BoT-SORT** | **YOLOv8l** | **40.07%** | **43.51%** | **37.74%** | **43.20%** |

Evaluated on BDD100K val set (200 sequences, 5 fps dashcam footage).

##Visualization





![跟踪演示](469a97c29dd3f93fe22e884ea4d41430.gif)







## Key Contributions

- **Switched from UCMCTrack to BoT-SORT**: UCMCTrack's ground-plane projection fails at dashcam 5 fps; BoT-SORT with GMC camera compensation is far more robust.
- **Per-class detection thresholds**: motorcycle/bicycle are severely under-detected; class-specific thresholds (`car=0.45`, `moto/bicycle=0.25`) significantly improve recall for rare categories.
- **Short-lived track filter**: tracks appearing in ≤2 frames are discarded as false positives (motorcycle/bicycle are exempted to avoid suppressing rare valid tracks).
- **DTI interpolation is harmful at 5 fps**: generates ghost boxes in gaps, dropping MOTA by ~11%. Disabled.
- **216-combo grid search**: systematic search over `track_thresh`, `match_thresh`, `track_buffer`, `min_box_area`, `n_dti`.

## Repository Structure

```
bdd100k-botsort/
├── run_tracking.py          # Main tracking script (BoT-SORT on BDD100K val)
├── evaluate.py              # Evaluation using TrackEval (MOTA/HOTA/IDF1/AssA)
├── visualize.py             # Render tracking results as video
├── configs/
│   └── botsort_bdd100k.yaml # Best hyperparameters
├── tracker/                 # BoT-SORT core (ByteTrack + GMC)
│   ├── byte_tracker.py
│   ├── kalman_filter.py
│   ├── gmc.py               # Global Motion Compensation
│   ├── matching.py
│   └── basetrack.py
├── detector/
│   ├── train.py             # YOLOv8 fine-tuning on BDD100K detection
│   └── inference.py         # Run YOLOv8 inference to produce MOT detection files
├── data/
│   └── prepare_dataset.py   # Convert BDD100K JSON labels to YOLO format
└── tools/
    └── param_search.py      # Hyperparameter grid search
```

## Setup

### Requirements

- Python 3.10+
- CUDA GPU (fine-tuning requires ≥12 GB VRAM at imgsz=1280; RTX 3050 4 GB is sufficient for tracking only)

```bash
pip install -r requirements.txt
```

### TrackEval (for evaluation)

The evaluation script requires [TrackEval](https://github.com/JonathonLuiten/TrackEval) via the UCMCTrack wrapper. Clone UCMCTrack and ensure `eval/eval.py` is importable:

```bash
git clone https://github.com/Tracygc/UCMCTrack UCMCTrack-master
```

### Data

Download from the [BDD100K website](https://bdd-data.berkeley.edu/):

| File | Purpose |
|------|---------|
| `bdd100k_images_track.zip` | Tracking val images (200 sequences) |
| `bdd100k_labels_release.zip` | Tracking annotations (GT JSON) |
| `bdd100k_images_100k.zip` | Detection images (for fine-tuning) |
| `bdd100k_det_labels.zip` | Detection annotations (for fine-tuning) |

### Model Weights

Download the fine-tuned YOLOv8l weights from [Releases](../../releases) and place as `best.pt`.

## Usage

### Step 1: Prepare Detection Dataset (for training only)

```bash
python data/prepare_dataset.py \
  --img_root   /path/to/bdd100k_images_100k/100k \
  --label_root /path/to/bdd100k/100k \
  --out_dir    bdd_yolo_dataset \
  --train_n    -1 \
  --val_n      -1
```

### Step 2: Fine-tune YOLOv8 (for training only)

```bash
# Requires ≥12 GB VRAM at imgsz=1280; reduce imgsz/batch for smaller GPUs
python detector/train.py \
  --dataset_yaml bdd_yolo_dataset/bdd100k_det.yaml \
  --model yolov8l.pt \
  --epochs 50 \
  --batch 4 \
  --imgsz 1280
```

### Step 3: Run YOLOv8 Inference

```bash
python detector/inference.py \
  --model   best.pt \
  --img_dir /path/to/bdd100k/images/track/val \
  --out_dir det_results/yolo_cls \
  --conf    0.15 \
  --imgsz   1280
```

### Step 4: Run BoT-SORT Tracking

```bash
python run_tracking.py \
  --det_dir det_results/yolo_cls \
  --img_dir /path/to/bdd100k/images/track/val \
  --gt_dir  /path/to/bdd100k/labels/box_track_20/val \
  --out_dir tracking_results/botsort_final
```

### Step 5: Evaluate

```bash
python evaluate.py \
  --result_dir tracking_results/botsort_final \
  --gt_dir     UCMCTrack-master/eval/bdd100k_gt \
  --ucmc_root  UCMCTrack-master
```

### Visualize a Sequence

```bash
python visualize.py \
  --json_file  tracking_results/botsort_final/b2505382-2905b23c.json \
  --image_dir  /path/to/bdd100k/images/track/val/b2505382-2905b23c \
  --output     demo.mp4 \
  --fps        5
```

## Hyperparameter Tuning

```bash
python tools/param_search.py \
  --det_dir   det_results/yolo_cls \
  --img_dir   /path/to/bdd100k/images/track/val \
  --gt_dir    /path/to/bdd100k/labels/box_track_20/val \
  --gt_folder UCMCTrack-master/eval/bdd100k_gt \
  --ucmc_root UCMCTrack-master \
  --out_csv   search_results.csv
```

## Vehicle Categories

| cls_id | Category   | Track Threshold |
|--------|------------|-----------------|
| 0      | car        | 0.45            |
| 1      | truck      | 0.40            |
| 2      | bus        | 0.40            |
| 3      | motorcycle | 0.25            |
| 4      | bicycle    | 0.25            |

## Acknowledgements

- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) — tracker foundation
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — two-stage association
- [BDD100K](https://bdd-data.berkeley.edu/) — dataset
- [UCMCTrack](https://github.com/Tracygc/UCMCTrack) — TrackEval wrapper
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — detector

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{cheng2026bdd100kbotsort,
  title  = {Low-Frame-Rate Multi-Object Vehicle Tracking:
Handling Large Displacement and Camera Motion},
  author = {Cheng Xianya},
  year   = {2026},
  url    = {https://github.com/Runfree-Coco/bdd100k-botsort}
}
```
