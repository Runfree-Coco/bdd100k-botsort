"""
Evaluate BoT-SORT tracking results on BDD100K using TrackEval.

Usage:
  cd /path/to/UCMCTrack-master    # TrackEval must be importable from here
  python /path/to/evaluate.py \
    --result_dir /path/to/tracking_results \
    --gt_dir     /path/to/eval/bdd100k_gt \
    --ucmc_root  /path/to/UCMCTrack-master

Note: TrackEval's eval_bdd() uses os.getcwd()/eval/ as TRACKERS_FOLDER,
so --ucmc_root (which must contain an eval/ subdirectory) is set as cwd.
"""

import sys
import os
import argparse
from pathlib import Path

EVAL_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']


def main():
    parser = argparse.ArgumentParser(description="Evaluate tracking results on BDD100K")
    parser.add_argument("--result_dir", required=True,
                        help="Directory containing per-sequence tracking JSON files")
    parser.add_argument("--gt_dir",     required=True,
                        help="TrackEval GT directory (bdd100k_gt/)")
    parser.add_argument("--ucmc_root",  required=True,
                        help="UCMCTrack-master root (contains eval/ with TrackEval code)")
    args = parser.parse_args()

    ucmc_root  = Path(args.ucmc_root).resolve()
    result_dir = Path(args.result_dir).resolve()
    gt_dir     = Path(args.gt_dir).resolve()

    if not result_dir.exists():
        print(f"Error: result directory not found: {result_dir}")
        return

    if not gt_dir.exists():
        print(f"Error: GT directory not found: {gt_dir}")
        return

    # eval_bdd resolves TRACKERS_FOLDER relative to cwd
    os.chdir(str(ucmc_root))
    sys.path.insert(0, str(ucmc_root))

    from eval.eval import eval_bdd

    # exp_name must match the subdirectory name inside ucmc_root/eval/
    exp_name = result_dir.name

    # Symlink result_dir into ucmc_root/eval/ if it's not already there
    eval_subdir = ucmc_root / "eval" / exp_name
    if not eval_subdir.exists():
        eval_subdir.symlink_to(result_dir)

    n_gt = len(list(gt_dir.glob("*.json")))
    n_tr = len(list(result_dir.glob("*.json")))
    print("=" * 60)
    print(f"BDD100K Evaluation  ({exp_name})")
    print(f"GT:      {gt_dir}  ({n_gt} sequences)")
    print(f"Results: {result_dir}  ({n_tr} sequences)")
    print(f"Classes: {EVAL_CLASSES}")
    print("=" * 60)

    if n_tr == 0:
        print("Error: no JSON files found in result_dir")
        return

    try:
        HOTA, IDF1, MOTA, AssA = eval_bdd(
            gt_folder=str(gt_dir),
            exp_name=exp_name,
            eval_classes=EVAL_CLASSES,
        )
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("Results (vehicle classes, mean):")
    print(f"  HOTA: {HOTA:.3f}")
    print(f"  IDF1: {IDF1:.3f}")
    print(f"  MOTA: {MOTA:.3f}")
    print(f"  AssA: {AssA:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
