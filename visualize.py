"""
Render tracking results onto BDD100K image frames and save as video.

Usage:
  python visualize.py \
    --json_file  /path/to/tracking_results/b2505382-2905b23c.json \
    --image_dir  /path/to/bdd100k/images/track/val/b2505382-2905b23c \
    --output     output.mp4 \
    --fps        5
"""

import cv2
import json
import argparse
from pathlib import Path

COLOR_MAP = {
    'car':        (0,   255,   0),    # green
    'truck':      (0,     0, 255),    # red
    'bus':        (255,   0,   0),    # blue
    'motorcycle': (0,   255, 255),    # yellow
    'bicycle':    (0,   255, 255),    # yellow
}
DEFAULT_COLOR = (255, 255, 255)


def draw_boxes(frame, labels):
    for label in labels:
        track_id = label['id']
        category = label['category']
        bbox     = label['box2d']
        x1, y1   = int(bbox['x1']), int(bbox['y1'])
        x2, y2   = int(bbox['x2']), int(bbox['y2'])
        color    = COLOR_MAP.get(category, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{category} {track_id}", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Visualize BoT-SORT tracking results")
    parser.add_argument("--json_file",  required=True, help="Tracking result JSON for one sequence")
    parser.add_argument("--image_dir",  required=True, help="Directory with JPG frames for that sequence")
    parser.add_argument("--output",     default="tracking.mp4", help="Output video path")
    parser.add_argument("--fps",        type=int, default=5, help="Output video FPS (default: 5)")
    parser.add_argument("--width",      type=int, default=1280)
    parser.add_argument("--height",     type=int, default=720)
    args = parser.parse_args()

    with open(args.json_file, 'r', encoding='utf-8') as f:
        frames_data = json.load(f)

    frame_labels = {}
    for frame_info in frames_data:
        name      = frame_info['name']
        frame_num = int(name.split('-')[-1].split('.')[0])
        frame_labels[frame_num] = frame_info.get('labels', [])

    img_files = sorted(Path(args.image_dir).glob("*.jpg"))
    if not img_files:
        print(f"No images found in: {args.image_dir}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))

    for i, img_path in enumerate(img_files):
        frame_id = i + 1
        frame    = cv2.imread(str(img_path))
        if frame is None:
            continue
        frame = draw_boxes(frame, frame_labels.get(frame_id, []))
        out.write(frame)
        if i % 50 == 0:
            print(f"Processing: {i+1}/{len(img_files)}")

    out.release()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
