#!/usr/bin/env python3
"""
Execute with:
python tracking_inference.py --config path/to/config/config.json

Specify split (val/test) and other params in the config file
"""

import argparse
import json
from pathlib import Path

import json

from tqdm import tqdm
from ultralytics import YOLO

IMAGES_DIR = Path("./data/raw/DETRAC-Images")
BENCHMARK = "DETRAC"
OUTPUT_DIR = Path("./data/trackeval")

IGNORED_OVERLAP = 0.5


def is_ignored(box, regions, threshold=IGNORED_OVERLAP):
    """
    Returns true if predicted box overlaps with ignored region above a treshold
    Needed to avoid false positives in TrackEval: ignored regions are NOT annotated in DETRAC
    """
    left, top, width, height = box
    area = width * height

    if area <= 0:
        return False

    for region_left, region_top, region_width, region_height in regions:
        inter_width = max(0.0, min(left + width, region_left + region_width)
                          - max(left, region_left))
        inter_height = max(0.0, min(top + height, region_top + region_height)
                           - max(top, region_top))
        if (inter_width * inter_height) / area > threshold:
            return True
    return False



def run_inference(config):
    seqmap = (OUTPUT_DIR / "gt" / "mot_challenge" / "seqmaps"
        / f"{BENCHMARK}-{config['split']}.txt")
    
    if not seqmap.exists():
        raise SystemExit(f"Seqmap not found: {seqmap}. Executes detrac_to_mot.py first")
    sequences = [line.strip() for line in seqmap.read_text().split("\n")[1:] if line.strip()]

    ignored_regions = json.loads((OUTPUT_DIR / "ignored_regions.json").read_text())

    output_dir = (OUTPUT_DIR / config['pred_name'] / "mot_challenge"
                    / f"{BENCHMARK}-{config['split']}" / config['tracker_name'] / "data")
    output_dir.mkdir(parents=True, exist_ok=True)


    model = YOLO(config['weights'])
    model.overrides['end2end'] = config['end2end']
    total_kept = 0
    total_skipped = 0

    progress = tqdm(sequences, desc=f"Tracking {config['split']}", unit="seq")

    for sequence in progress:
        progress.set_postfix_str(sequence)
        regions = ignored_regions.get(sequence, [])
        lines = []

        #reset ONCE for sequence
        predictor = getattr(model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) or []:
            tracker.reset()

        for path in sorted((IMAGES_DIR / sequence).glob("img*.jpg")):

            result = model.track(source=str(path), tracker=config['tracker_config_path'],
                                 persist=True, conf=config['conf'], verbose=False)[0]
            if result.boxes is None or result.boxes.id is None:
                continue

            frame = int(path.stem.replace("img", ""))

            #creation of pred.txt
            for track_id, xyxy, score in zip(result.boxes.id.int().tolist(),
                                             result.boxes.xyxy.cpu().numpy(),
                                             result.boxes.conf.cpu().tolist()):
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                box = (x1, y1, x2 - x1, y2 - y1)
                if is_ignored(box, regions):
                    total_skipped += 1
                    continue

                # MOT2D: frame, id, left, top, width, height, conf, -1, -1, -1
                lines.append(f"{frame},{track_id},{box[0]:.2f},{box[1]:.2f},"
                             f"{box[2]:.2f},{box[3]:.2f},{score:.4f},-1,-1,-1")
                total_kept += 1

        (output_dir / f"{sequence}.txt").write_text("\n".join(lines) + "\n")

    print(f"{len(sequences)} sequenze, {total_kept} detection scritte, "
          f"{total_skipped} scartate nelle ignored_region")
    print(f"Predizioni: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True, help="inference config file")
    args = parser.parse_args()

    with open(args.config, 'rb') as f:
        config = json.load(f)

    run_inference(config)