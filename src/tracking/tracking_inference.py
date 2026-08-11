"""
Inference with ByteTrack with the output of dump_detections.py.

Example usage:
python tracking_inference.py --config path/to/config/config.json

Specify params in the config file.

"""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml


def yaml_load(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


IMAGES_DIR = Path("./data/raw/DETRAC-Images")
DETECTIONS_DIR = Path("./data/detections")
BENCHMARK = "DETRAC"
OUTPUT_DIR = Path("./data/trackeval")

FRAME_RATE = 25  # videos are at 25fps in detrac


class Detections:
    def __init__(self, boxes, scores):
        self.xyxy = boxes.astype(np.float32)
        self.conf = scores.astype(np.float32)
        self.cls = np.zeros(len(boxes), dtype=np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return Detections(self.xyxy[mask], self.conf[mask])

    @property
    def xywh(self):
        boxes = np.empty_like(self.xyxy)
        boxes[:, 0] = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2
        boxes[:, 1] = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2
        boxes[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        boxes[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        return boxes


def load_detections(path, min_conf):
    """
    Creates a Detections Object for every frame
    """
    by_frame = {}
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        fields = line.split(",")
        frame = int(fields[0])
        left, top, width, height = (float(v) for v in fields[2:6])
        score = float(fields[6])
        if score < min_conf:
            continue
        by_frame.setdefault(frame, []).append(
            (left, top, left + width, top + height, score)
        )

    return {
        frame: Detections(np.array(rows, dtype=np.float32)[:, :4],
                          np.array(rows, dtype=np.float32)[:, 4])
        for frame, rows in by_frame.items()
    }


def empty_detections():
    """Frames with no detection still need an update() call as ByteTrack expects it"""
    return Detections(np.zeros((0, 4), dtype=np.float32),
                      np.zeros(0, dtype=np.float32))


def build_tracker(tracker_config_path):
    """
    Same config file used by model.track()
    """
    
    cfg = IterableSimpleNamespace(**yaml_load(check_yaml(tracker_config_path)))
    
    
    #useful for possible future usage of other trackers (e.g SORT)
    if cfg.tracker_type != "bytetrack":
        raise SystemExit(f"Expected bytetrack, got {cfg.tracker_type}")

    if "frame_rate" in inspect.signature(BYTETracker.__init__).parameters:
        return BYTETracker(args=cfg, frame_rate=FRAME_RATE)
    return BYTETracker(args=cfg)


def run_inference(config):
    seqmap = (OUTPUT_DIR / "gt" / "mot_challenge" / "seqmaps"
        / f"{BENCHMARK}-{config['split']}.txt")

    if not seqmap.exists():
        raise SystemExit(f"Seqmap not found: {seqmap}. Executes detrac_to_mot.py first")
    sequences = [line.strip() for line in seqmap.read_text().split("\n")[1:] if line.strip()]

    detections_dir = Path(config.get("detections_dir", DETECTIONS_DIR)) / config["split"]
    if not detections_dir.is_dir():
        raise SystemExit(f"Detections not found: {detections_dir}. Executes dump_detections.py first")

    output_dir = (OUTPUT_DIR / config['pred_name'] / "mot_challenge"
                    / f"{BENCHMARK}-{config['split']}" / config['tracker_name'] / "data")
    output_dir.mkdir(parents=True, exist_ok=True)

    min_conf = config.get("conf", 0.0)
    total_kept = 0
    total_frames = 0

    progress = tqdm(sequences, desc=f"Tracking {config['split']}", unit="seq")

    for sequence in progress:
        progress.set_postfix_str(sequence)
        lines = []

        # a new tracker is created for every sequence (to avoid leaks)
        tracker = build_tracker(config['tracker_config_path'])
        detections = load_detections(detections_dir / f"{sequence}.txt", min_conf)

        images = sorted((IMAGES_DIR / sequence).glob("img*.jpg"))
        for position, path in enumerate(images, start=1):
            tracks = tracker.update(detections.get(position, empty_detections()), None)
            total_frames += 1
            if len(tracks) == 0:
                continue

            frame = int(path.stem.replace("img", ""))

            # creation of pred.txt
            for track in tracks:
                x1, y1, x2, y2 = (float(v) for v in track[:4])
                track_id, score = int(track[4]), float(track[5])
                box = (x1, y1, x2 - x1, y2 - y1)

                # MOT2D format as pre-processing files
                lines.append(f"{frame},{track_id},{box[0]:.2f},{box[1]:.2f},"
                             f"{box[2]:.2f},{box[3]:.2f},{score:.4f},-1,-1,-1")
                total_kept += 1

        (output_dir / f"{sequence}.txt").write_text("\n".join(lines) + "\n")

    print(f"{len(sequences)} sequences, {total_frames} frame, {total_kept} track")
    print(f"Detection: {detections_dir}")
    print(f"Predictions: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True, help="inference config file")
    args = parser.parse_args()

    with open(args.config, 'rb') as f:
        config = json.load(f)

    run_inference(config)