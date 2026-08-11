#!/usr/bin/env python3
"""

Dump raw YOLO detections in MOTChallenge det format: to
- filter ignored regions
- prepare them for a rightful comparison between PuTR and ByteTrack (because PuTR only performs association).

Detections falling inside DETRAC ignored regions are removed here, so the
resulting file is the single shared input for both ByteTrack and PuTR.

Detections output is in MOT format

Example usage:
python.exe -m src.tracking.dump_detections -path/to/best.pt --split val --conf 0.05 --chunk 50
python.exe -m src.tracking.dump_detections -path/to/best.pt --split val --conf 0.05 --chunk 50

"""

import argparse
import gc
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from src.dataset.detrac_to_yolo import (
    ANNOTATIONS_DIR,
    DATA_ROOT,
    IMAGES_DIR,
    VAL_RATIO,
    assign_splits,
)
from src.dataset.detrac_to_mot import parse_ignored_regions

OUTPUT_DIR = DATA_ROOT / "detections"


def select_sequences(split, val_ratio):
    """Same sequence selection as detrac_to_mot.py (as in detrac_to_putr)."""
    if split == "val":
        annotations_dir = ANNOTATIONS_DIR["train"]
        sequences = sorted(path.stem for path in annotations_dir.glob("*.xml"))
        split_of = assign_splits(sequences, val_ratio)
        selected = [name for name in sequences if split_of[name] == "val"]
    else:
        annotations_dir = ANNOTATIONS_DIR["test"]
        selected = sorted(path.stem for path in annotations_dir.glob("*.xml"))

    return selected, annotations_dir


def frame_paths(sequence):
    return sorted((IMAGES_DIR / sequence).glob("img*.jpg"))


def regions_to_xyxy(regions):
    """Parse of ignored regions"""
    if not regions:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.array(regions, dtype=np.float32)
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def outside_ignored(boxes, regions, max_overlap):
    """
    
    To check if a detection falls inside an ignored region, the overlap ratio is measured
    regions (a box straddling two adjacent regions is still discarded).
    
    
    """
    if len(boxes) == 0 or len(regions) == 0:
        return np.ones(len(boxes), dtype=bool)

    left = np.maximum(boxes[:, None, 0], regions[None, :, 0])
    top = np.maximum(boxes[:, None, 1], regions[None, :, 1])
    right = np.minimum(boxes[:, None, 2], regions[None, :, 2])
    bottom = np.minimum(boxes[:, None, 3], regions[None, :, 3])

    intersection = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ratio = intersection.sum(axis=1) / np.maximum(areas, 1e-6)

    return ratio < max_overlap


def to_det_line(frame_number, box, score):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return (
        f"{frame_number},-1,{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},{score:.4f},-1,-1,-1"
    )


def release(model):
    """Free GPU memory manually to avoid OOM errors"""
    model.predictor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dump_sequence(model, sequence, regions, output_dir, args):
    images = frame_paths(sequence)
    if not images:
        return 0, 0

    lines = []
    n_dropped = 0
    frame_number = 0

    for start in range(0, len(images), args.chunk):
        chunk = images[start:start + args.chunk]

        results = model.predict(
            source=[str(path) for path in chunk],
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            quantize=16, #fp16 precision
            batch=args.batch,
            max_det=args.max_det,
            stream=True,
            verbose=False,
        )

        for result in results:
            frame_number += 1
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            del result

            keep = outside_ignored(boxes, regions, args.ignore_overlap)
            n_dropped += int((~keep).sum())
            boxes, scores = boxes[keep], scores[keep]

            for box, score in zip(boxes, scores):
                lines.append(to_det_line(frame_number, box, score))

        del results
        release(model)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{sequence}.txt").write_text("\n".join(lines) + "\n")
    return len(lines), n_dropped


def dump(args):
    sequences, annotations_dir = select_sequences(args.split, args.val_ratio)
    if not sequences:
        raise SystemExit(f"No sequences for split '{args.split}' in {annotations_dir}")

    model = YOLO(args.weights)
    model.overrides['end2end'] = False
    output_dir = OUTPUT_DIR / args.split

    total_boxes = 0
    total_dropped = 0
    total_frames = 0
    progress = tqdm(sequences, desc=f"Detecting {args.split}", unit="seq")
    for sequence in progress:
        progress.set_postfix_str(sequence)

        if args.keep_ignored:
            regions = np.zeros((0, 4), dtype=np.float32)
        else:
            regions = regions_to_xyxy(
                parse_ignored_regions(annotations_dir / f"{sequence}.xml")
            )

        n_boxes, n_dropped = dump_sequence(model, sequence, regions, output_dir, args)
        total_boxes += n_boxes
        total_dropped += n_dropped
        total_frames += len(frame_paths(sequence))

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            progress.write(f"{sequence}: {n_boxes} boxes, peak {peak:.2f} GiB")
            torch.cuda.reset_peak_memory_stats()
        release(model)

    print(f"{len(sequences)} sequences, {total_frames} frames, {total_boxes} boxes kept")
    print(f"{total_dropped} boxes dropped in ignored regions")
    print(f"Average {total_boxes / max(total_frames, 1):.1f} boxes/frame")
    print(f"Detections: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("--weights", required=True, help="YOLO checkpoint")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO, help="Same as other pre-processing scripts")
    parser.add_argument("--conf", type=float, default=0.05,
                        help="Confidence for detection (bytetracks prefers it low)")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU treshold for NMS")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="The same as the one used in training")
    parser.add_argument("--ignore-overlap", type=float, default=0.5,
                        help="Overlap treshold to filter ignored regions")
    parser.add_argument("--keep-ignored", action="store_true",
                        help="Keep ignored regions (not recommended)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--chunk", type=int, default=200,
                        help="Number of frames to process at once. Chunk=50 works well on a RTX 4090")
    parser.add_argument("--max-det", type=int, default=300,
                        help="Max detections per frame after NMS")
    parser.add_argument("--device", default=0)
    
    args = parser.parse_args()

    dump(args)
