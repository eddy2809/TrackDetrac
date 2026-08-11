#!/usr/bin/env python3
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

# All classes are 'vehicle', better for tracking. Change if needed
classes = ["vehicle"]
classes_ids = 0

# All images are 960x540
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 540

#fixed dirs
DATA_ROOT = Path("data")
IMAGES_DIR = DATA_ROOT / "raw" / "DETRAC-Images"
OUTPUT_DIR = DATA_ROOT / "processed"
ANNOTATIONS_DIR = {
    "train": DATA_ROOT / "raw" / "DETRAC-Train-Annotations-XML",
    "test": DATA_ROOT / "raw" / "DETRAC-Test-Annotations-XML",
}

#default values for split and sampling
VAL_RATIO = 0.2
SAMPLE_RATE = 1

#for reproducibility
SEED = 42

def parse_annotations(xml_path):
    """Parse XML to a dict"""
    root = ET.parse(xml_path).getroot()
    frames = {}
    for frame in root.findall("frame"):
        frames[int(frame.get("num"))] = [
            (
                classes_ids,
                float(target.find("box").get("left")),
                float(target.find("box").get("top")),
                float(target.find("box").get("width")),
                float(target.find("box").get("height")),
            )
            for target in frame.iter("target")
        ]
    return frames


def sample_frames(frames, sample_rate):
    """Returns 1 frame every sample_rate."""
    if sample_rate <= 1:
        return frames
    return {
        number: boxes
        for number, boxes in frames.items() if (number - 1) % sample_rate == 0
    }


def to_yolo_line(box):
    """Conversion from UAC-Detrac coordinates to YOLO ones."""
    class_id, left, top, width, height = box

    x_min = max(0.0, left)
    y_min = max(0.0, top)
    x_max = min(float(IMAGE_WIDTH), left + width)
    y_max = min(float(IMAGE_HEIGHT), top + height)

    if x_max <= x_min or y_max <= y_min:
        return None

    x_center = (x_min + x_max) / 2 / IMAGE_WIDTH
    y_center = (y_min + y_max) / 2 / IMAGE_HEIGHT
    box_width = (x_max - x_min) / IMAGE_WIDTH
    box_height = (y_max - y_min) / IMAGE_HEIGHT

    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def to_yolo_label(boxes):
    lines = [line for box in boxes if (line := to_yolo_line(box))]
    return "\n".join(lines) + "\n"


def assign_splits(sequences, val_ratio):
    """Train/Val split (on videos)"""
    shuffled = sorted(sequences)
    random.Random(SEED).shuffle(shuffled)
    val_size = round(len(shuffled) * val_ratio)
    return {
        sequence: "val" if i < val_size else "train"
        for i, sequence in enumerate(shuffled)
    }


def write_frame(sequence, frame_number, boxes, split):
    """Writes a single frame"""
    source_image = IMAGES_DIR / sequence / f"img{frame_number:05d}.jpg"
    if not source_image.exists():
        return False

    name = f"{sequence}_img{frame_number:05d}"

    destination_image = OUTPUT_DIR / "images" / split / f"{name}.jpg"
    if not destination_image.exists():
        shutil.copy2(source_image, destination_image)

    (OUTPUT_DIR / "labels" / split / f"{name}.txt").write_text(to_yolo_label(boxes))
    return True


def write_dataset_config():
    """Write detrac.yaml file for training YOLO"""
    splits = sorted(path.name for path in (OUTPUT_DIR / "images").iterdir() if path.is_dir())
    entries = "".join(f"{split}: images/{split}\n" for split in splits)
    names = "".join(f"  {i}: {name}\n" for i, name in enumerate(classes))

    config_path = OUTPUT_DIR / "detrac.yaml"
    config_path.write_text(f"path: {OUTPUT_DIR.resolve()}\n{entries}\nnames:\n{names}")
    return config_path


def convert(subset, val_ratio=VAL_RATIO, sample_rate=SAMPLE_RATE):
    """Converts the whole set (train/val/test) of frames to YOLO format"""
    annotations_dir = ANNOTATIONS_DIR[subset]
    sequences = sorted(path.stem for path in annotations_dir.glob("*.xml"))
    if not sequences:
        raise SystemExit(f"Nessun XML trovato in {annotations_dir}")

    if subset == "train":
        split_of = assign_splits(sequences, val_ratio)
    else:
        split_of = {sequence: "test" for sequence in sequences}

    frame_counts = dict.fromkeys(set(split_of.values()), 0)
    for split in frame_counts:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    progress = tqdm(sequences, desc=f"Conversione {subset}", unit="seq")
    for sequence in progress:
        split = split_of[sequence]
        progress.set_postfix_str(f"{sequence} -> {split}")
        frames = parse_annotations(annotations_dir / f"{sequence}.xml")
        frames = sample_frames(frames, sample_rate)
        for frame_number, boxes in frames.items():
            if write_frame(sequence, frame_number, boxes, split):
                frame_counts[split] += 1

    config_path = write_dataset_config()
    summary = ", ".join(f"{count} frame in {split}" for split, count in frame_counts.items())
    print(f"{len(sequences)} sequenze -> {summary}")
    print(f"Config: {config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subset", choices=["train", "test"])
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                        help="Keep 1 frame every sample-rate")
    args = parser.parse_args()
    convert(args.subset, args.val_ratio, args.sample_rate)