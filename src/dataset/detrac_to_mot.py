#!/usr/bin/env python3
"""
Example usage:
    python detrac_to_mot.py val     
    python detrac_to_mot.py test
    python detrac_to_mot.py both

    Specify --val-ratio to be the same as the one used in detection. 
    Default for both is 0.2 (80% train 20% val)
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

from src.dataset.detrac_to_yolo import (
    ANNOTATIONS_DIR,
    DATA_ROOT,
    IMAGES_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    VAL_RATIO,
    assign_splits,
)

OUTPUT_DIR = DATA_ROOT / "trackeval"
DATASET_NAME = "DETRAC"
FRAME_RATE = 25 #videos are at 25fps in detrac



def parse_box(element):
    return (
        float(element.get("left")),
        float(element.get("top")),
        float(element.get("width")),
        float(element.get("height")),
    )


def parse_tracks(xml_path):
    """
    Parse XML to (id, left, top, width, height) for to_mot_line
    """
    root = ET.parse(xml_path).getroot()
    rows = []
    for frame in root.findall("frame"):
        frame_number = int(frame.get("num"))
        for target in frame.iter("target"):
            left, top, width, height = parse_box(target.find("box"))
            rows.append((frame_number, int(target.get("id")), left, top, width, height))
    return sorted(rows)


def parse_ignored_regions(xml_path):
    """Parse of ignored region to be filtered in prediction"""
    root = ET.parse(xml_path).getroot()
    return [parse_box(box) for box in root.findall("ignored_region/box")]



def to_mot_line(row):
    """
    Convert ground truth to MOTChallenge2D format
    Riga ground truth MOTChallenge:
    frame, id, left, top, width, height, confidence, class, visibility

    Visibility=1 because TrackEval pre-processing is not needed
    """
    frame_number, track_id, left, top, width, height = row
    return f"{frame_number},{track_id},{left:.2f},{top:.2f},{width:.2f},{height:.2f},1,1,1"


def to_seqinfo(sequence, sequence_length):
    """Seqinfo.ini as needed by TrackEval"""
    return (
        "[Sequence]\n"
        f"name={sequence}\n"
        "imDir=img1\n"
        f"frameRate={FRAME_RATE}\n"
        f"seqLength={sequence_length}\n"
        f"imWidth={IMAGE_WIDTH}\n"
        f"imHeight={IMAGE_HEIGHT}\n"
        "imExt=.jpg\n"
    )


def sequence_length(sequence, tracks):
    image_dir = IMAGES_DIR / sequence
    if image_dir.is_dir():
        count = len(list(image_dir.glob("img*.jpg")))
        if count:
            return count
    return max(row[0] for row in tracks) if tracks else 0


def write_sequence(sequence, tracks, split_dir):
    """Writes gt.txt and seqinfo.ini"""
    sequence_dir = split_dir / sequence
    (sequence_dir / "gt").mkdir(parents=True, exist_ok=True)

    lines = [to_mot_line(row) for row in tracks]
    (sequence_dir / "gt" / "gt.txt").write_text("\n".join(lines) + "\n")
    (sequence_dir / "seqinfo.ini").write_text(
        to_seqinfo(sequence, sequence_length(sequence, tracks))
    )


def write_seqmap(split, sequences):
    """
    Writes the sequenca map to the correct folder as required by TrackEval
    """
    seqmap_dir = OUTPUT_DIR / "gt" / "mot_challenge" / "seqmaps"
    seqmap_dir.mkdir(parents=True, exist_ok=True)

    seqmap_path = seqmap_dir / f"{DATASET_NAME}-{split}.txt"
    seqmap_path.write_text("name\n" + "\n".join(sequences) + "\n")
    return seqmap_path


def select_sequences(split, val_ratio):
    """
    Select sequences (videos) for validation set or test set
    """
    if split == "val":
        annotations_dir = ANNOTATIONS_DIR["train"]
        sequences = sorted(path.stem for path in annotations_dir.glob("*.xml"))
        split_of = assign_splits(sequences, val_ratio) #as for detection
        selected = [name for name in sequences if split_of[name] == "val"]
    else:
        annotations_dir = ANNOTATIONS_DIR["test"]
        selected = sorted(path.stem for path in annotations_dir.glob("*.xml"))

    return selected, annotations_dir


def convert(split, val_ratio=VAL_RATIO):
    """Executes the full conversion pipeline"""
    sequences, annotations_dir = select_sequences(split, val_ratio)

    if not sequences:
        raise SystemExit(f"No sequences for split '{split}' in {annotations_dir}")

    split_dir = OUTPUT_DIR / "gt" / "mot_challenge" / f"{DATASET_NAME}-{split}"
    split_dir.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "trackers" / "mot_challenge" / f"{DATASET_NAME}-{split}").mkdir(
        parents=True, exist_ok=True
    )

    ignored_regions = {}
    total_boxes = 0

    for sequence in tqdm(sequences, desc=f"Converting {split} ...", unit="seq"):
        xml_path = annotations_dir / f"{sequence}.xml"
        tracks = parse_tracks(xml_path)
        write_sequence(sequence, tracks, split_dir)

        ignored_regions[sequence] = parse_ignored_regions(xml_path)
        total_boxes += len(tracks)

    seqmap_path = write_seqmap(split, sequences)
    save_ignored_regions(ignored_regions)

    print(f"{len(sequences)} videos, {total_boxes} annotated boxes")
    print(f"Ground truth: {split_dir}")
    print(f"Seqmap: {seqmap_path}")


def save_ignored_regions(regions):
    """Save ignored regions file as json, to be used in inference"""
    path = OUTPUT_DIR / "ignored_regions.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(regions)
    path.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=["val", "test", "both"])
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    args = parser.parse_args()

    splits = ["val", "test"] if args.split == "both" else [args.split]
    for split in splits:
        convert(split, args.val_ratio)