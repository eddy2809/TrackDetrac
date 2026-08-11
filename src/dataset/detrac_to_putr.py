"""
Example usage:
    python detrac_to_putr.py train --sample-rate 10
    python detrac_to_putr.py val
    python detrac_to_putr.py test

Frames are renumbered 1..N with 6 digits because PuTR requires it (data/mot17.py)
"""

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

from detrac_to_yolo import (
    ANNOTATIONS_DIR,
    DATA_ROOT,
    IMAGES_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SAMPLE_RATE,
    VAL_RATIO,
    assign_splits,
    sample_frames,
)
from detrac_to_mot import FRAME_RATE, parse_tracks, to_mot_line

OUTPUT_DIR = DATA_ROOT / "putr" / "DETRAC"
DETECTIONS_DIR = DATA_ROOT / "detections"
DET_NAME = "det.txt"


def select_sequences(split, val_ratio):
    """Same selection as detrac_to_mot.py and dump_detections.py, same seed used."""
    if split == "test":
        annotations_dir = ANNOTATIONS_DIR["test"]
        return sorted(path.stem for path in annotations_dir.glob("*.xml")), annotations_dir

    annotations_dir = ANNOTATIONS_DIR["train"]
    sequences = sorted(path.stem for path in annotations_dir.glob("*.xml"))
    split_of = assign_splits(sequences, val_ratio)
    return [name for name in sequences if split_of[name] == split], annotations_dir


def frame_mapping(sequence, sample_rate):
    """
    Returns {original_frame_number: new_frame_number}, new numbers contiguous from 1. 
    """
    paths = {
        int(path.stem.replace("img", "")): path
        for path in (IMAGES_DIR / sequence).glob("img*.jpg")
    }
    paths = sample_frames(paths, sample_rate)
    return {
        original: (index, paths[original])
        for index, original in enumerate(sorted(paths), start=1)
    }


def write_images(mapping, sequence_dir, hardlink):
    image_dir = sequence_dir / "img1"
    image_dir.mkdir(parents=True, exist_ok=True)

    for new_number, source in mapping.values():
        destination = image_dir / f"{new_number:06d}.jpg"
        if destination.exists():
            continue
        
        # hardlink or copy to the new location
        if hardlink:
            destination.hardlink_to(source)
        else:
            shutil.copy2(source, destination)


def write_gt(tracks, mapping, sequence_dir):
    """
    Ground truth with renumbered frames in 9-field MOT format
    """
    (sequence_dir / "gt").mkdir(parents=True, exist_ok=True)

    lines = []
    for original_frame, track_id, left, top, width, height in tracks:
        if original_frame not in mapping:
            continue
        new_frame = mapping[original_frame][0]
        lines.append(to_mot_line((new_frame, track_id, left, top, width, height)))

    (sequence_dir / "gt" / "gt.txt").write_text("\n".join(lines) + "\n")
    return len(lines)


def write_detections(sequence, sequence_dir, split):
    """
    Copies the dump produced by dump_detections.py. PuTR only performs association
    """
    source = DETECTIONS_DIR / split / f"{sequence}.txt"
    if not source.exists():
        raise SystemExit(f"Detections not found: {source}. Executes dump_detections.py first")

    (sequence_dir / "det").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, sequence_dir / "det" / DET_NAME)


def write_seqinfo(sequence, sequence_dir, n_frames, frame_rate):
    (sequence_dir / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={sequence}\n"
        "imDir=img1\n"
        f"frameRate={frame_rate}\n"
        f"seqLength={n_frames}\n"
        f"imWidth={IMAGE_WIDTH}\n"
        f"imHeight={IMAGE_HEIGHT}\n"
        "imExt=.jpg\n"
    )


def write_seqmap(split, sequences):
    path = OUTPUT_DIR / f"{split}_seqmap.txt"
    path.write_text("name\n" + "\n".join(sequences) + "\n")
    return path


def convert(split, val_ratio, sample_rate, hardlink):
    sequences, annotations_dir = select_sequences(split, val_ratio)
    if not sequences:
        raise SystemExit(f"No sequences for split '{split}' in {annotations_dir}")

    # only the training split is subsampled, val and test stay full
    rate = sample_rate if split == "train" else 1
    frame_rate = FRAME_RATE / rate

    split_dir = OUTPUT_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_boxes = 0

    for sequence in tqdm(sequences, desc=f"Preparing {split}", unit="seq"):
        mapping = frame_mapping(sequence, rate)
        if not mapping:
            raise SystemExit(f"No images found for {sequence} in {IMAGES_DIR}")

        sequence_dir = split_dir / sequence
        write_images(mapping, sequence_dir, hardlink)

        tracks = parse_tracks(annotations_dir / f"{sequence}.xml")
        total_boxes += write_gt(tracks, mapping, sequence_dir)

        if split != "train":
            write_detections(sequence, sequence_dir, split)

        write_seqinfo(sequence, sequence_dir, len(mapping), frame_rate)
        total_frames += len(mapping)

    seqmap_path = write_seqmap(split, sequences)

    print(f"{len(sequences)} videos, {total_frames} frames, {total_boxes} boxes")
    print(f"Sample rate {rate}, frame rate {frame_rate}")
    print(f"Dataset: {split_dir}")
    print(f"Seqmap: {seqmap_path}")


if __name__ == "__main__":
    
    #future todo: refactor with json?

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=["train", "val", "test"])
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                        help="Applied to the train split only")
    parser.add_argument("--hardlink", action="store_true",
                        help="Hardlink instead of copying, same drive only")
    args = parser.parse_args()

    convert(args.split, args.val_ratio, args.sample_rate, args.hardlink)