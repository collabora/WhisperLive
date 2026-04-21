#!/usr/bin/env python3
"""
Prepare audio data for Whisper fine-tuning.

Reads a CSV manifest (audio_path, transcription) and creates a
HuggingFace Dataset with proper audio formatting.

Usage:
    python scripts/prepare_finetune_data.py \
        --manifest data/manifest.csv \
        --output data/processed \
        --sample_rate 16000
"""

import argparse
import csv
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare audio data for fine-tuning")
    parser.add_argument("--manifest", type=str, required=True,
                        help="CSV file with columns: audio_path, transcription")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for the HuggingFace dataset")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Target sample rate (default: 16000)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")
    args = parser.parse_args()

    try:
        from datasets import Dataset, Audio, DatasetDict
    except ImportError:
        logger.error("datasets library required. Install: pip install datasets")
        sys.exit(1)

    if not os.path.exists(args.manifest):
        logger.error(f"Manifest file not found: {args.manifest}")
        sys.exit(1)

    audio_paths = []
    transcriptions = []

    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = row.get("audio_path", "").strip()
            transcription = row.get("transcription", "").strip()
            if audio_path and transcription:
                if not os.path.isabs(audio_path):
                    audio_path = os.path.join(os.path.dirname(args.manifest), audio_path)
                if os.path.exists(audio_path):
                    audio_paths.append(audio_path)
                    transcriptions.append(transcription)
                else:
                    logger.warning(f"Audio file not found, skipping: {audio_path}")

    logger.info(f"Found {len(audio_paths)} valid audio-transcription pairs")

    if not audio_paths:
        logger.error("No valid data found in manifest")
        sys.exit(1)

    dataset = Dataset.from_dict({
        "audio": audio_paths,
        "transcription": transcriptions,
    }).cast_column("audio", Audio(sampling_rate=args.sample_rate))

    if args.val_split > 0:
        split = dataset.train_test_split(test_size=args.val_split, seed=42)
        ds_dict = DatasetDict({
            "train": split["train"],
            "validation": split["test"],
        })
        ds_dict.save_to_disk(args.output)
        logger.info(f"Dataset saved: {len(split['train'])} train, {len(split['test'])} val")
    else:
        dataset.save_to_disk(args.output)
        logger.info(f"Dataset saved: {len(dataset)} samples")


if __name__ == "__main__":
    main()
