#!/usr/bin/env python3
"""
Whisper fine-tuning helper script.

Supports:
- Fine-tuning a Whisper model on custom datasets
- Converting fine-tuned models to CTranslate2 format
- Evaluating model performance (WER/CER)

Usage:
    # Fine-tune
    python scripts/finetune_whisper.py \
        --base_model openai/whisper-small \
        --dataset_path my_dataset \
        --output_dir models/my-custom-whisper \
        --epochs 3

    # Convert to CTranslate2
    python scripts/finetune_whisper.py \
        --convert_only \
        --model_path models/my-custom-whisper \
        --output_dir models/my-custom-whisper-ct2

    # Evaluate
    python scripts/finetune_whisper.py \
        --evaluate_only \
        --model_path models/my-custom-whisper-ct2 \
        --test_dataset test_data
"""

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def finetune(args):
    """Fine-tune a Whisper model."""
    try:
        from transformers import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
        from datasets import load_from_disk, Audio
        import evaluate
        import torch
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install transformers datasets evaluate jiwer")
        sys.exit(1)

    logger.info(f"Loading base model: {args.base_model}")
    processor = WhisperProcessor.from_pretrained(args.base_model)
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)

    if args.language:
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=args.language, task="transcribe"
        )

    logger.info(f"Loading dataset: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        return batch

    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        fp16=args.fp16 and torch.cuda.is_available(),
        evaluation_strategy="epoch" if "validation" in dataset else "no",
        save_strategy="epoch",
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        load_best_model_at_end=True if "validation" in dataset else False,
    )

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")


def convert(args):
    """Convert a fine-tuned model to CTranslate2 format."""
    try:
        import ctranslate2
    except ImportError:
        logger.error("ctranslate2 not installed. Install with: pip install ctranslate2")
        sys.exit(1)

    model_path = args.model_path
    output_dir = args.output_dir
    quantization = args.quantization

    logger.info(f"Converting {model_path} to CTranslate2 ({quantization})")

    converter = ctranslate2.converters.TransformersConverter(
        model_path,
        copy_files=["tokenizer.json", "preprocessor_config.json"],
    )
    converter.convert(
        output_dir=output_dir,
        quantization=quantization,
        force=False,
    )
    logger.info(f"CTranslate2 model saved to {output_dir}")


def evaluate_model(args):
    """Evaluate a model's WER on a test dataset."""
    try:
        from faster_whisper import WhisperModel
        from datasets import load_from_disk
        import jiwer
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)

    logger.info(f"Loading model: {args.model_path}")
    model = WhisperModel(args.model_path, device="cpu", compute_type="int8")

    logger.info(f"Loading test dataset: {args.test_dataset}")
    dataset = load_from_disk(args.test_dataset)

    predictions = []
    references = []

    for sample in dataset:
        audio = sample["audio"]["array"]
        segments, _ = model.transcribe(audio, language=args.language)
        pred_text = " ".join(s.text.strip() for s in segments)
        predictions.append(pred_text)
        references.append(sample["transcription"])

    wer = jiwer.wer(references, predictions)
    cer = jiwer.cer(references, predictions)

    logger.info(f"Word Error Rate (WER): {wer:.4f}")
    logger.info(f"Character Error Rate (CER): {cer:.4f}")
    logger.info(f"Evaluated {len(predictions)} samples")


def main():
    parser = argparse.ArgumentParser(description="WhisperLive fine-tuning helper")

    # Mode selection
    parser.add_argument("--convert_only", action="store_true",
                        help="Only convert model to CTranslate2")
    parser.add_argument("--evaluate_only", action="store_true",
                        help="Only evaluate model WER")

    # Fine-tuning args
    parser.add_argument("--base_model", type=str, default="openai/whisper-small")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="models/custom-whisper")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--language", type=str, default=None)

    # Conversion args
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--quantization", type=str, default="int8",
                        choices=["int8", "float16", "float32"])

    # Evaluation args
    parser.add_argument("--test_dataset", type=str, default=None)

    args = parser.parse_args()

    if args.convert_only:
        if not args.model_path:
            parser.error("--model_path required for --convert_only")
        convert(args)
    elif args.evaluate_only:
        if not args.model_path or not args.test_dataset:
            parser.error("--model_path and --test_dataset required for --evaluate_only")
        evaluate_model(args)
    else:
        if not args.dataset_path:
            parser.error("--dataset_path required for fine-tuning")
        finetune(args)


if __name__ == "__main__":
    main()
