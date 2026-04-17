# Fine-Tuning Custom Whisper Models for WhisperLive

This guide explains how to fine-tune a Whisper model on your own domain-specific data and deploy it with WhisperLive. This is a **major open-source differentiator** — commercial APIs don't let you customize the underlying model.

## Why Fine-Tune?

- **Domain vocabulary**: Medical, legal, technical terminology
- **Accent adaptation**: Regional accents, non-native speakers
- **Noise robustness**: Train on noisy data matching your environment
- **Language support**: Improve quality for low-resource languages

## Prerequisites

```bash
pip install transformers datasets evaluate jiwer tensorboard
pip install accelerate  # for multi-GPU training
```

## Step 1: Prepare Your Dataset

Your dataset needs audio files paired with transcriptions. Supported formats:

### Option A: HuggingFace Dataset Format

```python
from datasets import Dataset, Audio

# From local files
data = {
    "audio": ["path/to/audio1.wav", "path/to/audio2.wav"],
    "transcription": ["expected text one", "expected text two"],
}
dataset = Dataset.from_dict(data).cast_column("audio", Audio(sampling_rate=16000))
dataset.save_to_disk("my_dataset")
```

### Option B: CSV Manifest

```csv
audio_path,transcription
/data/audio/001.wav,the patient shows symptoms of hypertension
/data/audio/002.wav,administer fifty milligrams of metoprolol
```

Then use the helper script:

```bash
python scripts/prepare_finetune_data.py \
  --manifest data/manifest.csv \
  --output data/processed \
  --sample_rate 16000
```

## Step 2: Fine-Tune with the Helper Script

```bash
python scripts/finetune_whisper.py \
  --base_model openai/whisper-small \
  --dataset_path my_dataset \
  --output_dir models/my-custom-whisper \
  --epochs 3 \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --language en \
  --warmup_steps 500
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--base_model` | `openai/whisper-small` | Base model to fine-tune from |
| `--epochs` | 3 | Number of training epochs |
| `--batch_size` | 8 | Training batch size |
| `--learning_rate` | 1e-5 | Learning rate |
| `--warmup_steps` | 500 | Warmup steps for scheduler |
| `--fp16` | False | Use mixed precision training |
| `--gradient_accumulation` | 1 | Gradient accumulation steps |

## Step 3: Convert to CTranslate2 (faster-whisper format)

```bash
ct2-whisper-converter \
  --model models/my-custom-whisper \
  --output_dir models/my-custom-whisper-ct2 \
  --quantization int8
```

Or use the helper:

```bash
python scripts/finetune_whisper.py \
  --convert_only \
  --model_path models/my-custom-whisper \
  --output_dir models/my-custom-whisper-ct2 \
  --quantization int8
```

## Step 4: Deploy with WhisperLive

```bash
python run_server.py \
  --backend faster_whisper \
  --faster_whisper_custom_model_path models/my-custom-whisper-ct2 \
  --enable_rest
```

Or via Docker:

```bash
docker run -v $(pwd)/models:/models \
  -p 9090:9090 -p 8000:8000 \
  whisperlive-cpu \
  --faster_whisper_custom_model_path /models/my-custom-whisper-ct2 \
  --enable_rest
```

## Step 5: Evaluate

```bash
python scripts/finetune_whisper.py \
  --evaluate_only \
  --model_path models/my-custom-whisper-ct2 \
  --test_dataset test_data \
  --language en
```

This outputs Word Error Rate (WER) and Character Error Rate (CER).

## Tips

- **Start small**: Fine-tune `whisper-small` first, scale up if needed
- **Data quality > quantity**: Clean, accurately transcribed data matters more
- **Domain-specific data**: Even 1-2 hours of domain audio helps significantly
- **Validation set**: Always hold out 10-20% for validation
- **Early stopping**: Monitor validation WER to prevent overfitting
- **int8 quantization**: Reduces model size 4x with minimal quality loss

## Example: Medical Transcription

```bash
# Prepare medical audio dataset
python scripts/prepare_finetune_data.py \
  --manifest medical_data/manifest.csv \
  --output medical_data/processed

# Fine-tune
python scripts/finetune_whisper.py \
  --base_model openai/whisper-small \
  --dataset_path medical_data/processed \
  --output_dir models/whisper-medical \
  --epochs 5 \
  --learning_rate 5e-6

# Convert and deploy
python scripts/finetune_whisper.py \
  --convert_only \
  --model_path models/whisper-medical \
  --output_dir models/whisper-medical-ct2

python run_server.py \
  --faster_whisper_custom_model_path models/whisper-medical-ct2 \
  --enable_rest
```
