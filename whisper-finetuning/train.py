# original: https://huggingface.co/blog/fine-tune-whisper

"""Finetune Whisper on custom dataset"""
import argparse
import torch
import evaluate

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, DatasetDict, Audio, concatenate_datasets
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from audiomentations import (
    AddBackgroundNoise,
    AddGaussianNoise,
    Compose,
    Gain,
    OneOf,
    PitchShift,
    PolarityInversion,
    TimeStretch,
)
from indicnlp import common
from indicnlp import loader
from indicnlp.normalize.indic_normalize import DevanagariNormalizer
from indicnlp.transliterate.unicode_transliterate import ItransTransliterator
lang='hi'


common.set_resources_path('indic_nlp_resources')
loader.load()

normalizer = DevanagariNormalizer("hi")


augmentation = Compose(
    [
        Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.1),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=0.25),
    ]
)


AUDIO_COLUMN_NAME = "audio"
TEXT_COLUMN_NAME = "sentence"

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def train(dataset, opt, language="Hindi", dataset_name="shrutilipi_indic_mcv_indic_norm"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_size = opt.model_size
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        f"openai/whisper-{model_size}")

    tokenizer = WhisperTokenizer.from_pretrained(
        f"openai/whisper-{model_size}", language=language, task="transcribe")
    processor = WhisperProcessor.from_pretrained(
        f"openai/whisper-{model_size}", language=language, task="transcribe")

    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

        # encode target text to label ids
        input_str = normalizer.normalize(batch[TEXT_COLUMN_NAME]).strip()
        batch["labels"] = processor.tokenizer(input_str).input_ids
        return batch
    
    dataset = dataset.map(
        prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=8)
    
    def is_labels_in_length_range(labels):
        return len(labels) < 448
    
    dataset = dataset.filter(
        is_labels_in_length_range, num_proc=4, input_columns=["labels"]
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")
    # model = model.to(device)
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_size}")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    do_normalize_eval = True
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        if do_normalize_eval:
            pred_str = [normalizer.normalize(pred) for pred in pred_str]
            # # perhaps already normalised
            label_str = [normalizer.normalize(label) for label in label_str]
            # filtering step to only evaluate the samples that correspond to non-zero references
            pred_str = [pred_str[i].strip() for i in range(len(pred_str)) if len(label_str[i]) > 0]
            label_str = [label_str[i].strip() for i in range(len(label_str)) if len(label_str[i]) > 0]

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    num_steps = opt.epochs * len(dataset["train"])
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./whisper-{model_size}-{language}-{dataset_name}",  # change to a repo name of your choice
        per_device_train_batch_size=opt.batch_size,
        gradient_accumulation_steps=opt.grad_acc,  # increase by 2x for every 2x decrease in batch size
        learning_rate=4.25e-5,
        weight_decay=0.01,
        warmup_steps=1200,
        max_steps=int(num_steps/(opt.batch_size * opt.grad_acc)),
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=2000,
        eval_steps=2000,
        logging_steps=200,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
        dataloader_num_workers=opt.num_workers)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train(resume_from_checkpoint = False)
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


def normalize_dataset(ds, audio_column_name=None, text_column_name=None):
    if audio_column_name is not None and audio_column_name != AUDIO_COLUMN_NAME:
        ds = ds.rename_column(audio_column_name, AUDIO_COLUMN_NAME)
    if text_column_name is not None and text_column_name != TEXT_COLUMN_NAME:
        ds = ds.rename_column(text_column_name, TEXT_COLUMN_NAME)

    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds = ds.remove_columns(set(ds.features.keys()) - set([AUDIO_COLUMN_NAME, TEXT_COLUMN_NAME]))
    return ds


def augment_dataset(batch):
    # load and (possibly) resample audio data to 16kHz
    sample = batch[AUDIO_COLUMN_NAME]

    # apply augmentation
    print(batch)
    augmented_waveform = augmentation(sample["array"], sample_rate=sample["sampling_rate"])
    batch[AUDIO_COLUMN_NAME]["array"] = augmented_waveform
    return batch


def load_datasets(opt):
    ds = DatasetDict()
    ds_shrutilipi_train = load_dataset("collabora/ai4bharat-shrutilipi", split="train+validation")
    ds_shrutilipi_train = normalize_dataset(ds_shrutilipi_train)
    
    ds_mcv_hi_train = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train", use_auth_token=True)
    ds_mcv_hi_train = normalize_dataset(ds_mcv_hi_train)
    ds_mcv_hi_valid = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="validation", use_auth_token=True)
    ds_mcv_hi_valid = normalize_dataset(ds_mcv_hi_valid)

    ds_indic_superb = load_dataset("makaveli10/indic-superb-whisper", split="train+validation")
    ds_indic_superb = normalize_dataset(ds_indic_superb, text_column_name="transcription")

    # augmented shrutilipi dataset
    # ds_augmented_shrutilipi = load_dataset("collabora/ai4bharat-shrutilipi-augmented", split="train", use_auth_token=True)


    ds["train"] = concatenate_datasets([ds_shrutilipi_train, ds_mcv_hi_train, ds_indic_superb])
    # ds["train"] = concatenate_datasets([ds_mcv_hi_train])
    ds["test"] = concatenate_datasets([ds_mcv_hi_valid])

    if opt.augment:
        print("applying augmentations..")
        augmented_dataset = ds_shrutilipi_train_45_pct.map(
            augment_dataset, num_proc=1, desc="augment train dataset"
        )
        ds["train"] = concatenate_datasets([ds["train"], augmented_indic, augmented_dataset])
   
    ds["train"] = ds["train"].shuffle(seed=10)
    return ds

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-size', default="medium", type=str, help='whisper model size')
    parser.add_argument('--batch-size', default=8, type=int, help='batch-size per device')
    parser.add_argument('--grad-acc', default=4, type=int, help='gradient accumulation steps')
    parser.add_argument('--augment', action="store_true", help='apply augmentations')
    parser.add_argument('--num_workers', default=12, type=int, help='num dataloader workers')
    parser.add_argument('--epochs', default=3, type=int, help='num epochs')
    opt = parser.parse_args()
    ds = load_datasets(opt)
    train(ds, opt)
