"""Trainer utilities and callbacks for token classification."""

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tokenizer(model_name: str):
    """Load tokenizer for a given model."""
    return AutoTokenizer.from_pretrained(model_name)


def prepare_dataloaders(
    train_records: List[Any],
    valid_records: List[Any],
    tokenizer,
    label2id: Dict[str, int],
    max_length: int = 256,
    train_batch_size: int = 16,
    eval_batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader]:
    """Prepare PyTorch DataLoaders for training and evaluation."""
    from src.data_utils import TokenClassificationDataset

    train_dataset = TokenClassificationDataset(
        train_records, tokenizer, max_length=max_length, label2id=label2id
    )
    valid_dataset = TokenClassificationDataset(
        valid_records, tokenizer, max_length=max_length, label2id=label2id
    )

    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    return train_loader, valid_loader


def prepare_hf_dataset(
    records: List[Any],
    tokenizer,
    label2id: Dict[str, int],
    max_length: int = 256,
):
    """Convert records to HuggingFace Dataset format."""
    from datasets import Dataset

    texts = [' '.join(r.tokens) for r in records]
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        is_split_into_words=True,
        return_tensors=None,
    )

    def align_labels(encoding, record):
        word_ids = encoding.word_ids()
        labels = []
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id < len(record.bio_tags):
                tag = record.bio_tags[word_id]
                labels.append(label2id.get(tag, 0))
            else:
                labels.append(label2id.get('O', 0))
        return {'labels': labels}

    encodings['labels'] = [
        align_labels({'word_ids': encodings['word_ids'][i]}, records[i])['labels']
        for i in range(len(records))
    ]

    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': [a['labels'] for a in [
            align_labels({'word_ids': encodings['word_ids'][i]}, records[i])
            for i in range(len(records))
        ]],
    })
    return dataset


def build_training_args(
    output_dir: str,
    learning_rate: float = 2e-5,
    num_epochs: int = 5,
    train_batch_size: int = 16,
    eval_batch_size: int = 16,
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
    metric_for_best_model: str = 'entity_f1',
    fp16: bool = True,
    save_total_limit: int = 1,
    logging_steps: int = 50,
    eval_steps: int = 200,
    save_steps: int = 200,
    seed: int = 42,
    load_best_model_at_end: bool = True,
    early_stopping_patience: int = 3,
) -> TrainingArguments:
    """Build HuggingFace TrainingArguments."""
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        fp16=fp16,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=logging_steps,
        eval_strategy='steps',
        eval_steps=eval_steps,
        save_strategy='steps',
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        seed=seed,
        report_to='none',
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )


def log_gpu_info():
    """Print GPU name and memory."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'GPU: {gpu_name} ({gpu_mem:.1f} GB)')
    else:
        print('GPU: None (CPU only)')


def save_metrics(metrics: Dict[str, Any], output_path: str):
    """Save metrics to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(metrics_path: str) -> Dict[str, Any]:
    """Load metrics from JSON."""
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path) as f:
        return json.load(f)


def check_completed(model_dir: str, force_rerun: bool = False) -> bool:
    """Check if an experiment has already been completed."""
    if force_rerun:
        return False
    return os.path.exists(os.path.join(model_dir, 'completed_result.json'))
