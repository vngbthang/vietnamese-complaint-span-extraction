"""Data loading and preprocessing utilities."""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import Dataset


@dataclass
class BioRecord:
    """Represents a single record with BIO annotations."""
    id: str
    source: str
    task: str
    split: str
    text: str
    tokens: List[str]
    token_offsets: List[Dict[str, int]]
    bio_tags: List[str]
    spans: Optional[List[Dict[str, Any]]] = None
    review_level_label: Optional[int] = None

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    @property
    def num_spans(self) -> int:
        return len(self.spans) if self.spans else 0

    def to_dict(self) -> Dict[str, Any]:
        out = {
            'id': self.id,
            'source': self.source,
            'task': self.task,
            'split': self.split,
            'text': self.text,
            'tokens': self.tokens,
            'token_offsets': self.token_offsets,
            'bio_tags': self.bio_tags,
        }
        if self.spans is not None:
            out['spans'] = self.spans
        if self.review_level_label is not None:
            out['review_level_label'] = self.review_level_label
        return out


def load_jsonl(path: str) -> List[BioRecord]:
    """Load records from a JSONL file."""
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            records.append(BioRecord(
                id=data['id'],
                source=data.get('source', ''),
                task=data.get('task', ''),
                split=data.get('split', ''),
                text=data['text'],
                tokens=data['tokens'],
                token_offsets=data['token_offsets'],
                bio_tags=data['bio_tags'],
                spans=data.get('spans'),
                review_level_label=data.get('review_level_label'),
            ))
    return records


def load_split(data_dir: str, split: str) -> List[BioRecord]:
    """Load a single split from a dataset directory."""
    path = os.path.join(data_dir, f'{split}.jsonl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    return load_jsonl(path)


def load_all_splits(data_dir: str) -> Dict[str, List[BioRecord]]:
    """Load all train/valid/test splits from a dataset directory."""
    return {
        split: load_split(data_dir, split)
        for split in ['train', 'valid', 'test']
        if os.path.exists(os.path.join(data_dir, f'{split}.jsonl'))
    }


def derive_spans_from_bio(bio_tags: List[str],
                           offsets: List[Dict[str, int]],
                           text: str) -> List[Dict[str, Any]]:
    """Derive spans from BIO tags and token offsets."""
    spans = []
    i = 0
    while i < len(bio_tags):
        if bio_tags[i] == 'B':
            prefix = bio_tags[i].replace('B-', '')
            s_start = offsets[i]['start']
            j = i
            while (j + 1 < len(bio_tags)
                   and bio_tags[j + 1] == f'I-{prefix}'):
                j += 1
            s_end = offsets[j]['end']
            spans.append({
                'start': s_start,
                'end': s_end,
                'text': text[s_start:s_end],
                'span_type': prefix.lower(),
            })
            i = j + 1
        else:
            i += 1
    return spans


def spans_to_bio(spans: List[Dict[str, Any]],
                 offsets: List[Dict[str, int]]) -> List[str]:
    """Convert spans back to BIO tags (lossless only if no overlap)."""
    n = len(offsets)
    bio = ['O'] * n
    for sp in spans:
        s_start, s_end = sp['start'], sp['end']
        overlapped = [
            i for i, off in enumerate(offsets)
            if off['start'] < s_end and s_start < off['end']
        ]
        if not overlapped:
            continue
        span_type = sp.get('span_type', 'COMP').upper()
        bio[overlapped[0]] = f'B-{span_type}'
        for idx in overlapped[1:]:
            bio[idx] = f'I-{span_type}'
    return bio


class BioDataset(Dataset):
    """PyTorch Dataset for BIO-tagged token classification."""

    def __init__(self, records: List[BioRecord], tokenizer, max_length: int = 256,
                 label2id: Optional[Dict[str, int]] = None):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id or {'O': 0, 'B': 1, 'I': 2}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        text = ' '.join(record.tokens)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            is_split_into_words=False,
            return_tensors='pt',
        )

        word_ids = encoding.word_ids()
        labels = []
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            else:
                original_tag = record.bio_tags[word_id] if word_id < len(record.bio_tags) else 'O'
                labels.append(self.label2id.get(original_tag, 0))

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long),
            'word_ids': word_ids,
        }


class TokenClassificationDataset(Dataset):
    """Flexible token classification dataset supporting pre-tokenized inputs."""

    def __init__(self, records: List[BioRecord],
                 tokenizer,
                 max_length: int = 256,
                 label2id: Optional[Dict[str, int]] = None,
                 tokenized_text: Optional[List[str]] = None):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id or {'O': 0, 'B-COMP': 1, 'I-COMP': 2}
        self.tokenized_text = tokenized_text

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]

        if self.tokenized_text is not None and idx < len(self.tokenized_text):
            tokens = self.tokenized_text[idx]
            text = ' '.join(tokens)
        else:
            text = ' '.join(record.tokens)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            is_split_into_words=True,
            return_tensors='pt',
        )

        word_ids = encoding.word_ids()
        labels = []
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            else:
                original_tag = record.bio_tags[word_id] if word_id < len(record.bio_tags) else 'O'
                labels.append(self.label2id.get(original_tag, 0))

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


def compute_class_weights(records: List[BioRecord], label2id: Dict[str, int]) -> torch.Tensor:
    """Compute inverse-frequency class weights for weighted cross-entropy."""
    counts = {k: 0 for k in label2id}
    for rec in records:
        for tag in rec.bio_tags:
            if tag in label2id:
                counts[tag] += 1

    total = sum(counts.values())
    weights = []
    for label, idx in sorted(label2id.items(), key=lambda x: x[1]):
        w = total / (len(label2id) * counts[label]) if counts[label] > 0 else 1.0
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float)


def filter_by_source(records: List[BioRecord], source: str) -> List[BioRecord]:
    """Filter records by source dataset."""
    return [r for r in records if r.source == source]


def filter_by_split(records: List[BioRecord], split: str) -> List[BioRecord]:
    """Filter records by split."""
    return [r for r in records if r.split == split]
