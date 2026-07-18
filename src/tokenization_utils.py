"""Tokenization and subword alignment utilities."""

from typing import Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizer


def align_labels_to_subwords(
    bio_tags: List[str],
    tokenizer: PreTrainedTokenizer,
    text: str,
    label2id: Dict[str, int],
    max_length: int = 256,
    ignore_label_id: int = -100,
) -> Tuple[List[int], List[int]]:
    """
    Align token-level BIO labels to subword tokens.

    Args:
        bio_tags: List of BIO tags, one per whitespace token.
        tokenizer: HuggingFace tokenizer.
        text: The full text string.
        label2id: Mapping from BIO tag string to integer ID.
        max_length: Maximum sequence length.
        ignore_label_id: Label ID for special tokens and continuation subwords.

    Returns:
        Tuple of (input_ids, aligned_labels)
    """
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].squeeze(0).tolist()
    word_ids = encoding.word_ids()

    aligned_labels = []
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(ignore_label_id)
        elif word_id < len(bio_tags):
            tag = bio_tags[word_id]
            aligned_labels.append(label2id.get(tag, label2id.get('O', 0)))
        else:
            aligned_labels.append(label2id.get('O', 0))

    return input_ids, aligned_labels


def align_labels_pre_tokenized(
    tokens: List[str],
    bio_tags: List[str],
    tokenizer: PreTrainedTokenizer,
    label2id: Dict[str, int],
    max_length: int = 256,
    ignore_label_id: int = -100,
) -> Tuple[List[int], List[int]]:
    """
    Align token-level BIO labels to subword tokens when input is already tokenized.

    Args:
        tokens: List of whitespace tokens.
        bio_tags: List of BIO tags, one per token.
        tokenizer: HuggingFace tokenizer.
        label2id: Mapping from BIO tag string to integer ID.
        max_length: Maximum sequence length.
        ignore_label_id: Label ID for special tokens and continuation subwords.

    Returns:
        Tuple of (input_ids, aligned_labels)
    """
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].squeeze(0).tolist()
    word_ids = encoding.word_ids()

    aligned_labels = []
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(ignore_label_id)
        elif word_id < len(bio_tags):
            tag = bio_tags[word_id]
            aligned_labels.append(label2id.get(tag, label2id.get('O', 0)))
        else:
            aligned_labels.append(label2id.get('O', 0))

    return input_ids, aligned_labels


def convert_to_hf_format(
    tokens: List[str],
    bio_tags: List[str],
    tokenizer: PreTrainedTokenizer,
    label2id: Dict[str, int],
    max_length: int = 256,
) -> Dict[str, any]:
    """
    Convert a pre-tokenized record into HuggingFace format.

    Returns a dict with input_ids, attention_mask, labels (aligned).
    """
    input_ids, labels = align_labels_pre_tokenized(
        tokens, bio_tags, tokenizer, label2id, max_length
    )
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    return {
        'input_ids': encoding['input_ids'].squeeze(0),
        'attention_mask': encoding['attention_mask'].squeeze(0),
        'labels': torch.tensor(labels, dtype=torch.long),
    }


def decode_predictions(
    predictions: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    id2label: Dict[int, str],
) -> List[str]:
    """Convert model predictions back to tag strings."""
    return [id2label.get(p.item(), 'O') for p in predictions]


def get_token_level_predictions(
    subword_preds: List[int],
    word_ids: List[Optional[int]],
    id2label: Dict[int, str],
    ignore_label_id: int = -100,
) -> List[str]:
    """
    Convert subword-level predictions to token-level predictions.
    Uses first subword prediction for each word.
    """
    word_preds: Dict[int, int] = {}
    for subword_idx, word_id in enumerate(word_ids):
        if word_id is not None and subword_preds[subword_idx] != ignore_label_id:
            if word_id not in word_preds:
                word_preds[word_id] = subword_preds[subword_idx]

    token_preds = [
        word_preds.get(i, id2label.get(0, 'O'))
        for i in range(max(word_preds.keys(), default=-1) + 1)
    ]
    return [id2label.get(p, 'O') for p in token_preds]
