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


def encode_tokens_with_labels(
    tokenizer,
    tokens: List[str],
    labels: List[str],
    label2id: Dict[str, int],
    max_length: int = 256,
) -> Dict[str, List[int]]:
    """
    Align token-level BIO labels to subword tokens. Works for both fast
    and slow tokenizers.

    For every original token:
      - tokenize it into subword pieces
      - assign the BIO label to the first subword
      - assign -100 to all subsequent subwords
      - assign -100 to special tokens (CLS, SEP, PAD)

    Args:
        tokenizer: HuggingFace tokenizer (fast or slow).
        tokens:    List of whitespace-level tokens.
        labels:    List of BIO tags, one per token.
        label2id:  Mapping from tag string to integer ID.
        max_length: Maximum sequence length (including special tokens).

    Returns:
        Dict with keys: input_ids, attention_mask, labels.
        All lists have length == max_length.
    """
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    available_len = max_length - special_count

    # ---- Try fast tokenizer path ----
    batch_enc = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
        return_tensors=None,
    )

    if hasattr(batch_enc, "word_ids"):
        # Fast tokenizer: use word_ids() for alignment
        word_ids_list = batch_enc.word_ids()
        input_ids = batch_enc["input_ids"]
        attention_mask = batch_enc["attention_mask"]

        aligned_labels = []
        for wid in word_ids_list:
            if wid is None:
                aligned_labels.append(-100)
            elif wid < len(labels):
                tag = labels[wid]
                aligned_labels.append(label2id.get(tag, label2id.get("O", 0)))
            else:
                aligned_labels.append(label2id.get("O", 0))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": aligned_labels,
        }

    # ---- Slow tokenizer fallback: manual per-token encoding ----
    print("  [TOKENIZER] Using slow-tokenizer manual alignment fallback.")

    raw_input_ids: List[int] = []
    raw_label_ids: List[int] = []

    for token, label in zip(tokens, labels):
        sub_ids = tokenizer.encode(
            token,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        if not sub_ids:
            continue
        raw_input_ids.extend(sub_ids)
        raw_label_ids.append(label2id.get(label, label2id.get("O", 0)))
        raw_label_ids.extend([-100] * (len(sub_ids) - 1))

    # Truncate to available space (leave room for special tokens)
    raw_input_ids = raw_input_ids[:available_len]
    raw_label_ids = raw_label_ids[:available_len]

    # Add special tokens and pad to max_length
    encoded = tokenizer.prepare_for_model(
        raw_input_ids,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
    )

    # Reconstruct labels respecting special-token positions
    special_mask = encoded.get("special_tokens_mask", None)
    if special_mask is None:
        # Fallback: manually compute special token positions
        # CLS is at position 0, SEP is at last non-padding position
        input_ids = encoded["input_ids"]
        pad_token_id = tokenizer.pad_token_id
        sep_token_id = tokenizer.sep_token_id
        special_mask = [0] * len(input_ids)

    final_labels: List[int] = []
    raw_idx = 0
    for mask_val in special_mask:
        if mask_val == 1:
            final_labels.append(-100)
        elif raw_idx < len(raw_label_ids):
            final_labels.append(raw_label_ids[raw_idx])
            raw_idx += 1
        else:
            final_labels.append(-100)

    # Ensure padding tokens have -100
    attention_mask = encoded["attention_mask"]
    for i in range(len(final_labels)):
        if attention_mask[i] == 0:
            final_labels[i] = -100

    assert len(encoded["input_ids"]) == max_length, (
        f"input_ids length {len(encoded['input_ids'])} != max_length {max_length}"
    )
    assert len(attention_mask) == max_length, (
        f"attention_mask length {len(attention_mask)} != max_length {max_length}"
    )
    assert len(final_labels) == max_length, (
        f"labels length {len(final_labels)} != max_length {max_length}"
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": attention_mask,
        "labels": final_labels,
    }


def encode_tokens_with_labels_sanity_check(
    tokenizer,
    records,
    label2id: Dict[str, int],
    max_length: int = 256,
    n_samples: int = 3,
) -> None:
    """
    Print alignment sanity stats for the first n_samples records.
    Shows original token count, non-ignored label count, and B/I label count.
    """
    import sys
    for i, rec in enumerate(records[:n_samples]):
        tokens = rec.tokens if hasattr(rec, "tokens") else rec.get("tokens", [])
        labels = rec.bio_tags if hasattr(rec, "bio_tags") else rec.get("bio_tags", [])

        result = encode_tokens_with_labels(tokenizer, tokens, labels, label2id, max_length)

        non_ignored = sum(1 for l in result["labels"] if l != -100)
        b_count = sum(1 for l in result["labels"] if l == label2id.get("B-COMP", label2id.get("B-ASP", -100)))
        i_count = sum(1 for l in result["labels"] if l == label2id.get("I-COMP", label2id.get("I-ASP", -100)))

        print(
            f"  [Sanity #{i+1}] orig_tokens={len(tokens)}, "
            f"non_ignored_labels={non_ignored}, "
            f"B_labels={b_count}, I_labels={i_count}",
            file=sys.stderr,
        )


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
    result = encode_tokens_with_labels(tokenizer, tokens, bio_tags, label2id, max_length)
    return {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
        "labels": result["labels"],
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
