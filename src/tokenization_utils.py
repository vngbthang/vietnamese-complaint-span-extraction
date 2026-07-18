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
    if getattr(tokenizer, "is_fast", False):
        try:
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
        except Exception:
            pass

    tokens = text.split()
    return align_labels_pre_tokenized(
        tokens, bio_tags, tokenizer, label2id, max_length, ignore_label_id
    )


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
    result = encode_tokens_with_labels(tokenizer, tokens, bio_tags, label2id, max_length)
    return result["input_ids"], result["labels"]


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
    if len(tokens) != len(labels):
        raise ValueError(f"tokens and labels length mismatch: {len(tokens)} vs {len(labels)}")

    # Fast tokenizer path only if tokenizer.is_fast is True
    if getattr(tokenizer, "is_fast", False):
        try:
            enc = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_attention_mask=True,
            )

            word_ids = enc.word_ids()

            aligned_labels = []
            previous_word_id = None

            for word_id in word_ids:
                if word_id is None:
                    aligned_labels.append(-100)
                elif word_id != previous_word_id:
                    aligned_labels.append(label2id[labels[word_id]])
                else:
                    aligned_labels.append(-100)
                previous_word_id = word_id

            enc["labels"] = aligned_labels

            assert len(enc["input_ids"]) == max_length
            assert len(enc["attention_mask"]) == max_length
            assert len(enc["labels"]) == max_length

            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": enc["labels"],
            }

        except Exception as e:
            print(f"Fast tokenizer alignment failed, falling back to manual alignment: {repr(e)}")

    # Slow tokenizer fallback — used for PhoBERT and any non-fast tokenizer
    raw_input_ids: List[int] = []
    raw_label_ids: List[int] = []

    for token, label in zip(tokens, labels):
        if token is None:
            continue
        token = str(token)

        sub_ids = tokenizer.encode(token, add_special_tokens=False)

        if len(sub_ids) == 0:
            continue

        raw_input_ids.extend(sub_ids)
        raw_label_ids.append(label2id[label])
        raw_label_ids.extend([-100] * (len(sub_ids) - 1))

    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    available_len = max_length - special_count

    raw_input_ids = raw_input_ids[:available_len]
    raw_label_ids = raw_label_ids[:available_len]

    enc = tokenizer.prepare_for_model(
        raw_input_ids,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
    )

    final_labels: List[int] = []
    raw_label_pointer = 0

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    special_tokens_mask = enc["special_tokens_mask"]

    for i in range(len(input_ids)):
        if attention_mask[i] == 0:
            final_labels.append(-100)
        elif special_tokens_mask[i] == 1:
            final_labels.append(-100)
        else:
            if raw_label_pointer < len(raw_label_ids):
                final_labels.append(raw_label_ids[raw_label_pointer])
                raw_label_pointer += 1
            else:
                final_labels.append(-100)

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length
    assert len(final_labels) == max_length

    return {
        "input_ids": input_ids,
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
