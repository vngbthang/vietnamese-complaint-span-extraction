"""Model definitions for token classification."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def get_token_classifier(
    model_name: str,
    num_labels: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    loss_type: str = 'ce',
) -> nn.Module:
    """
    Build a token classification model from a pretrained checkpoint.

    Args:
        model_name: HuggingFace model name or path.
        num_labels: Number of classification labels.
        label2id: Tag -> integer mapping.
        id2label: Integer -> tag mapping.
        loss_type: 'ce' for standard cross-entropy or 'weighted_ce' for weighted.

    Returns:
        A HuggingFace AutoModelForTokenClassification.
    """
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
    )
    return model


class WeightedTokenClassifier(nn.Module):
    """Token classifier with weighted cross-entropy loss."""

    def __init__(self, encoder, config, num_labels: int, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_labels = num_labels

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        **kwargs,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights,
                reduction='mean',
                ignore_index=-100,
            )
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {'loss': loss, 'logits': logits}


def load_encoder_for_transfer(
    pretrained_path: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load pretrained encoder for transfer learning.
    Returns the base encoder (without classification head).
    """
    model = AutoModel.from_pretrained(pretrained_path)
    tokenizer = AutoModelForTokenClassification.from_pretrained(
        pretrained_path
    ).resize_token_embeddings(1).to_empty()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    return model, tokenizer


def create_complaint_model_from_encoder(
    encoder: PreTrainedModel,
    config: AutoConfig,
    num_labels: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> AutoModelForTokenClassification:
    """
    Create a new complaint span model using a pretrained encoder.
    Copies only the encoder weights, initializes a new classification head.
    """
    model = AutoModelForTokenClassification.from_config(config)
    model.roformer = encoder if hasattr(encoder, 'roformer') else encoder

    # Copy encoder weights
    encoder_state = encoder.state_dict()
    model_dict = model.state_dict()

    # Filter encoder layers
    prefix_map = {
        'phobert': 'roberta',
        'xlm-roberta': 'xlm_roberta',
    }

    transferred = 0
    for key, value in encoder_state.items():
        model_key = key
        for src_prefix, tgt_prefix in prefix_map.items():
            if src_prefix in model_dict and tgt_prefix not in model_dict:
                model_key = key.replace(src_prefix, tgt_prefix)
                break
        if model_key in model_dict:
            if model_dict[model_key].shape == value.shape:
                model_dict[model_key] = value
                transferred += 1

    model.load_state_dict(model_dict, strict=False)
    return model
