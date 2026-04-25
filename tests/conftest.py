import os
import pytest
import torch

from hw4lib.data import H4Tokenizer, LMDataset, ASRDataset
from hw4lib.decoding import SequenceGenerator
from hw4lib.model.decoder_layers import CrossAttentionDecoderLayer, SelfAttentionDecoderLayer
from hw4lib.model.encoder_layers import SelfAttentionEncoderLayer
from hw4lib.model.masks import CausalMask, PadMask
from hw4lib.model.positional_encoding import PositionalEncoding
from hw4lib.model.sublayers import CrossAttentionLayer, FeedForwardLayer, SelfAttentionLayer
from hw4lib.model.transformers import DecoderOnlyTransformer, EncoderDecoderTransformer


@pytest.fixture
def self_attn():
    return SelfAttentionLayer


@pytest.fixture
def cross_attention():
    return CrossAttentionLayer


@pytest.fixture
def encoder_layer():
    return SelfAttentionEncoderLayer


@pytest.fixture
def decoder_layer(request):
    name = request.node.path.name
    if "decoderlayer_selfattention" in name:
        return SelfAttentionDecoderLayer
    return CrossAttentionDecoderLayer


@pytest.fixture
def feedforward():
    return FeedForwardLayer


@pytest.fixture
def transformer(request):
    name = request.node.path.name
    if "decoder_only" in name:
        return DecoderOnlyTransformer
    return EncoderDecoderTransformer


@pytest.fixture
def mask_gen_fn(request):
    name = request.node.path.name
    if "causal" in name:
        return CausalMask
    return PadMask


@pytest.fixture
def positional_encoding_fn():
    return PositionalEncoding


@pytest.fixture
def generator():
    return SequenceGenerator


_TOKEN_CFG = {
    "token_type": "1k",
    "token_map": {
        "char": "./hw4lib/data/tokenizer_jsons/tokenizer_char.json",
        "1k": "./hw4lib/data/tokenizer_jsons/tokenizer_1000.json",
        "5k": "./hw4lib/data/tokenizer_jsons/tokenizer_5000.json",
        "10k": "./hw4lib/data/tokenizer_jsons/tokenizer_10000.json",
    },
}


@pytest.fixture
def tokenizer():
    return H4Tokenizer(
        token_map=_TOKEN_CFG["token_map"],
        token_type=_TOKEN_CFG["token_type"],
        validate=False,
    )


@pytest.fixture
def dataset(request):
    name = request.node.path.name
    tokenizer = H4Tokenizer(
        token_map=_TOKEN_CFG["token_map"],
        token_type=_TOKEN_CFG["token_type"],
        validate=False,
    )
    if "asr" in name.lower():
        data_cfg = {
            "root": "./hw4_data_subset/hw4p2_data",
            "subset": 1.0,
            "batch_size": 8,
            "NUM_WORKERS": 2,
            "num_feats": 80,
            "norm": "global_mvn",
            "specaug": True,
            "specaug_conf": {
                "apply_time_mask": True,
                "apply_freq_mask": True,
                "num_freq_mask": 2,
                "num_time_mask": 2,
                "freq_mask_width_range": 10,
                "time_mask_width_range": 10,
            },
        }
        train_ds = ASRDataset(
            partition="train-clean-100",
            config=data_cfg,
            tokenizer=tokenizer,
            isTrainPartition=True,
            global_stats=None,
        )
        stats = None
        if data_cfg["norm"] == "global_mvn":
            stats = (train_ds.global_mean, train_ds.global_std)
        return ASRDataset(
            partition="test-clean",
            config=data_cfg,
            tokenizer=tokenizer,
            isTrainPartition=False,
            global_stats=stats,
        )

    data_cfg = {
        "root": "./hw4_data_subset/hw4p1_data",
        "train_partition": "train",
        "val_partition": "valid",
        "test_partition": "test",
        "subset": 1.0,
        "batch_size": 8,
        "NUM_WORKERS": 2,
    }
    return LMDataset(
        partition=data_cfg["train_partition"],
        config=data_cfg,
        tokenizer=tokenizer,
    )
