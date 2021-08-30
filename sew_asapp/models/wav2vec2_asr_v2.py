# Copyright (c) ASAPP Inc.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import contextlib
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from omegaconf import MISSING, II, open_dict
from typing import Any

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    overwrite_args_by_name,
)
from fairseq.tasks import FairseqTask
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer

from fairseq.models.wav2vec.wav2vec2_asr import (
    Wav2Vec2AsrConfig, Wav2Vec2CtcConfig, Wav2Vec2Seq2SeqConfig,
    Wav2VecCtc, Wav2VecEncoder, Wav2Vec2Seq2SeqModel, Linear, Embedding
)


def forced_overwrite(cfg, overrides):
    from omegaconf import open_dict

    with open_dict(cfg):
        for k, v in overrides.items():
            setattr(cfg, k, v)

@dataclass
class Wav2Vec2CtcV2Config(Wav2Vec2CtcConfig):
    output_upsample: int = field(
        default=0, metadata={"help":
            "upsample the time dimension by this scale (0 means no scale, 1 adds a new layer)"}
    )
    chunk_mode: str = field(
        default="flat", metadata={"help": "mode of chunkwise feature extractor"}
    )
    drop_upsample_layers: int = field(
        default=0, metadata={"help": "number of layers to drop in pre-trained upsample layers"}
    )


@register_model("wav2vec_ctc_v2", dataclass=Wav2Vec2CtcV2Config)
class Wav2VecCtcV2(Wav2VecCtc):
    @classmethod
    def build_model(cls, cfg: Wav2Vec2CtcV2Config, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2VecEncoderV2(cfg, task.target_dictionary)
        return cls(cfg, w2v_encoder)

@dataclass
class Wav2Vec2Seq2SeqV2Config(Wav2Vec2Seq2SeqConfig):
    output_upsample: int = field(
        default=0, metadata={"help":
            "upsample the time dimension by this scale (0 means no scale, 1 adds a new layer)"}
    )
    chunk_mode: str = field(
        default="flat", metadata={"help": "mode of chunkwise feature extractor"}
    )


@register_model("wav2vec_seq2seq_v2", dataclass=Wav2Vec2Seq2SeqV2Config)
class Wav2Vec2Seq2SeqModelV2(Wav2Vec2Seq2SeqModel):
    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqConfig, task: FairseqTask):
        """Build a new model instance."""

        assert cfg.autoregressive, "Please set task.autoregressive=true for seq2seq asr models"

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)

        encoder = cls.build_encoder(cfg)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)

        return cls(encoder, decoder)

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2AsrConfig):
        return Wav2VecEncoderV2(cfg)


@register_model("wav2vec_ctc_seq2seq", dataclass=Wav2Vec2Seq2SeqV2Config)
class Wav2VecCtcSeq2Seq(Wav2Vec2Seq2SeqModelV2):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.ctc_proj = Linear(encoder.output_dim, len(decoder.dictionary))

    def forward(self, **kwargs):
        encoder_out = self.encoder(tbc=False, **kwargs)
        ctc_out = self.forward_ctc(encoder_out)
        decoder_out = self.decoder(encoder_out=encoder_out, **kwargs)
        return {
            "encoder_out": encoder_out,
            "ctc_out": ctc_out,
            "decoder_out": decoder_out,
        }

    def forward_ctc(self, encoder_out):
        return self.ctc_proj(encoder_out["encoder_out"])
        
    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        # if sample is not None:
        return self.decoder.get_normalized_probs(net_output, log_probs, sample)

        # logits = net_output["ctc_out"]
        # if log_probs:
        #     return utils.log_softmax(logits.float(), dim=-1)
        # else:
        #     return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["ctc_out"]
        padding = net_output["padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][...,0] = 0
            logits[padding][...,1:] = float('-inf')

        return logits

# We fixed the bug of not overriding args when trained with hydra.
# It should be identical to the original one in fairseq when trained without hydra.
class Wav2VecEncoderV2(Wav2VecEncoder):
    def __init__(self, cfg: Wav2Vec2AsrConfig, tgt_dict=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            forced_overwrite(w2v_args.model, arg_overrides)
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert 'normalize' not in cfg or cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            model.load_state_dict(state["model"], strict=True)

        model.remove_pretraining_modules()

        FairseqEncoder.__init__(self, task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        self.upsample_factor = cfg.output_upsample
        if self.upsample_factor > 0:
            self.upsample = nn.Sequential(nn.Linear(d, d * self.upsample_factor), nn.GELU())
        else:
            self.upsample = None

        if tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj = Linear(d, cfg.decoder_embed_dim)
        else:
            self.proj = None
        self.output_dim = getattr(cfg, "decoder_embed_dim", d)
        self.w2v_args = w2v_args

        if cfg.drop_upsample_layers > 0:
            old_upsample = [layer for layer in self.w2v_model.encoder.upsample]
            stages = []
            stage = []
            for layer in old_upsample[1:-1]:
                if isinstance(layer, nn.ConvTranspose1d):
                    if len(stage) > 0:
                        stages.append(stage)
                    stage = [layer]
                else:
                    stage.append(layer)
            if len(stage) > 0:
                stages.append(stage)
            upsample = old_upsample[:1] + [layer for stage in stages[:-cfg.drop_upsample_layers] for layer in stage] + old_upsample[-1:]
            if len(upsample) == 2:
                self.w2v_model.encoder.upsample = nn.Identity()
            else:
                self.w2v_model.encoder.upsample = nn.Sequential(*upsample)

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }
        optional_args = ['audio_feats']
        for k in optional_args:
            if k in kwargs:
                w2v_args[k] = kwargs[k]

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.upsample:
            T, B, C = x.shape
            x = self.upsample(x).view(T, B, -1, C).transpose(1, 2).reshape(-1, B, C)
            padding_mask = padding_mask.repeat_interleave(self.upsample_factor, dim=1)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask.transpose(0, 1),  # T x B
            "padding_mask": padding_mask, # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(1, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out


@dataclass
class MaskWav2Vec2CtcConfig(Wav2Vec2CtcConfig):
    self_attn_mask_type: str = field(
        default='none', metadata={"help": "type of self-attention mask"}
    )

@register_model("mask_wav2vec_ctc")
class MaskWav2VecCtc(Wav2VecCtcV2):
    def __init__(self, w2v_encoder, args, task):
        super().__init__()
        self.w2v_encoder = w2v_encoder
        self.args = args

        w2v_args = w2v_encoder.w2v_model.args
        w2v_args.arch = 'mask_wav2vec2'
        w2v_args.self_attn_mask_type = args.self_attn_mask_type

        state_dict = w2v_encoder.w2v_model.state_dict()
        self.w2v_encoder.w2v_model = task.build_model(w2v_args)
        self.w2v_encoder.w2v_model.remove_pretraining_modules()
        self.w2v_encoder.w2v_model.load_state_dict(state_dict)
