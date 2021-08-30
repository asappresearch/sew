# Copyright (c) ASAPP Inc.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, List, Dict
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCArgMaxDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)

        from fairseq import search
        from fairseq.data.dictionary import Dictionary
        assert isinstance(tgt_dict, Dictionary)
        self.search = search.BeamSearch(tgt_dict)
        self.normalize_scores = True

        self.blank = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)
        return self.decode(emissions)

    def get_emissions(self, models, encoder_input):
        """Run encoder and normalize emissions"""
        encoder_out = models[0](**encoder_input)
        emissions = encoder_out['encoder_out'].transpose(0, 1).contiguous()
        emissions[encoder_out['padding_mask']] = 0.
        emissions[encoder_out['padding_mask']][:, self.blank] = 1.
        return emissions

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)
        return torch.LongTensor(list(idxs))

    def decode(self, emissions):
        B, T, N = emissions.size()

        # set length bounds
        toks = emissions.argmax(dim=-1)
        hypos = []
        for i in range(B):
            hypos.append([{
                'tokens': toks[i].unique_consecutive().cpu(),
                'score': emissions[i].sum().item(),
            }])
        return hypos

