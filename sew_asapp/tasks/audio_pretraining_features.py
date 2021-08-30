import os
import sys
import torch

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any
from omegaconf import MISSING

from fairseq.data import AddTargetDataset, Dictionary, FileAudioDataset, encoders
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig

from fairseq.tasks import FairseqTask, register_task
from fairseq import utils
from fairseq.logging import metrics

from fairseq.tasks.audio_pretraining import AudioPretrainingTask, AudioPretrainingConfig

from ..data.audio_feat_dataset import FileAudioFeatDataset, FileAudioDatasetV2

class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


@dataclass
class AudioPretrainingFeaturesConfig(AudioPretrainingConfig):
    # new
    load_with_librosa: bool = field(
        default=False,
        metadata={
            "help": "load audio with librosa to support resampling the audio"
        },
    )
    fbank_dim: int = field(
        default=0,
        metadata={
            "help": "filter bank feature dimensions"
        },
    )
    mfcc_dim: int = field(
        default=0,
        metadata={
            "help": "MFCC feature dimensions"
        },
    )
    audio_feat_frame_length: float = field(
        default=25.,
        metadata={
            "help": "frame length of FBank or MFCC features in ms"
        },
    )
    audio_feat_frame_shift: float = field(
        default=10.,
        metadata={
            "help": "frame shift size of FBank or MFCC features in ms"
        },
    )
    


@register_task("audio_pretraining_features", dataclass=AudioPretrainingFeaturesConfig)
class AudioPretrainingFeaturesTask(FairseqTask):
    """"""
    cfg: AudioPretrainingFeaturesConfig

    def __init__(
        self,
        cfg: AudioPretrainingFeaturesConfig,
    ):
        super().__init__(cfg)
        if cfg.eval_wer:
            assert cfg.labels is not None, "eval_wer can only be set during fine-tuning"
        self.blank_symbol = "<s>"

        self.state.add_factory("target_dictionary", self.load_target_dictionary)

    @classmethod
    def setup_task(cls, cfg: AudioPretrainingFeaturesConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingFeaturesConfig): configuration of this task
        """

        return cls(cfg)

    def load_target_dictionary(self):
        if self.cfg.labels:
            dict_path = os.path.join(self.cfg.data, f"dict.{self.cfg.labels}.txt")
            return Dictionary.load(dict_path)
        return None

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == 'ctc'

        manifest = os.path.join(data_path, "{}.tsv".format(split))
        if self.cfg.fbank_dim == 0 and self.cfg.mfcc_dim == 0:
            self.datasets[split] = FileAudioDatasetV2(
                manifest,
                sample_rate=self.cfg.sample_rate,
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=self.cfg.labels is not None or self.cfg.enable_padding,
                normalize=self.cfg.normalize,
                use_librosa=self.cfg.load_with_librosa,
            )
        else:
            self.datasets[split] = FileAudioFeatDataset(
                manifest,
                sample_rate=self.cfg.sample_rate,
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=self.cfg.labels is not None or self.cfg.enable_padding,
                normalize=self.cfg.normalize,
                use_librosa=self.cfg.load_with_librosa,
                fbank_dim=self.cfg.fbank_dim,
                mfcc_dim=self.cfg.mfcc_dim,
                frame_length=self.cfg.audio_feat_frame_length,
                frame_shift=self.cfg.audio_feat_frame_shift,
            )

        if task_cfg.labels:
            label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
            with open(label_path, "r") as f:
                labels = [
                    line for i, line in enumerate(f)
                    if i in self.datasets[split].line_inds
                ]

            assert len(labels) == len(self.datasets[split]), (
                    f"labels length ({len(labels)}) and dataset length "
                    f"({len(self.datasets[split])}) do not match")

            process_label = LabelEncoder(self.target_dictionary)

            self.datasets[split] = AddTargetDataset(
                self.datasets[split],
                labels,
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
                batch_targets=True,
                process_label=process_label,
                add_to_input=getattr(task_cfg, 'autoregressive', False),
            )

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.state.target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self,
        indices,
        dataset,
        max_positions=None,
        ignore_invalid_inputs=False,
    ):
        # we do not need to filter by size in this task as dataloaders take care of this
        return indices

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_wer and self.cfg.autoregressive:
            metrics = self._inference_with_wer(self.sequence_generator, sample, model)
            logging_output["_num_char_errors"] = metrics["num_char_errors"]
            logging_output["_num_chars"] = metrics["num_chars"]
            logging_output["_num_word_errors"] = metrics["num_word_errors"]
            logging_output["_num_words"] = metrics["num_words"]
        return loss, sample_size, logging_output

    def build_model(self, model_cfg: FairseqDataclass):
        model = super().build_model(model_cfg)

        if self.cfg.eval_wer and self.cfg.autoregressive:
            self.sequence_generator = self.build_generator(
                [model],
                self.cfg.eval_wer_config,
            )
            if self.cfg.eval_wer_tokenizer:
                self.tokenizer = encoders.build_tokenizer(self.cfg.eval_wer_tokenizer)
            else:
                self.tokenizer = None
        return model

    def _inference_with_wer(self, generator, sample, model):
        import editdistance

        def decode(toks):
            s = self.target_dictionary.string(
                toks.int().cpu(),
                self.cfg.eval_wer_post_process,
                escape_unk=True,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        num_word_errors, num_char_errors = 0, 0
        num_chars, num_words = 0, 0
        gen_out = self.inference_step(generator, [model], sample, None)
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"])
            ref = decode(
                utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
            )
            num_char_errors += editdistance.eval(hyp, ref)
            num_chars += len(ref)
            hyp_words = hyp.split()
            ref_words = ref.split()
            num_word_errors += editdistance.eval(hyp_words, ref_words)
            num_words += len(ref_words)

        return {
            "num_char_errors": num_char_errors,
            "num_chars": num_chars,
            "num_word_errors": num_word_errors,
            "num_words": num_words,
        }

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        zero = torch.scalar_tensor(0.0)
        num_char_errors = sum(
            log.get("_num_char_errors", zero) for log in logging_outputs
        )
        num_chars = sum(log.get("_num_chars", zero) for log in logging_outputs)
        num_word_errors = sum(
            log.get("_num_word_errors", zero) for log in logging_outputs
        )
        num_words = sum(log.get("_num_words", zero) for log in logging_outputs)
        metrics.log_scalar("_num_char_errors", num_char_errors)
        metrics.log_scalar("_num_chars", num_chars)
        metrics.log_scalar("_num_word_errors", num_word_errors)
        metrics.log_scalar("_num_words", num_words)
        if num_words > 0:
            metrics.log_derived(
                "uer",
                lambda meters: meters["_num_char_errors"].sum
                * 100.0
                / meters["_num_chars"].sum
                if meters["_num_chars"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "wer",
                lambda meters: meters["_num_word_errors"].sum
                * 100.0
                / meters["_num_words"].sum
                if meters["_num_words"].sum > 0
                else float("nan"),
            )