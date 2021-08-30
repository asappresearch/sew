# SEW (Squeezed and Efficient Wav2vec)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The repo contains the code of the paper "[Performance-Efficiency Trade-offsin Unsupervised Pre-training for Speech Recognition]()" by Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q Weinberger, and Yoav Artzi.

## Model Checkpoints
<!-- <details><summary>Unsupervisedly Pre-trained on Librispeech 960h (click to unfold) </summary><p> -->

### Unsupervisedly Pre-trained on LibriSpeech 960h

Model | Pre-training updates | Dataset| Model
|---|---|---|---
W2V2-tiny        | 100K | [Librispeech 960h](http://www.openslr.org/12) | [download](https://papers-sew.awsdev.asapp.com/save-pre/w2v2-tiny-100k.pt)
W2V2-small       | 100K | [Librispeech 960h](http://www.openslr.org/12) | [download](https://papers-sew.awsdev.asapp.com/save-pre/w2v2-small-100k.pt)
W2V2-mid         | 100K | [Librispeech 960h](http://www.openslr.org/12) | [download](https://papers-sew.awsdev.asapp.com/save-pre/w2v2-mid-100k.pt)
W2V2-base        | 100K | [Librispeech 960h](http://www.openslr.org/12) | [download](https://papers-sew.awsdev.asapp.com/save-pre/w2v2-base-100k.pt)
SEW-tiny         | 100K | [Librispeech 960h](http://www.openslr.org/12) | [download](https://papers-sew.awsdev.asapp.com/save-pre/sew-tiny-100k.pt)
SEW-small        | 100K | [Librispeech 960h](http://www.openslr.org/12) | [download](https://papers-sew.awsdev.asapp.com/save-pre/sew-small-100k.pt)
SEW-mid          | 100K | [Librispeech 960h](http://www.openslr.org/12) | [download](https://papers-sew.awsdev.asapp.com/save-pre/sew-mid-100k.pt)
SEW-D-tiny       | 100K | [Librispeech 960h](http://www.openslr.org/12) | [download](https://papers-sew.awsdev.asapp.com/save-pre/sew-d-tiny-100k.pt)
SEW-D-small      | 100K | [Librispeech 960h](http://www.openslr.org/12) | [download](https://papers-sew.awsdev.asapp.com/save-pre/sew-d-small-100k.pt)
SEW-D-mid        | 100K | [Librispeech 960h](http://www.openslr.org/12) | [download](https://papers-sew.awsdev.asapp.com/save-pre/sew-d-mid-100k.pt)
SEW-D-mid (k127) | 100K | [Librispeech 960h](http://www.openslr.org/12) | [download](https://papers-sew.awsdev.asapp.com/save-pre/sew-d-mid-k127-100k.pt)
SEW-D-base       | 100K | [Librispeech 960h](http://www.openslr.org/12) | [download](https://papers-sew.awsdev.asapp.com/save-pre/sew-d-base-100k.pt)
SEW-D-base+      | 100K | [Librispeech 960h](http://www.openslr.org/12) | [download](https://papers-sew.awsdev.asapp.com/save-pre/sew-d-base%2B-100k.pt)
SEW-D-mid        | 400K | [Librispeech 960h](http://www.openslr.org/12) | [download](https://papers-sew.awsdev.asapp.com/save-pre/sew-d-mid-400k.pt)
SEW-D-mid (k127) | 400K | [Librispeech 960h](http://www.openslr.org/12) | [download](https://papers-sew.awsdev.asapp.com/save-pre/sew-d-mid-k127-400k.pt)
SEW-D-base+      | 400K | [Librispeech 960h](http://www.openslr.org/12) | [download](https://papers-sew.awsdev.asapp.com/save-pre/sew-d-base%2B-400k.pt)

<!-- </p></details> -->

<!-- <details><summary>Semi-supervised Librispeech ASR model (click to unfold)</summary><p>

Model | Pre-training updates | Finetuning split | Model
|---|---|---|---
SEW-tiny    | 100K | 100 hours | [download]()
SEW-D-mid   | 100K | 100 hours | [download]()
SEW-D-mid   | 400K | 100 hours | [download]()
SEW-D-base+ | 100K | 100 hours | [download]()
SEW-D-base+ | 400K | 100 hours | [download]()

</p></details> -->

## Usage

### Dependencies
The code is tested with [fairseq commit 05255f9](https://github.com/pytorch/fairseq/tree/05255f96410e5b1eaf3bf59b767d5b4b7e2c3a35), [deberta commit bf17ca4](https://github.com/microsoft/DeBERTa/tree/bf17ca43fa429875c823536b5993cdf783ae5049) and the following packages.

```
torch==1.8.0
torchaudio==0.8.0
tqdm==4.49.0
Hydra==2.5
hydra-core==1.0.4
fvcore==0.1.5.post20210330
omegaconf==2.0.5
einops==0.3.0
fire==0.2.1
```

#### Apex
Please install NVIDIA's [apex](https://github.com/NVIDIA/apex) with
```sh
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

#### wav2letter decoder
Currently, we are decoding with wav2letter v0.2 python binding at commit `96f5f9d`
Please install the python binding here
[https://github.com/flashlight/wav2letter/tree/96f5f9d3b41e01af0a031ee0d2604acd9ef3b1b0/bindings/python](https://github.com/flashlight/wav2letter/tree/96f5f9d3b41e01af0a031ee0d2604acd9ef3b1b0/bindings/python)
The newest commit `d5a93f0
` in v0.2 branch leads to worse WER for wav2vec 2.0 baselines.


### Installation
```sh
git clone https://github.com/asappresearch/sew.git
cd sew 
pip install -e .
```
### Pre-training

**Pre-training SEW models**

Run the following command where `$model_size` can be `tiny`, `small`, or `mid`, and `$ngpu` is tne number of GPUs you want to use.
```sh
bash scripts/pt-sew.sh $model_size $ngpu
```

**Pre-training SEW-D models**

```sh
bash scripts/pt-sew-d.sh $model_size $ngpu
```
where `$model_size` can be `tiny`, `small`, `mid`, `mid-k127`, `base`, or `base+`.

### Fine-tuning

Run the following script to fine-tune a model with the hyperparameters from wav2vec 2.0. 
```sh
bash scripts/ft-model.sh $pre_trained_model $split $ngpu
```
where `$pre_trained_model` can be either a W2V2, SEW, or a SEW-D model checkpoint and `$split` can be `10m`, `1h`, `10h`, or `100h`.

Here we also provide a set of hyperparameters which sets all dropouts the same as the pre-training stage, and we found it to be more stable.
```sh
bash scripts/ft-model-stable.sh $pre_trained_model $split $ngpu
```

If you see out of GPU memory error, please scale down the `dataset.max_tokens` and scale up the `optimization.update_freq` in `scripts/ft-model.sh`.
For example modifying these lines
```sh
  dataset.max_tokens=3200000 \
  optimization.update_freq="[$((8 / $ngpu))]" \
```
to
```sh
  dataset.max_tokens=1600000 \
  optimization.update_freq="[$((16 / $ngpu))]" \
```
which reduces the batch size and increases the gradient accumulation steps in order to use less GPU memory.

### Evaluation
1. Please run this script to prepare the official LibriSpeech 4-gram language model.
```sh
bash scripts/prepare_librispeech_lm.sh $kenlm_build_bin
```
where `$kenlm_build_bin` is the folder that contains the KenLM `build_binary` executable file (e.g. `/home/user/kenlm/build/bin`).

2. Then run this script to evaluate a pre-trained ASR model
```sh
python tools/eval_w2v.py tunelm --subsets '["dev-clean", "dev-other", "test-clean", "test-other"]' --model $asr_checkpoint
```