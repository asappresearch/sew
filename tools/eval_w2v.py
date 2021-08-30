# Copyright (c) ASAPP Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import fire
import shlex
import subprocess


def run_exp(
    model='save/pretrained/wav2vec_small_100h.pt',
    batch_scale=1,
    lm="nolm-argmax",
    beam_size=50,
    lm_weight=2.,
    word_score=-1.,
    subset="dev-other",
    data="manifest/librispeech",
    upsample=1.,
    save_results=False,
    dump_emissions=False,
    ctc_temp=1.,
    csv_log_file='exp-eval-logs.csv',
    fp16=False,
    batch_size=-1,
    quiet=False,
    use_bpe=False,
):
    if os.path.isdir(model):
        ckpt = os.path.join(model, 'checkpoints/checkpoint_best.pt')
        # eval_log_file = os.path.join(ckpt, 'eval.log')
        eval_log_file = None
        if save_results:
            if "nolm" in lm:
                results_path = os.path.join(model, 'decode', subset, lm)
            else:
                results_path = os.path.join(model, 'decode', subset,
                                            f'{lm}-b{beam_size}-lw{lm_weight}-ws{word_score}')
        else:
            results_path = None
        emission_path = os.path.join(
            model, 'decode', subset, 'emissions.npy') if dump_emissions else None
    else:
        ckpt = model
        eval_log_file = None
        if save_results:
            if "nolm" in lm:
                results_path = os.path.join(
                    os.path.basename(model) + '-decode', subset, lm)
            else:
                results_path = os.path.join(
                    os.path.basename(model) + '-decode', subset,
                    f'{lm}-b{beam_size}-lw{lm_weight}-ws{word_score}')
        else:
            results_path = None
        emission_path = os.path.join(os.path.basename(
            model) + '-decode', subset) if dump_emissions else None

    if not quiet:
        print(f"ckpt: {ckpt}")
        print(f"lm: {lm}")
        if 'nolm' not in lm:
            print(f"lm_weight: {lm_weight} word_score: {word_score} beam_size: {beam_size}")

    user_dir = os.path.abspath("sew_asapp")
    max_tokens = 4000000 * batch_scale

    cmd = (
        f"python tools/infer.py {data}"
        f" --user-dir {user_dir}"
        f" --task audio_pretraining_features"
        f" --nbest 1 --path {ckpt} --gen-subset {subset}"
        f" --sil-weight 0 --max-tokens {max_tokens}"
        f" --lm-weight {lm_weight} --word-score {word_score}"
        f" --criterion ctc"
        f" --beam {beam_size}" 
        f" --eval-upsample {upsample}"
        # f" --beam-size-token {beam_size}"
        # f" --beam-threshold {beam_size}"
    )

    if results_path is not None:
        cmd += f" --results-path {results_path}"
    if emission_path is not None:
        cmd += f" --dump-emissions {emission_path}"
    if "bpe" in ckpt or use_bpe:
        cmd += " --labels bpe --post-process sentencepiece"
    else:
        cmd += " --labels ltr --post-process letter"
    if ctc_temp != 1.:
        cmd += f" --eval-temperature {ctc_temp}"
    if batch_size > 0:
        cmd += f" --batch-size {batch_size}"

    if lm == "nolm":
        cmd += " --w2l-decoder viterbi" 
    elif lm == "nolm-argmax":
        cmd += " --w2l-decoder argmax"
    else:
        cmd += f" --w2l-decoder kenlm --lm-model save/kenlm/{lm}/4gram.bin --lexicon save/kenlm/{lm}/lexicon.lst"

    if fp16:
        cmd += " --fp16"
        

    if "vox" in ckpt:
        cmd += " --normalize"

    if eval_log_file is not None:
        os.makedirs(os.path.dirname(eval_log_file), exist_ok=True)
        cmd += f" | tee -a {eval_log_file}"

    if not quiet:
        print("cmd:")
        print(cmd)
    result = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wer, time_used, model_size, extract_size = parse_result(result, quiet=quiet)

    if not quiet:
        print(f"WER: {wer} time_used: {time_used} model_size: {model_size} extract_size: {extract_size}")
    msg = f'{subset},{model},{lm},{model_size},{extract_size},{time_used},{wer}'
    if not quiet:
        print(msg)
    if csv_log_file is not None:
        with open(csv_log_file, 'a') as f:
            print(msg, file=f)

    return wer, time_used, model_size, extract_size 


def parse_result(result, quiet=False):
    extract_size = 0
    model_size = 0
    wer = -1
    time_used = -1
    for line in result.stderr.decode("utf-8").split('\n'):
        if not quiet:
            print(line)
        pos = line.find("WER: ")
        if pos >= 0:
            wer = float(line[pos+5:].rstrip())

        pos = line.find("time used: ")
        if pos >= 0:
            time_used = float(line[pos+11:].rstrip())

        query = 'model 0 size: '
        pos = line.find(query)
        if pos >= 0:
            model_size = int(line[pos+len(query):].rstrip())

        query = 'w2v_encoder.w2v_model.feature_extractor size: '
        pos = line.find(query)
        if pos >= 0:
            extract_size += int(line[pos+len(query):].rstrip())

        query = 'w2v_encoder.w2v_model.spec_feature_extractor size: '
        pos = line.find(query)
        if pos >= 0:
            extract_size += int(line[pos+len(query):].rstrip())

    return wer, time_used, model_size, extract_size 
    

def tune_lm(
    model='save/pretrained/wav2vec_small_100h.pt',
    batch_scale=1,
    lms=["nolm-argmax", "librispeech-official"],
    beam_size=50,
    lm_weight=2.,
    word_score=-1.,
    subsets=["dev-other"],
    data="manifest/librispeech",
    upsample=1.,
    save_results=False,
    dump_emissions=False,
    ctc_temp=1.,
    csv_log_file='exp-eval-logs.csv',
    fp16=False,
    batch_size=-1,
    use_bpe=False,
):
    for lm in lms:
        for subset in subsets:
            try:
                run_exp(
                    model, batch_scale, lm, 
                    beam_size=beam_size, 
                    lm_weight=lm_weight,
                    word_score=word_score,
                    subset=subset, 
                    data=data, 
                    upsample=upsample,
                    save_results=save_results,
                    dump_emissions=dump_emissions,
                    ctc_temp=ctc_temp,
                    csv_log_file=csv_log_file,
                    fp16=fp16,
                    batch_size=batch_size,
                    use_bpe=use_bpe,
                )
            except:
                pass
            print("-"*80)
            # only need to dump once per subset
            dump_emissions = False


def run_folder(
    root='save-ft-100h/example',
    batch_scale=1,
    lms=["nolm-argmax", "librispeech-official"],
    beam_size=50,
    lm_weight=2.,
    word_score=-1.,
    subsets=["dev-other"],
    data="manifest/librispeech",
    log_file="tune_lm.csv",
    upsample=1.,
    save_results=False,
    dump_emissions=False,
    ctc_temp=1.,
    checkpoint_name='checkpoint_best.pt',
    skip=0,
    fp16=False,
    batch_size=-1,
    csv_log_file='exp-eval-logs.csv',
    use_bpe=False,
):
    exp_dirs = []
    for dirname, dirs, files in os.walk(root):
        if checkpoint_name in files:
            exp_dirs.append(os.path.join(dirname, checkpoint_name))
    print('skipped folders:', *exp_dirs[:skip], sep='\n')
    exp_dirs = exp_dirs[skip:]
    print('folders:', *exp_dirs, sep='\n')
    print('')

    for model in exp_dirs: 
        tune_lm(
            model=model,
            batch_scale=batch_scale,
            lms=lms,
            beam_size=beam_size, 
            lm_weight=lm_weight,
            word_score=word_score,
            subsets=subsets, 
            data=data, 
            upsample=upsample,
            save_results=save_results,
            dump_emissions=dump_emissions,
            ctc_temp=ctc_temp,
            fp16=fp16,
            batch_size=batch_size,
            csv_log_file=csv_log_file,
            use_bpe=use_bpe,
        )
        print("="*80)


def time_folder(
    file_list="tools/time_file_list.txt",
    output_file="time-eval-logs.csv",
    fp16=False,
    repeat=5,
):
    with open(file_list) as f:
        folders = [line.strip() for line in f]
    if os.path.exists(output_file):
        with open(output_file) as f:
            finished = set([line.split(',')[0].strip() for line in f])
    else:
        finished = set([])
    
    folders = [d for d in folders if d not in finished]
    print("finished:")
    print(*finished, sep='\n')
    print()
    print("folders:")
    print(*folders, sep='\n')
    print()

    for folder in folders:
        try:
            msg = f"{folder}"
            print(folder)
            for r in range(repeat):
                wer, time_used, model_size, extract_size = run_exp(
                    folder, 1, 
                    lm="nolm-argmax", 
                    subset="dev-other", 
                    csv_log_file=None,
                    fp16=fp16,
                    quiet=True,
                )
                msg += f",{time_used}"
                print(time_used)
            
            with open(output_file, "a") as f:
                print(msg, file=f)
        except:
            pass

    return 

if __name__ == '__main__':
    fire.Fire({
        'run': run_exp,
        'tunelm': tune_lm,
        'runf': run_folder,
        'timef': time_folder,
    })
