size=$1
ngpu=$2

data=`pwd`/manifest/librispeech/train-960
user_dir=`pwd`/sew_asapp

task=pt
root=example

config=wav2vec2_base_librispeech


export MKL_THREADING_LAYER=GNU
export PYTHONWARNINGS="ignore"

function set_dir {
    save=`pwd`/save-${task}/${root}/${tag}
    tb_save=`pwd`/tb-logs-${task}/${root}/${tag}
}


function train_model {
set_dir

case $wfe_type in
O)
# WFE-O
conv_feature_layers="[($wfe_c, 11, 5)] + [($wfe_c, 3, 2)] * 4 + [($wfe_c,2,2)] * 2"
;;
C)
# WFE-C
conv_feature_layers="""[($wfe_c, 10, 5), ($(($wfe_c * 2)), 3, 2)] + [($(($wfe_c * 2)), 1, 1)] * $wfe_l + \
[($(($wfe_c * 2)), 3, 2)] + [($(($wfe_c * 2)), 1, 1)] * $wfe_l + \
[($(($wfe_c * 4)), 3, 2)] + [($(($wfe_c * 4)), 1, 1)] * $wfe_l + \
[($(($wfe_c * 4)), 3, 2)] + [($(($wfe_c * 4)), 1, 1)] * $wfe_l + \
[($(($wfe_c * 8)), 2, 2)] + [($(($wfe_c * 8)), 1, 1)] * $wfe_l + \
[($(($wfe_c * 8)), 2, 2)] + [($(($wfe_c * 8)), 1, 1)] * $wfe_l"""
;;
esac


fairseq-hydra-train \
    hydra.run.dir=$save \
    hydra.output_subdir=$save \
    common.user_dir=$user_dir \
    common.tensorboard_logdir=$tb_save \
    common.log_interval=100 \
    task.data=$data \
    task._name=audio_pretraining_features \
    +task.fbank_dim=0 \
    +task.mfcc_dim=0 \
    distributed_training.distributed_world_size=$ngpu \
    dataset.max_tokens=$((1400000 * $batch_scale)) \
    optimization.update_freq="[$((64 / $batch_scale / $ngpu))]" \
    optimization.max_update=100000 \
    +lr_scheduler.total_num_update=400000 \
    model._name=$model_name \
    +model.conv_pos=$conv_pos \
    +model.conv_pos_groups=$conv_pos_groups \
    +model.encoder_layers=$l \
    model.encoder_embed_dim=$(($wf * 64)) \
    +model.encoder_ffn_embed_dim=$(($wf * 64 * 4)) \
    +model.encoder_attention_heads=$wf \
    +model.conv_feature_layers="'${conv_feature_layers}'" \
    +model.squeeze_factor=$squeeze \
    +model.fbank_dim=0 \
    +model.mfcc_dim=0 \
    model.encoder_layerdrop=$layerdrop \
    +model.use_mlp=true \
    +model.mlp_version=$mlp_version \
    +model.mlp_num_layers=$mlp_layers \
    +model.proj_mlp_norm_type=$mlp_norm_type \
    checkpoint.write_checkpoints_asynchronously=false \
    --config-dir config/pretraining \
    --config-name $config

}


# default
tag=sew-d-$size

model_name=squeeze_wav2vec2_deberta

squeeze=2
batch_scale=1
wf=6
l=12
layerdrop=0.05
mlp_version=v3
mlp_norm_type=bn
mlp_layers=2
conv_pos=31
conv_pos_groups=16
wfe_type=C
wfe_c=64
wfe_l=1


case $tag in 


sew-d-tiny)
batch_scale=2
wf=6
l=12
train_model

;;
sew-d-small)
wf=8
l=12
train_model

;;
sew-d-mid)
wf=8
l=24
layerdrop=0.2
train_model

;;
sew-d-mid-k127)
wf=8
l=24
layerdrop=0.2
conv_pos=127
train_model

;;
sew-d-base)
wf=12
l=24
layerdrop=0.2

train_model

;;
sew-d-base+)
wf=12
l=24
wfe_c=96
layerdrop=0.2

train_model

;;
*)
echo "invalid model size '$size'"


;;
esac


