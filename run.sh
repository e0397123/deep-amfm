#!/bin/bash
# general configuration

#SBATCH --job-name=compute
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=compute.log

export CUDA_VISIBLE_DEVICES=0

stage=1
stop_stage=5

# config files

# Data related
d_root=sample-data
SRC_LANG=en
TGT_LANG=km
TASK=ALT
SYSID=bad
# for translation from other languages to english
# uncomment the following line.
# am_model_path=./models/${TASK}/${TGT_LANG}-${SRC_LANG}_am
# for translation to other languages
# uncomment the following line
am_model_path=./models/${TASK}/${SRC_LANG}-${TGT_LANG}_am

fm_model_path=./models/${TASK}/${TGT_LANG}_lm
hyp_path=${d_root}/${TASK}/${TGT_LANG}/${SYSID}_hyp.txt
hyp_fm_output_path=${d_root}/${TASK}/${TGT_LANG}/${SYSID}_hyp.fm.prob
ref_path=${d_root}/${TASK}/${TGT_LANG}/${SYSID}_ref.txt
ref_fm_output_path=${d_root}/${TASK}/${TGT_LANG}/${SYSID}_ref.fm.prob
num_test_cases=200


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

result_path=./result

# ===========================================
# AM part
# ===========================================

if [ ! -d ${result_path} ]; then
	mkdir -p ${result_path}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Compute AM Score"
    python calc_am.py \
        --hyp_file=${hyp_path} \
        --ref_file=${ref_path} \
        --num_test=${num_test_cases} \
        --save_path=${result_path}/${TASK}_${SYSID}_${TGT_LANG}_am.score \
		--model_path=${am_model_path}
fi

## # ===========================================
## # FM part
## # ===========================================

## compute FM score
#
#

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
	echo "stage 2: Compute hypothesis sentence-level probability"
	python compute_ppl.py \
        --model_type=xlm-roberta \
        --output_dir=${fm_model_path} \
        --model_name_or_path=xlm-roberta-base \
        --do_eval \
        --eval_data_file=${hyp_path} \
        --overwrite_cache \
        --mlm
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Compute reference sentence-level probability"
	python compute_ppl.py \
        --model_type=xlm-roberta \
        --output_dir=${fm_model_path} \
        --model_name_or_path=xlm-roberta-base \
        --do_eval \
        --eval_data_file=${ref_path} \
        --overwrite_cache \
        --mlm
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Compute FM Score"
    python calc_fm.py \
        --hyp_file=${hyp_fm_output_path} \
        --ref_file=${ref_fm_output_path} \
        --num_test=${num_test_cases} \
        --save_path=${result_path}/${TASK}_${SYSID}_${TGT_LANG}_fm.score
fi


# ===========================================
# combined both scores
# ===========================================

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then

    echo "stage 5: Combine AM & FM scores"
    python amfm.py \
        --am_score=${result_path}/${TASK}_${SYSID}_${TGT_LANG}_am.score \
        --fm_score=${result_path}/${TASK}_${SYSID}_${TGT_LANG}_fm.score \
        --lambda_value=0.5 \
		--save_path=./result/${TASK}_${SYSID}_${TGT_LANG}_amfm.score
fi

echo "Thank you for using Deep AMFM Framework"
