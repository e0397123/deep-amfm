#!/bin/bash
# general configuration

#SBATCH --job-name=compute
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=compute.log


stage=2
stop_stage=4

# config files

# Data related
d_root=data
LANG=th
TASK=ALT-SAP
SYSID=bad
hyp_path=${d_root}/${TASK}/${LANG}/${SYSID}_hyp.txt
hyp_fm_output_path=${d_root}/${TASK}/${LANG}/${SYSID}_hyp.fm.prob
ref_path=${d_root}/${TASK}/${LANG}/ref.txt
ref_fm_output_path=${d_root}/${TASK}/${LANG}/ref.fm.prob
num_test_cases=10


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# ===========================================
# AM part
# ===========================================
am_model_path=./models/${TASK}/${LANG}/am/${LANG}.bin
am_result_file=./result

if [ ! -d ${am_result_file} ]; then
	mkdir -p ${am_result_file}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Compute AM Score"
    python calc_am.py \
        --hyp_file=${hyp_path} \
        --ref_file=${ref_path} \
        --num_test=${num_test_cases} \
        --save_path=${am_result_file}/${TASK}_${SYSID}_am.score \
		--model_path=${am_model_path} \
		--lang=${LANG}
fi

## # ===========================================
## # FM part
## # ===========================================

fm_model_path=./models/${TASK}/${LANG}_lm
fm_result_file=./result

if [ ! -d ${fm_result_file} ]; then
	mkdir -p ${fm_result_file}
fi

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
        --per_gpu_eval_batch_size=1 \
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
        --per_gpu_eval_batch_size=1 \
        --overwrite_cache \
        --mlm
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Compute FM Score"
    python calc_fm.py \
        --hyp_file=${hyp_fm_output_path} \
        --ref_file=${ref_fm_output_path} \
        --num_test=${num_test_cases} \
        --save_path=${fm_result_file}/${TASK}_${SYSID}_fm.score
fi


# ===========================================
# combined both scores
# ===========================================

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then

    echo "stage 5: Combine AM & FM scores"
    python amfm.py \
        --am_score=${am_result_file}/${SYSID}_am.score \
        --fm_score=${fm_result_file}/${SYSID}_fm.score \
        --lambda_value=0.5 \
		--save_path=./result/${TASK}/${LANG}/${SYSID}_amfm.score
fi

echo "Thank you for using Deep AMFM Framework"
