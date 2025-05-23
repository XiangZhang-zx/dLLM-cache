#!/bin/bash

# 设置模型
model="Dream-org/Dream-v0-Base-7B"

# 允许代码评估 (如果任务需要，保留)
export HF_ALLOW_CODE_EVAL=1

# Accelerate 配置和主进程端口
ACCEL_CONFIG="accelerate_config.yaml"
MAIN_PORT="29510" # 使用与 humaneval 相同的默认端口

echo "Starting evaluation for bbh"

# --- Task Specific Parameters for bbh ---
TASK="mmlu_pro"
NUM_FEWSHOT=0     # From tasks="... bbh", nshots="... 3"
MAX_NEW_TOKENS=256 # From tasks="... bbh", lengths="... 512"
DIFFUSION_STEPS=256 # Note: based on original script (equal to max_new_tokens)
TEMPERATURE=0.2    # From tasks="... bbh", temperatures="... 0"
TOP_P=0.95        # Constant in the original loop's model_args
ADD_BOS_TOKEN="true" # Constant in the original loop's model_args
# Note: original loop did NOT include escape_until=true

# 输出路径
OUTPUT_PATH="./${TASK}_log"

# 执行评估命令
accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg="entropy",alg_temp=0.0,add_bos_token=${ADD_BOS_TOKEN},prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 2 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --confirm_run_unsafe_code \
    --trust_remote_code \

accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg="entropy",alg_temp=0.0,add_bos_token=${ADD_BOS_TOKEN},prompt_interval_steps=25,gen_interval_steps=2,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 2 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --confirm_run_unsafe_code \
    --trust_remote_code \
 

echo "Completed evaluation for ${TASK}"