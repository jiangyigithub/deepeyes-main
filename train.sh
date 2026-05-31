#!/bin/bash

# 1. 基础环境变量配置（改成 8 卡）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPUS=8
export BASE_MODEL="/home/jan5szh/workspaces/TinyZero-main/Qwen/Qwen2.5-3B-Instruct"
export DATA_DIR="/home/jan5szh/workspaces/TinyZero-main/data/countdown"
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS
export LOG_FILE_NAME="verl_qwen2.5-3b-4gpu_0522.log"

# 2. 使用 nohup 在后台运行训练脚本
# stdout 和 stderr 都会重定向到你定义的 LOG_FILE_NAME 中
nohup bash ./scripts/train_tiny_zero.sh > "$LOG_FILE_NAME" 2>&1 &

# 3. 打印提示信息
echo "训练任务已通过 nohup 提交到后台运行。"
echo "日志正在实时写入: $LOG_FILE_NAME"
echo "你可以使用以下命令查看运行状态: tail -f $LOG_FILE_NAME"