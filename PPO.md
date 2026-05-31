##
2.8.3 版本 不行，报gcc 不兼容
data/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

## GPU 数为2，4，8

## 关闭wandb log

## 
export CUDA_VISIBLE_DEVICES=0,1
export N_GPUS=2
export BASE_MODEL="/home/jan5szh/workspaces/TinyZero-main/Qwen/Qwen2.5-3B-Instruct"
export DATA_DIR="/home/jan5szh/workspaces/TinyZero-main/data/countdown"
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS
export LOG_FILE_NAME="verl_qwen2.5-3b.log"
bash ./scripts/train_tiny_zero.sh

========================================== [ 1. 前向阶段：生成与评估 ] ==========================================

  Prompt (输入)
    │
    ├───> [ Policy Model (Actor) ] ────生成───> Response (Tokens) ──┐
    │                                                               │ (计算散度约束)
    ├───> [ Reference Model ] ──────────────────> Ref Logits ───────┼──> KL Divergence 
    │                                                               │         │
    ├───> [ Reward Model ] ─────────────────────> 评分 (Scalar) ────┘         ▼
    │                                                                   [ token_level_rewards ]
    │                                                                         │
    └───> [ Value Model (Critic) ] ─────────────> 预测价值 ───────────────────┼──> [ values ]
                                                                              │      │
                                                                              ▼      ▼
========================================== [ 2. 核心处理：GAE 估计 ] ==========================================
                                                                       
                                                                     [ compute_gae_advantage_return ]
                                                                              │              │
                                                                              ▼              ▼
                                                                        [ advantages ]  [ returns ]
                                                                              │              │
========================================== [ 3. 反向阶段：Loss 计算 ] ==========================================
                                                                              │              │
  新 Policy 再次前向计算新 Logits ───> 算 Ratio (新/旧概率比) ──────────────────┤              │
                                         │                                    ▼              ▼
                                         ▼                             [ policy_loss ] [ value_loss ]
                                  [ Policy Loss ]                             │              │
                                         │                                    ▼              ▼
                                         └───────────────────────────────> 梯度更新       梯度更新