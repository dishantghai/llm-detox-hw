#!/usr/bin/env bash
# Task 8 PPO run (custom reward). Run from the repo root:
#   bash scripts/run_task8_ppo.sh
set -euo pipefail

sudo docker run --rm --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/torch":/root/.cache/torch \
  -e TOXIC_REWARD=custom:tasks.task8_custom_reward \
  -e TOXIC_CUSTOM_RM_DIR=/workspace/checkpoints/rm \
  -e HYDRA_FULL_ERROR=1 \
  -e PYTHONPATH=/workspace \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 detoxify 2>&1 | tail -1 && \
           python -m src.toxic_rl.verl_runner --algo ppo \
             --train-parquet data/train.parquet \
             --val-parquet data/val.parquet \
             --actor-path Qwen/Qwen2.5-0.5B \
             --out outputs/ppo_custom \
             --reward custom:tasks.task8_custom_reward \
             --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
             --rollout-n 8 --max-response-length 64 \
             --rollout-gpu-mem 0.25 \
             --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
             --save-freq 20 --test-freq 10" \
  2>&1 | tee submissions/task8_log.txt
