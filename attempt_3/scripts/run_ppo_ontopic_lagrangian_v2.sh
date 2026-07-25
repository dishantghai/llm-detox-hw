#!/bin/bash
# Stage 12c -- re-run of Stage 12b's dual_lagrangian_multi_ontopic
# (identical spec/reward code, no changes needed there) with
# rm_harmlessness_v3 in place of the original rm_harmlessness.
#
# Motivation: Stage 12b's own run found harm_rm scoring 6/8 real
# toxic-echo-then-moralize exploits in or above genuine content's own
# range -- a real blind spot, not a relevance-gate-only problem (see
# LOGBOOK.md's Stage 12b write-up). rm_harmlessness_v3 was retrained
# with 400 naturally-generated (Nebius teacher, few-shot primed on the
# actual 8 exploits, NOT templated -- a first templated attempt was
# tried and validated as not generalizing, see the two commits before
# this script's own) echo+moralize negatives, and checked directly
# against those same 8 exploits before this run was launched: all 8 now
# score -2.3 to -15.2, down from 12-25, with no regression on either
# established red-team gate.
#
# Fresh Lagrangian state path (both harmlessness and ontopic multipliers
# start at lam=0.0) -- does not reuse Stage 12b's state file.
#
# Same hyperparameters as Stage 11/12a/12b for a direct four-way A/B,
# same disk discipline (single checkpoint, critic pruned after merge).
set -uo pipefail
cd /home/user/projects/llm-detox-hw
mkdir -p attempt_3/submissions

sudo docker run --rm --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/torch":/root/.cache/torch \
  -e TOXIC_REWARD=dual_lagrangian_multi_ontopic:/workspace/attempt_3/checkpoints/rm_helpfulness:/workspace/attempt_3/checkpoints/rm_harmlessness_v3:/workspace/attempt_3/checkpoints/rm_ontopic \
  -e TOXIC_LAGRANGIAN_STATE_PATH=/workspace/attempt_3/checkpoints/_lagrangian_state_ontopic_multi_v2.json \
  -e TOXIC_LAGRANGIAN_UPDATE_EVERY=16 \
  -e HYDRA_FULL_ERROR=1 -e PYTHONPATH=/workspace -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 detoxify 2>&1 | tail -1 && \
           python -m attempt_3.scripts.run_ppo --algo ppo \
             --train-parquet attempt_3/data/train.parquet \
             --val-parquet attempt_3/data/val.parquet \
             --actor-path Qwen/Qwen2.5-0.5B \
             --out attempt_3/outputs/ppo_ontopic_lagrangian_v2 \
             --reward dual_lagrangian_multi_ontopic:/workspace/attempt_3/checkpoints/rm_helpfulness:/workspace/attempt_3/checkpoints/rm_harmlessness_v3:/workspace/attempt_3/checkpoints/rm_ontopic \
             --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
             --rollout-n 8 --max-response-length 64 --rollout-gpu-mem 0.25 \
             --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
             --save-freq 100 --test-freq 10" \
  > attempt_3/submissions/ppo_ontopic_lagrangian_v2_log.txt 2>&1
rc=$?
echo "=== FINISHED dual_lagrangian_multi_ontopic, exit=$rc at $(date -u) ==="

if [ $rc -eq 0 ]; then
  sudo docker run --rm --gpus all --ipc=host \
    -v "$(pwd)":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -w /workspace \
    verlai/verl:vllm023.dev1 \
    bash -c "pip install -q verl==0.8.0 2>&1 | tail -1 && \
             python -m verl.model_merger merge --backend fsdp \
               --local_dir /workspace/attempt_3/outputs/ppo_ontopic_lagrangian_v2/global_step_100/actor \
               --target_dir /workspace/attempt_3/checkpoints/ppo_ontopic_lagrangian_v2_merged"
  sudo chmod 644 attempt_3/checkpoints/ppo_ontopic_lagrangian_v2_merged/model.safetensors
  echo "=== merged to attempt_3/checkpoints/ppo_ontopic_lagrangian_v2_merged ==="

  sudo rm -rf attempt_3/outputs/ppo_ontopic_lagrangian_v2/global_step_100/critic
  echo "=== pruned critic checkpoint ==="
fi
