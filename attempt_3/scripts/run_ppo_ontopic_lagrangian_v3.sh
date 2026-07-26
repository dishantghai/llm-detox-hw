#!/bin/bash
# Stage 12d -- re-run of Stage 12c's dual_lagrangian_multi_ontopic_v2 spec
# (rm_harmlessness_v3 unchanged) with the harmlessness cost's tanh
# calibration fixed: mu=16.0/sigma=6.0 (derived from rm_harmlessness_v3's
# own score distribution) instead of the mu=3.0/sigma=2.0 default every
# other spec in this file inherited from the ORIGINAL rm_harmlessness/
# rm_helpfulness pair and never revisited per-checkpoint.
#
# Root cause this fixes, per LOGBOOK.md's Stage 12c write-up: rm_harmlessness_v3
# itself generalized correctly (confirmed by scoring Stage 12c's own worst
# completions directly through it -- all flagged harmful), but the old
# mu=3/sigma=2 calibration saturated 98%+ of genuine completions to the
# same maximal-safe cost regardless of how safe they actually were, so the
# Lagrangian controller's batch-mean cost carried almost no gradient and
# harmlessness lambda only reached 0.074 in 115 updates (vs Stage 12b's
# 0.42 under the OLD RM, same code, same cadence). New spec
# `dual_lagrangian_multi_ontopic_v2:` (verl_reward_v2.py,
# `_dual_lagrangian_multi_ontopic_v2_score`) is byte-identical to Stage
# 12b/12c's reward code except this one calibration constant -- see that
# function's own docstring and `_HARM_V3_MU`'s calibration note for the
# full derivation and the direct before/after check against
# rm_harmlessness_v3's real training data.
#
# Fresh Lagrangian state path (both harmlessness and ontopic multipliers
# start at lam=0.0) -- does not reuse Stage 12b or 12c's state file.
#
# Same hyperparameters as Stage 11/12a/12b/12c for a direct five-way A/B,
# same disk discipline (single checkpoint, critic pruned after merge).
set -uo pipefail
cd /home/user/projects/llm-detox-hw
mkdir -p attempt_3/submissions

sudo docker run --rm --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/torch":/root/.cache/torch \
  -e TOXIC_REWARD=dual_lagrangian_multi_ontopic_v2:/workspace/attempt_3/checkpoints/rm_helpfulness:/workspace/attempt_3/checkpoints/rm_harmlessness_v3:/workspace/attempt_3/checkpoints/rm_ontopic \
  -e TOXIC_LAGRANGIAN_STATE_PATH=/workspace/attempt_3/checkpoints/_lagrangian_state_ontopic_multi_v3.json \
  -e TOXIC_LAGRANGIAN_UPDATE_EVERY=16 \
  -e HYDRA_FULL_ERROR=1 -e PYTHONPATH=/workspace -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 detoxify 2>&1 | tail -1 && \
           python -m attempt_3.scripts.run_ppo --algo ppo \
             --train-parquet attempt_3/data/train.parquet \
             --val-parquet attempt_3/data/val.parquet \
             --actor-path Qwen/Qwen2.5-0.5B \
             --out attempt_3/outputs/ppo_ontopic_lagrangian_v3 \
             --reward dual_lagrangian_multi_ontopic_v2:/workspace/attempt_3/checkpoints/rm_helpfulness:/workspace/attempt_3/checkpoints/rm_harmlessness_v3:/workspace/attempt_3/checkpoints/rm_ontopic \
             --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
             --rollout-n 8 --max-response-length 64 --rollout-gpu-mem 0.25 \
             --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
             --save-freq 100 --test-freq 10" \
  > attempt_3/submissions/ppo_ontopic_lagrangian_v3_log.txt 2>&1
rc=$?
echo "=== FINISHED dual_lagrangian_multi_ontopic_v2, exit=$rc at $(date -u) ==="

if [ $rc -eq 0 ]; then
  sudo docker run --rm --gpus all --ipc=host \
    -v "$(pwd)":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -w /workspace \
    verlai/verl:vllm023.dev1 \
    bash -c "pip install -q verl==0.8.0 2>&1 | tail -1 && \
             python -m verl.model_merger merge --backend fsdp \
               --local_dir /workspace/attempt_3/outputs/ppo_ontopic_lagrangian_v3/global_step_100/actor \
               --target_dir /workspace/attempt_3/checkpoints/ppo_ontopic_lagrangian_v3_merged"
  sudo chmod 644 attempt_3/checkpoints/ppo_ontopic_lagrangian_v3_merged/model.safetensors
  echo "=== merged to attempt_3/checkpoints/ppo_ontopic_lagrangian_v3_merged ==="

  sudo rm -rf attempt_3/outputs/ppo_ontopic_lagrangian_v3/global_step_100/critic
  echo "=== pruned critic checkpoint ==="
fi
