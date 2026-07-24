#!/bin/bash
# Stage 5 — runs the three baseline PPO reward variants (GUIDE.md §7.2) back
# to back on the single H100: inv:detoxify, rm:rm_harmlessness (see
# LOGBOOK.md's Stage 4 decision-gate note for why the harmlessness RM alone
# stands in for GUIDE.md's literal single "rm" checkpoint -- attempt_3 built
# two RMs, not one, and this is the one GUIDE.md's slot maps to), and
# custom:tasks.task8_custom_reward. Sequential because there's one GPU and
# each run wants its own actor+critic+rollout copy of Qwen2.5-0.5B.
#
# Uses attempt_3/scripts/run_ppo_stage5.py, which is just an unmodified
# pass-through of the base verl_runner build_command (an earlier version
# tried a batched-reward-manager perf fix; that reward manager doesn't
# exist in this verl build's actual runtime registry -- see that file's
# docstring for the full story).
#
# save-freq=100 (only the final step), not 20: the first Stage 6 run
# (save-freq=20) wrote a full actor+critic FSDP checkpoint -- weights +
# AdamW optimizer states, ~12GB -- at steps 20/40/60/80/100, ~60GB total,
# and filled the 193GB host disk outright, killing a checkpoint save
# mid-run. Only the final checkpoint's actor weights are ever read (by
# the merge step), so intermediate saves are pure waste here. This is a
# pure I/O-cadence change -- doesn't touch training dynamics -- so it
# doesn't compromise the "same flags" comparison GUIDE.md asks for.
set -uo pipefail
cd /home/user/projects/llm-detox-hw
mkdir -p attempt_3/submissions

run_variant() {
  local reward_spec="$1" out_dir="$2" log_file="$3"
  echo "=== STARTING $reward_spec -> $out_dir at $(date -u) ==="
  sudo docker run --rm --gpus all --ipc=host \
    -v "$(pwd)":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -v "$HOME/.cache/torch":/root/.cache/torch \
    -e TOXIC_REWARD="$reward_spec" -e HYDRA_FULL_ERROR=1 -e PYTHONPATH=/workspace -w /workspace \
    verlai/verl:vllm023.dev1 \
    bash -c "pip install -q verl==0.8.0 detoxify 2>&1 | tail -1 && \
             python -m attempt_3.scripts.run_ppo_stage5 --algo ppo \
               --train-parquet attempt_3/data/train.parquet \
               --val-parquet attempt_3/data/val.parquet \
               --actor-path Qwen/Qwen2.5-0.5B \
               --out $out_dir \
               --reward '$reward_spec' \
               --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
               --rollout-n 8 --max-response-length 64 --rollout-gpu-mem 0.25 \
               --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
               --save-freq 100 --test-freq 10" \
    > "$log_file" 2>&1
  local rc=$?
  echo "=== FINISHED $reward_spec, exit=$rc at $(date -u) ==="
  if [ $rc -eq 0 ]; then
    echo "=== pruning critic checkpoint (only actor needed for merge) ==="
    sudo rm -rf "$out_dir/global_step_100/critic"
  fi
  return $rc
}

run_variant "inv:detoxify" "attempt_3/outputs/ppo_inv_detoxify" "attempt_3/submissions/ppo_detoxify_log.txt" \
&& run_variant "rm:/workspace/attempt_3/checkpoints/rm_harmlessness" "attempt_3/outputs/ppo_rm" "attempt_3/submissions/ppo_rm_log.txt" \
&& run_variant "custom:tasks.task8_custom_reward" "attempt_3/outputs/ppo_custom" "attempt_3/submissions/ppo_custom_log.txt"

echo "=== STAGE 5 CHAIN COMPLETE, overall exit=$? ==="
