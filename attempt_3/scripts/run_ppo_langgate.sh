#!/bin/bash
# Stage 7 -- re-run the Stage 6 fix with a language/coherence gate added to
# the reward (dual_lagrangian_langgate: spec, see verl_reward_v2.py's
# _non_latin_penalty), after the original dual_lagrangian run was found
# (this session) to collapse into fluent-looking Russian-script text on
# 100% of an out-of-distribution prompt set. save-freq=100 (single
# checkpoint, then prune critic) -- learned from the disk-full incident
# during the Stage 5 chain.
set -uo pipefail
cd /home/user/projects/llm-detox-hw
mkdir -p attempt_3/submissions

sudo docker run --rm --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/torch":/root/.cache/torch \
  -e TOXIC_REWARD=dual_lagrangian_langgate:/workspace/attempt_3/checkpoints/rm_helpfulness:/workspace/attempt_3/checkpoints/rm_harmlessness \
  -e TOXIC_LAGRANGIAN_STATE_PATH=/workspace/attempt_3/checkpoints/_lagrangian_state_langgate.json \
  -e TOXIC_LAGRANGIAN_UPDATE_EVERY=16 \
  -e HYDRA_FULL_ERROR=1 -e PYTHONPATH=/workspace -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 detoxify 2>&1 | tail -1 && \
           python -m attempt_3.scripts.run_ppo --algo ppo \
             --train-parquet attempt_3/data/train.parquet \
             --val-parquet attempt_3/data/val.parquet \
             --actor-path Qwen/Qwen2.5-0.5B \
             --out attempt_3/outputs/ppo_langgate \
             --reward dual_lagrangian_langgate:/workspace/attempt_3/checkpoints/rm_helpfulness:/workspace/attempt_3/checkpoints/rm_harmlessness \
             --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
             --rollout-n 8 --max-response-length 64 --rollout-gpu-mem 0.25 \
             --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
             --save-freq 100 --test-freq 10" \
  > attempt_3/submissions/ppo_langgate_log.txt 2>&1
rc=$?
echo "=== FINISHED dual_lagrangian_langgate, exit=$rc at $(date -u) ==="
if [ $rc -eq 0 ]; then
  sudo rm -rf attempt_3/outputs/ppo_langgate/global_step_100/critic
  echo "=== pruned critic checkpoint ==="
fi
