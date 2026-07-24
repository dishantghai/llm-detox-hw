#!/bin/bash
# Stage 8 -- re-run Stage 7's langgate fix with tasks/task8_custom_reward.py's
# existing prompt-relevance gate added on top (dual_lagrangian_langgate_relevance:
# spec, see verl_reward_v2.py's _relevance_penalty), after Stage 7 closed the
# language-collapse and repetition-loop failures but was found to have
# converged onto a templated, prompt-independent "It's important to X... by
# doing Y..." response on 85-94% of completions across every eval surface.
# save-freq=100 (single checkpoint, then prune critic) -- learned from the
# disk-full incident during the Stage 5 chain.
set -uo pipefail
cd /home/user/projects/llm-detox-hw
mkdir -p attempt_3/submissions

sudo docker run --rm --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/torch":/root/.cache/torch \
  -e TOXIC_REWARD=dual_lagrangian_langgate_relevance:/workspace/attempt_3/checkpoints/rm_helpfulness:/workspace/attempt_3/checkpoints/rm_harmlessness \
  -e TOXIC_LAGRANGIAN_STATE_PATH=/workspace/attempt_3/checkpoints/_lagrangian_state_langgate_relevance.json \
  -e TOXIC_LAGRANGIAN_UPDATE_EVERY=16 \
  -e HYDRA_FULL_ERROR=1 -e PYTHONPATH=/workspace -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 detoxify 2>&1 | tail -1 && \
           python -m attempt_3.scripts.run_ppo --algo ppo \
             --train-parquet attempt_3/data/train.parquet \
             --val-parquet attempt_3/data/val.parquet \
             --actor-path Qwen/Qwen2.5-0.5B \
             --out attempt_3/outputs/ppo_langgate_relevance \
             --reward dual_lagrangian_langgate_relevance:/workspace/attempt_3/checkpoints/rm_helpfulness:/workspace/attempt_3/checkpoints/rm_harmlessness \
             --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
             --rollout-n 8 --max-response-length 64 --rollout-gpu-mem 0.25 \
             --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
             --save-freq 100 --test-freq 10" \
  > attempt_3/submissions/ppo_langgate_relevance_log.txt 2>&1
rc=$?
echo "=== FINISHED dual_lagrangian_langgate_relevance, exit=$rc at $(date -u) ==="
if [ $rc -eq 0 ]; then
  sudo rm -rf attempt_3/outputs/ppo_langgate_relevance/global_step_100/critic
  echo "=== pruned critic checkpoint ==="
fi
