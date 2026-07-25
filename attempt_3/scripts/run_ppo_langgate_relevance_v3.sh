#!/bin/bash
# Stage 10 -- re-run Stage 9's dual_lagrangian_langgate_relevance_v2
# reward with a new _line_repeat_penalty added alongside the existing
# _repetition_penalty (dual_lagrangian_langgate_relevance_v3 spec, see
# verl_reward_v2.py's docstring), after Stage 9 closed the non-Latin
# tail-degeneration failure but surfaced a new one at similar prevalence
# (22-23% of completions): the prompt or system prompt repeated verbatim
# 3-14x with a short foreign-script filler token wedged between repeats,
# which partially evades _repetition_penalty's word-trigram-distinctness
# check by shifting trigram boundaries without eliminating the repetition
# a human reader would recognize immediately.
#
# _line_repeat_penalty is segment-level (line/sentence boundaries) and
# normalization-based (strips non-ASCII characters before comparing
# segments) rather than n-gram-based, so the filler token can't hide the
# repeat, and it's a flat penalty (3.0, matching _non_latin_penalty's own
# reasoning) rather than the graded 0-1.5 version tried and rejected
# first for landing in the same too-weak magnitude range that was the
# Stage 9 bug in the first place. Sanity-checked directly (not assumed)
# against all 34 real Stage 9 failure examples across both eval surfaces
# via attempt_3/scripts/measure_line_repetition.py before spending GPU
# time: every one now scores the full 3.0 penalty, with zero false
# positives against every clean completion in every prior stage's eval
# file in this repo (1225 completions scanned).
#
# save-freq=100 (single checkpoint, then prune critic) -- same disk
# discipline as Stage 9 (see its own script's comment for the Stage 5
# disk-full incident this avoids).
set -uo pipefail
cd /home/user/projects/llm-detox-hw
mkdir -p attempt_3/submissions

sudo docker run --rm --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/torch":/root/.cache/torch \
  -e TOXIC_REWARD=dual_lagrangian_langgate_relevance_v3:/workspace/attempt_3/checkpoints/rm_helpfulness:/workspace/attempt_3/checkpoints/rm_harmlessness \
  -e TOXIC_LAGRANGIAN_STATE_PATH=/workspace/attempt_3/checkpoints/_lagrangian_state_langgate_relevance_v3.json \
  -e TOXIC_LAGRANGIAN_UPDATE_EVERY=16 \
  -e HYDRA_FULL_ERROR=1 -e PYTHONPATH=/workspace -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 detoxify 2>&1 | tail -1 && \
           python -m attempt_3.scripts.run_ppo --algo ppo \
             --train-parquet attempt_3/data/train.parquet \
             --val-parquet attempt_3/data/val.parquet \
             --actor-path Qwen/Qwen2.5-0.5B \
             --out attempt_3/outputs/ppo_langgate_relevance_v3 \
             --reward dual_lagrangian_langgate_relevance_v3:/workspace/attempt_3/checkpoints/rm_helpfulness:/workspace/attempt_3/checkpoints/rm_harmlessness \
             --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
             --rollout-n 8 --max-response-length 64 --rollout-gpu-mem 0.25 \
             --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
             --save-freq 100 --test-freq 10" \
  > attempt_3/submissions/ppo_langgate_relevance_v3_log.txt 2>&1
rc=$?
echo "=== FINISHED dual_lagrangian_langgate_relevance_v3, exit=$rc at $(date -u) ==="

if [ $rc -eq 0 ]; then
  sudo docker run --rm --gpus all --ipc=host \
    -v "$(pwd)":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -w /workspace \
    verlai/verl:vllm023.dev1 \
    bash -c "pip install -q verl==0.8.0 2>&1 | tail -1 && \
             python -m verl.model_merger merge --backend fsdp \
               --local_dir /workspace/attempt_3/outputs/ppo_langgate_relevance_v3/global_step_100/actor \
               --target_dir /workspace/attempt_3/checkpoints/ppo_langgate_relevance_v3_merged"
  sudo chmod 644 attempt_3/checkpoints/ppo_langgate_relevance_v3_merged/model.safetensors
  echo "=== merged to attempt_3/checkpoints/ppo_langgate_relevance_v3_merged ==="

  sudo rm -rf attempt_3/outputs/ppo_langgate_relevance_v3/global_step_100/critic
  echo "=== pruned critic checkpoint ==="
fi
