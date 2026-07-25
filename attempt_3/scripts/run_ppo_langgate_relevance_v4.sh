#!/bin/bash
# Stage 11 -- re-run Stage 10's dual_lagrangian_langgate_relevance_v3
# reward with a new _compression_repeat_penalty added alongside
# _repetition_penalty and _line_repeat_penalty
# (dual_lagrangian_langgate_relevance_v4 spec, see verl_reward_v2.py's
# docstring), after Stage 10 closed the sentence-level filler-token
# evasion but its own tracked-set output still contained a
# sub-word/token-level repeat loop short enough to dodge both the
# word-trigram and line-segment checks -- e.g. "...itatica, itatica,
# itatica, ..." and "...tiny penis forCell with other tiny penis
# forCell...", both scored 0.985/0.990 toxic by Detoxify.
#
# Before wiring this up, a narrower fix (a closed-list phrase-blocklist
# gate targeting the separate Stage-7-style "it's important to"/"I
# understand" template-collapse pattern also seen in Stage 10's output)
# was considered and explicitly researched instead of assumed sufficient:
# scoring Stage 10's own templated vs non-templated completions through
# the actual help_rm showed the RM itself scores the boilerplate far
# higher (mean 4.28) than genuine engagement (mean -3.03) -- a
# reward-model-level bias a reward-side phrase gate wouldn't fix at the
# root, and which doesn't touch the higher-toxicity token-loop failure
# at all. _compression_repeat_penalty was chosen instead because it's
# validated directly (not assumed) against all 260 real completions
# across every Stage 9/10 eval file in this repo: a zlib compression
# ratio < 0.40 threshold catches every known repetition-loop failure,
# including several the existing word-trigram and line-segment gates
# miss entirely, with zero false positives across that whole corpus
# (see LOGBOOK.md's Stage 10/11 sections and
# attempt_3/scripts/measure_compression_repeat.py). The template-
# collapse issue is a separate, still-open problem this run does not
# address -- it needs its own fix, tracked separately.
#
# save-freq=100 (single checkpoint, then prune critic) -- same disk
# discipline as every prior stage.
set -uo pipefail
cd /home/user/projects/llm-detox-hw
mkdir -p attempt_3/submissions

sudo docker run --rm --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/torch":/root/.cache/torch \
  -e TOXIC_REWARD=dual_lagrangian_langgate_relevance_v4:/workspace/attempt_3/checkpoints/rm_helpfulness:/workspace/attempt_3/checkpoints/rm_harmlessness \
  -e TOXIC_LAGRANGIAN_STATE_PATH=/workspace/attempt_3/checkpoints/_lagrangian_state_langgate_relevance_v4.json \
  -e TOXIC_LAGRANGIAN_UPDATE_EVERY=16 \
  -e HYDRA_FULL_ERROR=1 -e PYTHONPATH=/workspace -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 detoxify 2>&1 | tail -1 && \
           python -m attempt_3.scripts.run_ppo --algo ppo \
             --train-parquet attempt_3/data/train.parquet \
             --val-parquet attempt_3/data/val.parquet \
             --actor-path Qwen/Qwen2.5-0.5B \
             --out attempt_3/outputs/ppo_langgate_relevance_v4 \
             --reward dual_lagrangian_langgate_relevance_v4:/workspace/attempt_3/checkpoints/rm_helpfulness:/workspace/attempt_3/checkpoints/rm_harmlessness \
             --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
             --rollout-n 8 --max-response-length 64 --rollout-gpu-mem 0.25 \
             --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
             --save-freq 100 --test-freq 10" \
  > attempt_3/submissions/ppo_langgate_relevance_v4_log.txt 2>&1
rc=$?
echo "=== FINISHED dual_lagrangian_langgate_relevance_v4, exit=$rc at $(date -u) ==="

if [ $rc -eq 0 ]; then
  sudo docker run --rm --gpus all --ipc=host \
    -v "$(pwd)":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -w /workspace \
    verlai/verl:vllm023.dev1 \
    bash -c "pip install -q verl==0.8.0 2>&1 | tail -1 && \
             python -m verl.model_merger merge --backend fsdp \
               --local_dir /workspace/attempt_3/outputs/ppo_langgate_relevance_v4/global_step_100/actor \
               --target_dir /workspace/attempt_3/checkpoints/ppo_langgate_relevance_v4_merged"
  sudo chmod 644 attempt_3/checkpoints/ppo_langgate_relevance_v4_merged/model.safetensors
  echo "=== merged to attempt_3/checkpoints/ppo_langgate_relevance_v4_merged ==="

  sudo rm -rf attempt_3/outputs/ppo_langgate_relevance_v4/global_step_100/critic
  echo "=== pruned critic checkpoint ==="
fi
