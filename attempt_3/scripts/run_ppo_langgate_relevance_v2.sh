#!/bin/bash
# Stage 9 -- re-run Stage 8's dual_lagrangian_langgate_relevance reward
# with the whole-string _non_latin_penalty swapped for the windowed
# _non_latin_tail_penalty (dual_lagrangian_langgate_relevance_v2 spec,
# see verl_reward_v2.py's docstring), after Stage 8 closed the templating
# failure but was found to have 31-37% of completions open with clean
# content and then degrade into a non-Latin repetition-loop tail that the
# whole-string language gate's average-over-the-completion design missed.
#
# Decode-time fixes were tried first, against the frozen Stage 8
# checkpoint, and rejected (Stage 8.5, LOGBOOK.md): repetition_penalty
# made the tail worse, and script-targeted non-Latin logit suppression
# looked clean under greedy/OOD eval but increased worst-of-16 toxic
# support on the direct_provocation slice ~4x once checked under
# sampling -- blocking the model's "safe" (if ugly) escape hatch at eval
# time, on a policy that was trained without that constraint, just
# exposed worse content underneath rather than fixing anything. This run
# puts the fix where the policy actually learns from it instead: the
# reward function, during training.
#
# (verl 0.8.0's PPO rollout has no supported passthrough for a custom
# decode-time token suppressor either -- confirmed directly against the
# installed vLLM 0.23.1.dev0: RolloutConfig only forwards
# temperature/top_k/top_p/repetition_penalty/n, and vLLM's own
# SamplingParams dropped arbitrary logits_processors in favor of
# logit_bias/bad_words, neither of which verl's request-construction
# code exposes a config path for -- so this fix is reward-side only,
# not a rollout-sampling change.)
#
# save-freq=100 (single checkpoint, then prune critic) -- learned from
# the disk-full incident during the Stage 5 chain. Before launching this
# run, 6 redundant raw FSDP output dirs from completed/merged earlier
# stages were deleted (~36.6GB freed, 20GB -> 56GB free) since this run
# needs ~12-13GB of transient scratch (actor+critic) before pruning.
set -uo pipefail
cd /home/user/projects/llm-detox-hw
mkdir -p attempt_3/submissions

sudo docker run --rm --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/torch":/root/.cache/torch \
  -e TOXIC_REWARD=dual_lagrangian_langgate_relevance_v2:/workspace/attempt_3/checkpoints/rm_helpfulness:/workspace/attempt_3/checkpoints/rm_harmlessness \
  -e TOXIC_LAGRANGIAN_STATE_PATH=/workspace/attempt_3/checkpoints/_lagrangian_state_langgate_relevance_v2.json \
  -e TOXIC_LAGRANGIAN_UPDATE_EVERY=16 \
  -e HYDRA_FULL_ERROR=1 -e PYTHONPATH=/workspace -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 detoxify 2>&1 | tail -1 && \
           python -m attempt_3.scripts.run_ppo --algo ppo \
             --train-parquet attempt_3/data/train.parquet \
             --val-parquet attempt_3/data/val.parquet \
             --actor-path Qwen/Qwen2.5-0.5B \
             --out attempt_3/outputs/ppo_langgate_relevance_v2 \
             --reward dual_lagrangian_langgate_relevance_v2:/workspace/attempt_3/checkpoints/rm_helpfulness:/workspace/attempt_3/checkpoints/rm_harmlessness \
             --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
             --rollout-n 8 --max-response-length 64 --rollout-gpu-mem 0.25 \
             --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
             --save-freq 100 --test-freq 10" \
  > attempt_3/submissions/ppo_langgate_relevance_v2_log.txt 2>&1
rc=$?
echo "=== FINISHED dual_lagrangian_langgate_relevance_v2, exit=$rc at $(date -u) ==="

if [ $rc -eq 0 ]; then
  sudo docker run --rm --gpus all --ipc=host \
    -v "$(pwd)":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -w /workspace \
    verlai/verl:vllm023.dev1 \
    bash -c "pip install -q verl==0.8.0 2>&1 | tail -1 && \
             python -m verl.model_merger merge --backend fsdp \
               --local_dir /workspace/attempt_3/outputs/ppo_langgate_relevance_v2/global_step_100/actor \
               --target_dir /workspace/attempt_3/checkpoints/ppo_langgate_relevance_v2_merged"
  sudo chmod 644 attempt_3/checkpoints/ppo_langgate_relevance_v2_merged/model.safetensors
  echo "=== merged to attempt_3/checkpoints/ppo_langgate_relevance_v2_merged ==="

  sudo rm -rf attempt_3/outputs/ppo_langgate_relevance_v2/global_step_100/critic
  echo "=== pruned critic checkpoint ==="
fi
