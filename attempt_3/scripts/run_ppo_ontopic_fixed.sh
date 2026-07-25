#!/bin/bash
# Stage 12a -- fixed-weight A/B baseline for the coherence-RM fix scoped
# after Stage 11 (see LOGBOOK.md). Adds rm_ontopic's _ontopic_penalty as
# one more additive term on top of v4's full stack (language-tail,
# repetition x3, relevance), NOT yet promoted to a Lagrangian constraint --
# that's the deliberately separate, bigger next step, and this run exists
# to give it something to A/B against.
#
# rm_ontopic itself required two fixes before this run was safe to launch,
# both found by testing against real policy completions rather than
# trusting the first training pass (see the two commits before this one):
# a severe training-data skew (98:2 dpo_diverse-vs-benign-topic positives,
# which taught it "good" as "declines a hostile prompt" rather than
# genuine on-topicness) and, even after fixing that, a residual
# Bradley-Terry scale-inconsistency bias against benign-topic content that
# calibration alone only partially closes -- documented directly in
# verl_reward_v2.py's _ONTOPIC_MU/_ONTOPIC_SIGMA comment. Launching anyway
# with that limitation on the record: the point of this run is exactly to
# see how a real 100-step PPO run behaves under a fixed-weight, imperfectly
# calibrated version of this signal, which is itself useful evidence for
# whether the Lagrangian-constraint version (Stage 12b) is worth building.
#
# Same hyperparameters as Stage 11 (v4) for a direct A/B, same disk
# discipline (single checkpoint, critic pruned after merge).
set -uo pipefail
cd /home/user/projects/llm-detox-hw
mkdir -p attempt_3/submissions

sudo docker run --rm --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/torch":/root/.cache/torch \
  -e TOXIC_REWARD=dual_lagrangian_langgate_relevance_ontopic:/workspace/attempt_3/checkpoints/rm_helpfulness:/workspace/attempt_3/checkpoints/rm_harmlessness:/workspace/attempt_3/checkpoints/rm_ontopic \
  -e TOXIC_LAGRANGIAN_STATE_PATH=/workspace/attempt_3/checkpoints/_lagrangian_state_ontopic_fixed.json \
  -e TOXIC_LAGRANGIAN_UPDATE_EVERY=16 \
  -e HYDRA_FULL_ERROR=1 -e PYTHONPATH=/workspace -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 detoxify 2>&1 | tail -1 && \
           python -m attempt_3.scripts.run_ppo --algo ppo \
             --train-parquet attempt_3/data/train.parquet \
             --val-parquet attempt_3/data/val.parquet \
             --actor-path Qwen/Qwen2.5-0.5B \
             --out attempt_3/outputs/ppo_ontopic_fixed \
             --reward dual_lagrangian_langgate_relevance_ontopic:/workspace/attempt_3/checkpoints/rm_helpfulness:/workspace/attempt_3/checkpoints/rm_harmlessness:/workspace/attempt_3/checkpoints/rm_ontopic \
             --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
             --rollout-n 8 --max-response-length 64 --rollout-gpu-mem 0.25 \
             --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
             --save-freq 100 --test-freq 10" \
  > attempt_3/submissions/ppo_ontopic_fixed_log.txt 2>&1
rc=$?
echo "=== FINISHED dual_lagrangian_langgate_relevance_ontopic, exit=$rc at $(date -u) ==="

if [ $rc -eq 0 ]; then
  sudo docker run --rm --gpus all --ipc=host \
    -v "$(pwd)":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -w /workspace \
    verlai/verl:vllm023.dev1 \
    bash -c "pip install -q verl==0.8.0 2>&1 | tail -1 && \
             python -m verl.model_merger merge --backend fsdp \
               --local_dir /workspace/attempt_3/outputs/ppo_ontopic_fixed/global_step_100/actor \
               --target_dir /workspace/attempt_3/checkpoints/ppo_ontopic_fixed_merged"
  sudo chmod 644 attempt_3/checkpoints/ppo_ontopic_fixed_merged/model.safetensors
  echo "=== merged to attempt_3/checkpoints/ppo_ontopic_fixed_merged ==="

  sudo rm -rf attempt_3/outputs/ppo_ontopic_fixed/global_step_100/critic
  echo "=== pruned critic checkpoint ==="
fi
