"""Stage 4 — train the helpfulness RM + harmlessness RM pair.

Mirrors `attempt_2/src/toxic_rl/train_dual_rm.py`'s design exactly (Safe
RLHF's decoupled reward/cost architecture): call the ORIGINAL, unmodified
``src.detox_hw.train_rm.train`` twice against two different negative-side
views of ``attempt_3/data/dpo_dual.jsonl`` (built by
``data_prep/build_dual_rm_data.py``):

- **harmlessness RM**: ``chosen`` vs. ``rejected_toxic`` — what
  `GUIDE.md`'s literal Stage 4 instructions describe on their own.
- **helpfulness RM**: ``chosen`` vs. ``rejected_evasive`` — the axis
  Stage 3's DPO run showed is missing. Nothing in the single-RM setup
  ever taught a model that a generic non-answer scores worse than a real
  one; this is the new contrast.

No new training code — same `train_rm.py` Bradley-Terry loss, LoRA setup,
held-out accuracy. Only the data view changes.

Usage (from repo root):
    python -m attempt_3.scripts.train_dual_rm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _split_view(data_path: Path, rejected_key: str, out_path: Path) -> int:
    n = 0
    with data_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            fout.write(json.dumps({
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row[rejected_key],
            }) + "\n")
            n += 1
    return n


def main() -> None:
    from src.detox_hw.train_rm import TrainConfig, train

    p = argparse.ArgumentParser()
    p.add_argument("--data", default="attempt_3/data/dpo_dual.jsonl")
    p.add_argument("--out-dir", default="attempt_3/checkpoints")
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--val-fraction", type=float, default=0.1)
    a = p.parse_args()

    data_path = Path(a.data)
    out_dir = Path(a.out_dir)
    views_dir = out_dir / "_rm_views"
    views_dir.mkdir(parents=True, exist_ok=True)

    harm_view = views_dir / "harmlessness_pairs.jsonl"
    help_view = views_dir / "helpfulness_pairs.jsonl"
    n_harm = _split_view(data_path, "rejected_toxic", harm_view)
    n_help = _split_view(data_path, "rejected_evasive", help_view)
    print(f"harmlessness view: {n_harm} pairs -> {harm_view}")
    print(f"helpfulness view:  {n_help} pairs -> {help_view}")

    for name, view_path in [("harmlessness", harm_view), ("helpfulness", help_view)]:
        rm_out = out_dir / f"rm_{name}"
        print(f"\n=== training {name} RM -> {rm_out} ===")
        cfg = TrainConfig(
            train_path=str(view_path),
            out_dir=str(rm_out),
            base_name=a.base,
            epochs=a.epochs,
            batch_size=a.batch_size,
            lr=a.lr,
            val_fraction=a.val_fraction,
        )
        train(cfg)

    print("\ndone. Run scripts/rm_redteam_gate.py against BOTH new RM dirs "
          "before trusting either for PPO.")


if __name__ == "__main__":
    main()
