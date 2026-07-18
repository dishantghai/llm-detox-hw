"""Phase 3 — three-way DPO: adds ``rejected_evasive`` pressure alongside the
original ``rejected_toxic``, using the ORIGINAL, unmodified
``src.detox_hw.train_dpo.train`` and ``tasks/task2_dpo_loss.py::dpo_loss``.

No new DPO math. ``dpo_loss`` is inherently pairwise (one chosen, one
rejected per row) — so the extension here is entirely at the data-adapter
level: each row of ``attempt_2/data/dpo_v2.jsonl``
(``{prompt, chosen, rejected_toxic, rejected_evasive}``) becomes TWO
``{prompt, chosen, rejected}`` training pairs sharing the same ``chosen``,
before handing off to the untouched trainer. This is a standard, minimal way
to fold a second negative class into a pairwise preference loss without
rederiving it (sometimes called "multi-negative DPO" in the literature) —
picked here specifically because it requires zero changes to
``dpo_loss``/the trainer's forward pass, which this project's own Task 2
work already validated end-to-end.

Usage (from repo root):
    python -m attempt_2.src.detox_hw.train_dpo_v2 \\
        --train attempt_2/data/dpo_v2.jsonl \\
        --sft-dir attempt_2/checkpoints/sft \\
        --out attempt_2/checkpoints/dpo
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.common.io import read_jsonl
from src.detox_hw.train_dpo import train


def expand_three_way(rows: list[dict]) -> list[dict]:
    """{prompt, chosen, rejected_toxic, rejected_evasive} -> two rows each:
    {prompt, chosen, rejected=toxic} and {prompt, chosen, rejected=evasive}.
    """
    out: list[dict] = []
    for r in rows:
        out.append({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected_toxic"]})
        out.append({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected_evasive"]})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="JSONL of {prompt, chosen, rejected_toxic, rejected_evasive} rows")
    p.add_argument("--sft-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=32)
    a = p.parse_args()

    rows = list(read_jsonl(a.train))
    pairs = expand_three_way(rows)
    print(f"{len(rows)} three-way rows -> {len(pairs)} DPO pairs "
          f"({len(rows)} vs rejected_toxic + {len(rows)} vs rejected_evasive)")

    train(
        pairs, Path(a.sft_dir), Path(a.out), base_name=a.base,
        beta=a.beta, lr=a.lr, batch_size=a.batch_size,
        grad_accum=a.grad_accum, epochs=a.epochs, lora_r=a.lora_r,
    )


if __name__ == "__main__":
    main()
