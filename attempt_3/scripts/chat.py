"""Interactive REPL for eyeballing any merged checkpoint by hand.

Usage:
    python -m attempt_3.scripts.chat --model-dir attempt_3/checkpoints/ppo_dual_lagrangian_merged
    python -m attempt_3.scripts.chat --model-dir attempt_3/checkpoints/ppo_rm_merged --sample

Type a prompt and hit enter; empty line or Ctrl-D to exit.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--sample", action="store_true", help="sample instead of greedy decode")
    args = ap.parse_args()

    from src.detox_hw import eval_lib

    print(f"loading {args.model_dir} ...")
    model = eval_lib.load_merged_hf(Path(args.model_dir))
    print("ready. empty line or Ctrl-D to quit.\n")

    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            break
        if not prompt:
            break
        if args.sample:
            tok = eval_lib.get_tokenizer()
            text = eval_lib._chat_text(prompt)
            enc = tok(text, return_tensors="pt").to(eval_lib.DEVICE)
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=1.0, top_p=0.95, top_k=50,
                pad_token_id=tok.eos_token_id,
            )
            completion = tok.decode(out[0, enc["input_ids"].size(1):], skip_special_tokens=True)
        else:
            completion = eval_lib.greedy_generate(model, [prompt], max_new_tokens=args.max_new_tokens)[0]
        print(completion)
        print()


if __name__ == "__main__":
    main()
