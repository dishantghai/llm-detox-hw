#!/usr/bin/env python3
"""Append one run's numbers to attempt_3's running comparison table, then
print the whole table so far.

This is a bookkeeping tool, not a modeling tool: it does not train, eval,
or judge anything on your behalf. It exists because "record and look at
the baseline and all the different runs" is hard to do by eyeballing five
different JSON files, and easy to do if every run lands one row in one
CSV that you can re-print at any point.

Two ways to log a row:

1. From an eval_lib-style JSON (what tasks/task1_sft_eval.py,
   tasks/task3_dpo_eval.py, tasks/task6_ppo_detoxify_eval.py, and
   tasks/task7_ppo_rm_eval.py all write — a dict with "greedy" and
   "sampled" keys):

       python attempt_3/scripts/record_run.py \\
           --label sft --eval-json attempt_3/submissions/sft_eval.json \\
           --notes "uniqueness better than base but hedge template still top hit"

2. From free-form metrics, for stages that don't produce that JSON shape
   (RM pairwise accuracy, uniqueness rate, red-team gate verdict, ...):

       python attempt_3/scripts/record_run.py \\
           --label rm --metric pairwise_acc=0.94 --metric mean_margin=3.1 \\
           --notes "RM ranks held-out chosen above rejected 94% of the time"

Re-run with --show at any point to reprint the table without adding a row.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

FIELDS = [
    "timestamp", "label",
    "mild_prefix_greedy", "direct_provocation_greedy", "rtp_challenging_greedy",
    "mild_prefix_support", "direct_provocation_support", "rtp_challenging_support",
    "mild_prefix_mean_max", "direct_provocation_mean_max", "rtp_challenging_mean_max",
    "mild_prefix_mean_std", "direct_provocation_mean_std", "rtp_challenging_mean_std",
    "extra_metrics", "notes",
]

SLICES = ("mild_prefix", "direct_provocation", "rtp_challenging")


def _row_from_eval_json(label: str, path: Path, extra: dict, notes: str) -> dict:
    data = json.loads(path.read_text())
    greedy = data.get("greedy", {})
    sampled = data.get("sampled", {})
    row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "label": label}
    for s in SLICES:
        row[f"{s}_greedy"] = greedy.get(s, "")
        sv = sampled.get(s, {})
        row[f"{s}_support"] = sv.get("support_rate", "")
        row[f"{s}_mean_max"] = sv.get("mean_max", "")
        row[f"{s}_mean_std"] = sv.get("mean_std", "")
    row["extra_metrics"] = json.dumps(extra) if extra else ""
    row["notes"] = notes
    return row


def _row_from_metrics(label: str, extra: dict, notes: str) -> dict:
    row = {f: "" for f in FIELDS}
    row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row["label"] = label
    row["extra_metrics"] = json.dumps(extra) if extra else ""
    row["notes"] = notes
    return row


def _load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def _append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def _fmt(v) -> str:
    if v == "" or v is None:
        return "-"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("(no runs recorded yet)")
        return

    print(f"\n{'='*100}")
    print(f"attempt_3 — recorded runs ({len(rows)} so far)")
    print(f"{'='*100}")

    header = f"{'label':22s} {'mild_pfx':>9s} {'direct':>9s} {'rtp':>9s}   {'mild_supp':>9s} {'direct_supp':>11s} {'rtp_supp':>9s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['label']:22s} "
            f"{_fmt(r.get('mild_prefix_greedy')):>9s} "
            f"{_fmt(r.get('direct_provocation_greedy')):>9s} "
            f"{_fmt(r.get('rtp_challenging_greedy')):>9s}   "
            f"{_fmt(r.get('mild_prefix_support')):>9s} "
            f"{_fmt(r.get('direct_provocation_support')):>11s} "
            f"{_fmt(r.get('rtp_challenging_support')):>9s}"
        )
    print("(greedy cols = mean Detoxify, lower is better. "
          "*_support = K=16 support_rate, lower is better.)")

    ranked = [r for r in rows if r.get("rtp_challenging_greedy")]
    if len(ranked) > 1:
        ranked.sort(key=lambda r: float(r["rtp_challenging_greedy"]))
        print(f"\nranked by rtp_challenging greedy toxicity (lowest first):")
        for i, r in enumerate(ranked, 1):
            print(f"  {i}. {r['label']:22s} {_fmt(r['rtp_challenging_greedy'])}")

    print(f"\nnotes so far:")
    for r in rows:
        if r.get("notes"):
            print(f"  [{r['label']}] {r['notes']}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", help="short run name, e.g. baseline / sft / dpo / rm_ppo / custom_ppo / fixed_ppo_v2")
    p.add_argument("--eval-json", help="path to a tasks/task*_eval.py-style JSON output")
    p.add_argument("--metric", action="append", default=[], metavar="key=value",
                    help="free-form metric, repeatable, e.g. --metric pairwise_acc=0.94")
    p.add_argument("--notes", default="", help="one or two sentences: what did you see, what will you do next")
    p.add_argument("--csv", default="attempt_3/results/comparison.csv")
    p.add_argument("--show", action="store_true", help="just print the table, don't add a row")
    a = p.parse_args()

    csv_path = Path(a.csv)

    if not a.show:
        if not a.label:
            p.error("--label is required unless you pass --show")
        extra = {}
        for kv in a.metric:
            if "=" not in kv:
                p.error(f"--metric expects key=value, got {kv!r}")
            k, v = kv.split("=", 1)
            extra[k] = v

        if a.eval_json:
            row = _row_from_eval_json(a.label, Path(a.eval_json), extra, a.notes)
        else:
            row = _row_from_metrics(a.label, extra, a.notes)

        _append_row(csv_path, row)
        print(f"recorded '{a.label}' -> {csv_path}")

    _print_table(_load_rows(csv_path))


if __name__ == "__main__":
    main()
