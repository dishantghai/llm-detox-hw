"""
===========================================================================
Build Detox-Direction Preference Pairs from Anthropic/hh-rlhf
===========================================================================

OVERALL PURPOSE:
    This script constructs preference pair datasets used for training a
    language model to produce safer (less toxic) responses. It leverages
    the Anthropic/hh-rlhf dataset (harmless-base split), which contains
    pairs of "chosen" (safer) and "rejected" (more harmful) assistant
    replies to the same human prompt.

PIPELINE OVERVIEW:
    1. Load the hh-rlhf harmless-base training split from Hugging Face.
    2. Parse each row to extract the prompt and the final assistant reply
       for both the chosen (benign) and rejected (toxic) sides.
    3. Score every response with the Detoxify toxicity classifier.
    4. Filter: keep only pairs where the toxic side exceeds a toxicity
       threshold AND the benign side stays below a safety threshold.
       This ensures the reward model receives clean, high-contrast signal.
    5. Write two output files:
       - dpo.jsonl  : {prompt, chosen, rejected} triples for DPO training.
       - sft.jsonl  : {prompt, response} rows (response = chosen) for SFT.

DATA FORMAT (hh-rlhf):
    chosen   = "\\n\\nHuman: <question>\\n\\nAssistant: <safe-reply>..."
    rejected = "\\n\\nHuman: <question>\\n\\nAssistant: <unsafe-reply>..."

    For the **detox direction** we preserve this polarity:
        chosen   = benign (safe reply)
        rejected = toxic  (harmful reply)
===========================================================================
"""

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------
from __future__ import annotations  # Enable modern type-hint syntax (PEP 604 unions, etc.)

import argparse  # Command-line argument parsing
import json      # JSON serialization for output files
from pathlib import Path  # Object-oriented filesystem paths

import torch                  # PyTorch — used here to check for GPU availability
from datasets import load_dataset  # Hugging Face datasets library to load hh-rlhf
from tqdm.auto import tqdm    # Progress bar utility for long-running loops


# --------------------------------------------------------------------------
# Helper: Split an hh-rlhf transcript into (prompt, response)
# --------------------------------------------------------------------------
def _split_prompt_response(text: str) -> tuple[str, str] | None:
    """Parse a single hh-rlhf conversational transcript.

    hh-rlhf rows are multi-turn conversations formatted as:
        \\n\\nHuman: ...\\n\\nAssistant: ...\\n\\nHuman: ...\\n\\nAssistant: ...

    The *prompt* is everything up to and including the final "Assistant:"
    marker. The *response* is everything after that marker.

    Returns None if parsing fails (missing marker or empty fields).
    """
    marker = "\n\nAssistant:"  # Delimiter that separates prompt from final response
    idx = text.rfind(marker)   # Find the LAST occurrence (final assistant turn)
    if idx < 0:
        # No "Assistant:" marker found — malformed row; skip it
        return None
    # Prompt = everything up to and including the final "Assistant:" tag
    prompt = text[: idx + len(marker)].strip()
    # Response = everything after the final "Assistant:" tag
    response = text[idx + len(marker):].strip()
    if not prompt or not response:
        # Either side is empty after stripping — unusable; skip it
        return None
    return prompt, response  # Return the parsed (prompt, response) tuple


# --------------------------------------------------------------------------
# Helper: Score a list of texts for toxicity using the Detoxify model
# --------------------------------------------------------------------------
def _detoxify_scores(texts: list[str], batch_size: int = 32) -> list[float]:
    """Run the Detoxify 'original' model on a list of texts and return
    the toxicity score (float in [0, 1]) for each text.

    Processes texts in mini-batches to avoid OOM on large inputs.
    Uses GPU if available, otherwise falls back to CPU.
    """
    from detoxify import Detoxify  # Lazy import to avoid loading model unless needed

    # Select compute device: prefer GPU for speed, fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Instantiate the Detoxify "original" model on the chosen device
    dx = Detoxify("original", device=device)

    out: list[float] = []  # Accumulator for all toxicity scores
    # Iterate over texts in batches, showing a progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="detoxify"):
        chunk = texts[i : i + batch_size]  # Slice current batch of texts
        s = dx.predict(chunk)["toxicity"]  # Get toxicity predictions for the batch
        if isinstance(s, (list, tuple)):
            # Batch prediction returned a list — extend output with all scores
            out.extend(float(x) for x in s)
        else:
            # Single-element batch returned a scalar — append directly
            out.append(float(s))
    return out  # Return the complete list of toxicity scores


# --------------------------------------------------------------------------
# Main pipeline: parse arguments, load data, score, filter, and write output
# --------------------------------------------------------------------------
def main() -> None:
    """Entry point: orchestrates the full data preparation pipeline."""

    # --- Argument parsing -------------------------------------------------
    p = argparse.ArgumentParser()  # Create CLI argument parser
    # Output directory for the generated JSONL files
    p.add_argument("--out-dir", default="data")
    # Maximum number of rows to process from hh-rlhf (-1 = unlimited)
    p.add_argument("--max-rows", type=int, default=80000,
                   help="cap the hh-rlhf rows to score; -1 = no cap")
    # Minimum toxicity score for the rejected (toxic) response to be kept
    p.add_argument("--toxic-threshold", type=float, default=0.5,
                   help="rejected (toxic) side must score >= this")
    # Maximum toxicity score for the chosen (benign) response to be kept
    p.add_argument("--benign-threshold", type=float, default=0.10,
                   help="chosen (benign) side must score <= this")
    # Random seed for reproducible dataset shuffling
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()  # Parse command-line arguments into namespace `a`

    # --- Setup output paths -----------------------------------------------
    out_dir = Path(a.out_dir)             # Convert output dir string to Path object
    out_dir.mkdir(parents=True, exist_ok=True)  # Create output dir (and parents) if missing
    dpo_path = out_dir / "dpo.jsonl"      # Path for DPO preference-pair output
    sft_path = out_dir / "sft.jsonl"      # Path for SFT (supervised fine-tuning) output

    # --- Load dataset -----------------------------------------------------
    print("loading Anthropic/hh-rlhf (harmless-base train split)…")
    # Load the harmless-base training split from the Anthropic/hh-rlhf dataset
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
    if a.max_rows > 0:
        # If a row cap is set, shuffle and select up to max_rows entries
        ds = ds.shuffle(seed=a.seed).select(range(min(a.max_rows, len(ds))))

    # --- Parse rows into (prompt, benign_response, toxic_response) triples -
    parsed: list[tuple[str, str, str]] = []  # Accumulator: (prompt, benign, toxic)
    for row in tqdm(ds, desc="parse"):
        # Parse the "chosen" (safer) side into prompt and response
        c = _split_prompt_response(row["chosen"])
        # Parse the "rejected" (harmful) side into prompt and response
        r = _split_prompt_response(row["rejected"])
        if c is None or r is None or c[0] != r[0]:
            # Skip if either side failed to parse, or if prompts don't match
            continue
        prompt = c[0]       # The shared prompt (identical for chosen & rejected)
        benign = c[1]       # The safer assistant reply ('chosen' in hh-rlhf)
        toxic  = r[1]       # The harmful assistant reply ('rejected' in hh-rlhf)
        parsed.append((prompt, benign, toxic))  # Store the triple
    print(f"parsed {len(parsed)} candidate triples")

    # --- Score both sides with Detoxify -----------------------------------
    print("scoring benign side…")
    # Compute toxicity scores for all benign (chosen) responses
    benign_scores = _detoxify_scores([b for _, b, _ in parsed])
    print("scoring toxic side…")
    # Compute toxicity scores for all toxic (rejected) responses
    toxic_scores  = _detoxify_scores([t for _, _, t in parsed])

    # --- Filter pairs by toxicity thresholds ------------------------------
    kept: list[dict] = []  # Accumulator for pairs that pass filtering
    for (prompt, benign, toxic), bs, ts in zip(parsed, benign_scores, toxic_scores):
        # Keep only pairs where toxic side is clearly toxic AND benign side is clearly safe
        if ts >= a.toxic_threshold and bs <= a.benign_threshold:
            kept.append({
                "prompt": prompt,          # The shared human prompt
                "chosen": benign,          # The safe (benign) response
                "rejected": toxic,         # The harmful (toxic) response
                "chosen_tox": bs,          # Toxicity score of the chosen response
                "rejected_tox": ts,        # Toxicity score of the rejected response
            })
    print(f"kept {len(kept)} pairs after filtering "
          f"(toxic ≥ {a.toxic_threshold}, benign ≤ {a.benign_threshold})")

    # --- Write output JSONL files -----------------------------------------
    # Write DPO file: each line is a JSON object with prompt, chosen, rejected, and scores
    with open(dpo_path, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")  # One JSON object per line
    # Write SFT file: each line has prompt and response (response = chosen/benign reply)
    with open(sft_path, "w") as f:
        for row in kept:
            f.write(json.dumps({"prompt": row["prompt"], "response": row["chosen"]}) + "\n")
    print(f"wrote {dpo_path}\nwrote {sft_path}")  # Confirm output paths to user


# --------------------------------------------------------------------------
# Script entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Execute main() only when this file is run directly (not imported)
    main()
