"""Decode-time, script-targeted alternative to ``repetition_penalty`` for
the Stage 8 (attempt_3) non-Latin tail-degeneration failure.

``repetition_penalty`` was tested directly against the
``ppo_langgate_relevance_merged`` checkpoint (see
``attempt_3/scripts/measure_tail_degeneration.py`` and
``eval_lib.greedy_generate``'s docstring) and made the failure *worse*
(OOD tail-degen 31%->46%, fully-non-Latin 7%->38%): it discounts every
token already used in the response, including ordinary English function
words, so once a short completion's natural continuations are used up,
the tokens it hasn't discounted yet are disproportionately the non-Latin
ones -- it steers straight into the failure it was meant to prevent.

This module instead identifies, once per tokenizer, every vocabulary
token whose *decoded surface form* contains a non-Latin letter (Cyrillic,
CJK, etc., but not digits/punctuation/whitespace/Latin diacritics) and
returns a ``transformers.LogitsProcessor`` that sets those tokens' logits
to ``-inf`` at every generation step. This bans the failure mode by
script identity, independent of how many times any token has already
appeared -- it can't have the "penalize already-used English words"
side effect because it never looks at what's already been generated.
"""
from __future__ import annotations

import re

import torch
from transformers import LogitsProcessor

# A token is banned only if it contains at least one letter OUTSIDE these
# ranges (Latin + Latin-1/Extended, i.e. ASCII English plus common
# loanword diacritics). Tokens are matched for *presence* of a
# disallowed letter, not required to be *entirely* letters -- most BPE
# tokens mix letters with a leading space, punctuation, or digits (e.g.
# " the", "don't", "2023"), and those must stay allowed.
_DISALLOWED_LETTER_RE = re.compile(r"[^\W\d_A-Za-zÀ-ɏ]", re.UNICODE)

_CACHE: dict[int, torch.Tensor] = {}


def _non_latin_token_ids(tokenizer) -> list[int]:
    vocab_size = len(tokenizer)
    all_ids = list(range(vocab_size))
    # batch_decode is a pure Python/Rust detokenize, no model forward pass --
    # fast even over a ~150k vocab.
    strings = tokenizer.batch_decode([[i] for i in all_ids], skip_special_tokens=True)
    banned = [tok_id for tok_id, s in zip(all_ids, strings) if _DISALLOWED_LETTER_RE.search(s)]
    return banned


class NonLatinLogitsProcessor(LogitsProcessor):
    def __init__(self, banned_mask: torch.Tensor):
        self._banned_mask = banned_mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = self._banned_mask
        width = scores.size(-1)
        # The model's output/embedding dim is often padded past the
        # tokenizer's actual vocab (e.g. Qwen2.5: 151936 vs 151665) --
        # those extra ids are unused, never ban them, just size-match.
        if mask.numel() < width:
            mask = torch.cat([mask, torch.zeros(width - mask.numel(), dtype=torch.bool)])
        elif mask.numel() > width:
            mask = mask[:width]
        mask = mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        return scores


def build_non_latin_suppressor(tokenizer) -> list[LogitsProcessor]:
    """Cached per-vocab-size; safe to call on every eval invocation."""
    key = len(tokenizer)
    if key not in _CACHE:
        banned_ids = _non_latin_token_ids(tokenizer)
        mask = torch.zeros(key, dtype=torch.bool)
        mask[banned_ids] = True
        _CACHE[key] = mask
        print(f"[non_latin_logits_processor] banned {mask.sum().item()}/{key} vocab tokens")
    return [NonLatinLogitsProcessor(_CACHE[key])]
