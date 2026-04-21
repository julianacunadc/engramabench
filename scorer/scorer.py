#!/usr/bin/env python3
"""
EngramaBench Scorer (v1.4.0)

Supports all 7 answer types:
  - entity:     exact match with alias table (binary 0/1)
  - date:       exact or fuzzy date match ±7 days (binary 0/1)
  - number:     numeric match with tolerance ±5% (binary 0/1)
  - set:        set F1 (precision + recall over items, string or list expected)
  - short_span: token-level F1 against reference span
  - abstain:    binary — system should refuse to answer
  - insight:    placeholder rubric (full GPT-4o judge deferred to Phase 4)

Usage:
  python3 scripts/engramabench/scorer.py \
    --queries datasets/engramabench/full_v1/queries.json \
    --predictions <predictions.json> \
    --registry datasets/engramabench/full_v1/entity_registry.json \
    --output <report.json>

  # Dry-run with perfect predictions (validates scoring feasibility):
  python3 scripts/engramabench/scorer.py \
    --queries datasets/engramabench/pilot_v1/queries.json \
    --registry datasets/engramabench/pilot_v1/entity_registry.json \
    --dry-run
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Alias table
# ---------------------------------------------------------------------------

def build_alias_table(registry: dict) -> dict[str, str]:
    """Map every alias (lowercased) → canonical entity_id."""
    table: dict[str, str] = {}
    for entity in registry["entities"]:
        eid = entity["entity_id"]
        table[entity["canonical_name"].lower()] = eid
        for alias in entity.get("aliases", []):
            table[alias.lower()] = eid
    return table


def _strip_framing(text: str) -> str:
    """Remove conversational framing from an answer.

    Handles patterns like:
      'It was Kaszek Ventures'
      'The answer is MRR'
      'It uses React, Python, and PostgreSQL.'
      'That would be FinFlow.'
    """
    text = text.strip()
    # Strip trailing punctuation (period, comma, exclamation, colon, semicolon)
    text = re.sub(r"[.,!;:]+$", "", text).strip()
    # Remove common answer framing prefixes
    framing = re.compile(
        r"^(?:it\s+(?:was|is|uses?|would\s+be)|"
        r"the\s+answer\s+is|"
        r"that\s+(?:was|is|would\s+be)|"
        r"they\s+(?:are|were|use)|"
        r"the\s+\w+(?:\s+\w+)?\s+(?:are|were|is|include|includes?)|"
        r"(?:i\s+believe\s+)?(?:it'?s|that'?s))\s+",
        re.IGNORECASE,
    )
    text = framing.sub("", text).strip()
    # Strip trailing punctuation again after framing removal
    text = re.sub(r"[.,!;:]+$", "", text).strip()
    return text


def normalize_entity(text: str, alias_table: dict[str, str]) -> str:
    """Normalize an entity mention through the alias table.

    Strips framing, punctuation, and does case-insensitive alias lookup.
    """
    cleaned = _strip_framing(text)
    key = cleaned.lower()
    if key in alias_table:
        return alias_table[key]
    # Also try without any remaining articles
    no_article = re.sub(r"^(?:the|a|an)\s+", "", key)
    if no_article in alias_table:
        return alias_table[no_article]
    return key


# ---------------------------------------------------------------------------
# Answer-type scorers
# ---------------------------------------------------------------------------

def _is_negated_mention(text_lower: str, match_start: int, match_end: int) -> bool:
    """Check if a substring match is negated/contrasted in context.

    Looks for negation cues in the ~40 chars before the match:
      - "not X", "wasn't X", "isn't X", "was not X", "is not X"
      - "never X", "no X", "nor X"
      - "rather than X", "instead of X"
    Also checks the containing sentence for contrast patterns that indicate the
    entity was mentioned but the answer pivots to something else:
      - "X was considered but the real partner is Y"
      - "X however ..."
    """
    # Window before the match (up to 40 chars)
    prefix_start = max(0, match_start - 40)
    prefix = text_lower[prefix_start:match_start].strip()

    # Pre-match negation patterns
    _NEG_BEFORE = re.compile(
        r"(?:not|n't|never|no|nor|neither|rather\s+than|instead\s+of)\s*$",
        re.IGNORECASE,
    )
    if _NEG_BEFORE.search(prefix):
        return True

    # Sentence-level contrast: find the sentence containing the match and check
    # for contrast conjunctions (but, however, instead, rather) AFTER the entity
    # that signal a pivot away from it.
    # Find sentence boundaries around the match
    sent_start = text_lower.rfind(".", 0, match_start)
    sent_start = sent_start + 1 if sent_start >= 0 else 0
    sent_end = text_lower.find(".", match_end)
    sent_end = sent_end if sent_end >= 0 else len(text_lower)
    after_entity = text_lower[match_end:sent_end]

    _CONTRAST_IN_SENTENCE = re.compile(
        r"\b(?:but|however|instead|rather)\b",
        re.IGNORECASE,
    )
    if _CONTRAST_IN_SENTENCE.search(after_entity):
        return True

    return False


def score_entity(expected: str, predicted: str, alias_table: dict[str, str]) -> float:
    """Entity match with alias normalization. Binary 0/1.

    Strategy:
      1. Try exact match after framing strip (original behavior).
      2. If that fails, try word-boundary substring extraction: check if ANY
         known alias of the expected entity appears as a whole phrase inside the
         predicted answer, AND is not negated/contrasted in its surrounding
         context. This handles cases where the system wraps the entity in a full
         sentence like "Kaszek Ventures requires the 20% month-over-month
         growth..." while rejecting "It was not Bancolombia, it was Rappi."
    """
    e_norm = normalize_entity(expected, alias_table)
    p_norm = normalize_entity(predicted, alias_table)
    if e_norm == p_norm:
        return 1.0

    # Fallback: scan predicted text for any alias of the expected entity
    expected_id = e_norm  # already resolved through alias table
    target_aliases = [alias for alias, eid in alias_table.items() if eid == expected_id]
    if not target_aliases:
        target_aliases = [expected.strip().lower()]

    p_lower = predicted.strip().lower()
    for alias in target_aliases:
        # Word-boundary match to avoid partial matches inside other words
        pattern = re.compile(r"\b" + re.escape(alias) + r"\b")
        m = pattern.search(p_lower)
        if m and not _is_negated_mention(p_lower, m.start(), m.end()):
            return 1.0

    return 0.0


def _parse_number(text: str) -> float | None:
    """Extract a numeric value from text like '$45K', '4.2%', '$2.3 billion', '3,200'."""
    text = text.strip().lower()

    # Multiplier suffixes
    multipliers = {
        "k": 1_000, "thousand": 1_000,
        "m": 1_000_000, "million": 1_000_000,
        "b": 1_000_000_000, "billion": 1_000_000_000,
    }

    # Remove currency symbols and common prefixes
    text = re.sub(r"[$€£]", "", text)
    text = text.replace(",", "")

    # Try: <number> <suffix>
    m = re.search(r"([\d.]+)\s*(%|" + "|".join(multipliers.keys()) + r")", text)
    if m:
        num = float(m.group(1))
        suffix = m.group(2)
        if suffix == "%":
            return num  # keep as percentage value
        if suffix in multipliers:
            return num * multipliers[suffix]
        return num

    # Try: plain number (must contain at least one digit)
    m = re.search(r"\d[\d.]*", text)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None

    return None


def score_number(expected: str, predicted: str) -> float:
    """Numeric match with ±5% tolerance. Binary 0/1."""
    e_val = _parse_number(expected)
    p_val = _parse_number(predicted)
    if e_val is None or p_val is None:
        return 0.0
    if e_val == 0:
        return 1.0 if p_val == 0 else 0.0
    return 1.0 if abs(e_val - p_val) / abs(e_val) <= 0.05 else 0.0


def _parse_date(text: str) -> datetime | None:
    """Parse various date formats. Returns datetime or None."""
    text = text.strip()

    # Quarter format: Q1 2026, Q2 2026, etc.
    m = re.search(r"Q(\d)\s*(\d{4})", text)
    if m:
        quarter = int(m.group(1))
        year = int(m.group(2))
        month = (quarter - 1) * 3 + 1
        return datetime(year, month, 1)

    # Written quarter format: "first quarter of 2026", "second quarter of 2026", etc.
    _ORDINAL_TO_Q = {
        "first": 1, "1st": 1,
        "second": 2, "2nd": 2,
        "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4,
    }
    m = re.search(
        r"(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+of\s+(\d{4})",
        text, re.IGNORECASE,
    )
    if m:
        quarter = _ORDINAL_TO_Q[m.group(1).lower()]
        year = int(m.group(2))
        month = (quarter - 1) * 3 + 1
        return datetime(year, month, 1)

    # Try common formats
    for fmt in ["%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y",
                "%B %Y", "%b %Y", "%Y-%m", "%m/%d/%Y"]:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    # Year-month like "March 2026", "October 2025"
    m = re.match(r"(?:~?\s*)(\w+)\s+(\d{4})", text)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%B %Y")
        except ValueError:
            pass

    return None


def score_date(expected: str, predicted: str) -> float:
    """Exact or fuzzy date match ±7 days. Binary 0/1."""
    e_date = _parse_date(expected)
    p_date = _parse_date(predicted)
    if e_date is None or p_date is None:
        # Fallback: string containment for quarter/month references
        e_clean = expected.strip().lower()
        p_clean = predicted.strip().lower()
        return 1.0 if e_clean in p_clean or p_clean in e_clean else 0.0
    return 1.0 if abs((e_date - p_date).days) <= 7 else 0.0


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"\w+", text.lower())


def score_short_span(expected: str, predicted: str) -> float:
    """Token-level F1 against reference span."""
    e_tokens = set(_tokenize(expected))
    p_tokens = set(_tokenize(predicted))
    if not e_tokens or not p_tokens:
        return 0.0
    overlap = e_tokens & p_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(p_tokens)
    recall = len(overlap) / len(e_tokens)
    return 2 * precision * recall / (precision + recall)


def _parse_set_items(value: str | list) -> list[str]:
    """Parse a set answer from either a list or a comma/and-separated string."""
    if isinstance(value, list):
        return value
    # Strip overall framing first (e.g., "It uses React, Python, and PostgreSQL.")
    cleaned = _strip_framing(str(value))
    # Split on commas, "and", semicolons
    items = re.split(r"[,;]|\band\b", cleaned)
    return [x.strip() for x in items if x.strip()]


def _scan_all_entities_in_text(text: str, alias_table: dict[str, str]) -> set[str]:
    """Find ALL known entities mentioned in text using word-boundary matching.

    Scans the alias table for any surface form that appears in the text,
    using word boundaries to avoid substring false positives.
    Returns normalized entity IDs.
    """
    t_lower = text.lower()
    found = set()
    for surface, eid in alias_table.items():
        # Use word boundaries to avoid matching "An" inside "Anika"
        if len(surface) < 2:
            continue  # Skip single-char aliases
        pattern = r"\b" + re.escape(surface) + r"\b"
        if re.search(pattern, t_lower):
            found.add(eid)
    return found


def score_set(expected: list | str, predicted: str | list, alias_table: dict[str, str]) -> float:
    """Set F1 (precision + recall over items).

    Both expected and predicted can be either a list or a comma-separated string.
    Uses two strategies:
      1. Naive split (works when predicted is a clean list)
      2. Entity scanning (works when predicted is prose containing entity mentions)
    Takes the better score of the two.
    """
    e_items = _parse_set_items(expected)
    e_normalized = {normalize_entity(str(x), alias_table) for x in e_items}

    # Debug: empty sets mean something went wrong in parsing
    if not e_normalized:
        print(f"  WARNING: score_set got empty expected set from: {expected!r}")
        return 0.0

    # Strategy 1: Naive split
    p_items_naive = _parse_set_items(predicted)
    p_normalized_naive = {normalize_entity(str(x), alias_table) for x in p_items_naive}

    score_naive = 0.0
    if p_normalized_naive:
        overlap_naive = e_normalized & p_normalized_naive
        if overlap_naive:
            prec = len(overlap_naive) / len(p_normalized_naive)
            rec = len(overlap_naive) / len(e_normalized)
            score_naive = 2 * prec * rec / (prec + rec)

    # Strategy 2: Entity scanning — find ALL known entities in the prose,
    # not just expected ones. This correctly penalizes wrong extra items.
    p_found_scan = _scan_all_entities_in_text(str(predicted), alias_table)

    score_scan = 0.0
    if p_found_scan:
        overlap_scan = e_normalized & p_found_scan
        if overlap_scan:
            prec = len(overlap_scan) / len(p_found_scan)
            rec = len(overlap_scan) / len(e_normalized)
            score_scan = 2 * prec * rec / (prec + rec)

    return max(score_naive, score_scan)


# Abstention detection patterns
_ABSTAIN_PATTERNS = [
    r"(?:i\s+)?(?:don'?t|do\s+not|cannot|can'?t)\s+(?:have|find|see|recall|know|tell|determine|confirm)",
    r"(?:no|not)\s+(?:information|mention|evidence|data|record)",
    r"not\s+(?:discussed|mentioned|stated|provided|available|present)",
    r"there\s+is\s+no\s+(?:information|mention|evidence|record)",
    r"(?:isn'?t|is\s+not)\s+(?:mentioned|discussed|present|available)",
    r"never\s+(?:discussed|mentioned|came\s+up)",
    r"no\s+(?:such|relevant)\s+(?:conversation|discussion|mention)",
    r"abstain",
    r"(?:i\s+)?(?:can'?t|cannot)\s+answer",
    r"not\s+enough\s+information",
    r"does\s+not\s+(?:contain|provide|include|mention|have|show)",
    r"(?:you\s+)?haven'?t\s+(?:mentioned|discussed|shared|talked\s+about|brought\s+up)",
]


# Patterns that indicate the model is hedging with a factual guess after abstaining
_HEDGE_PATTERNS = [
    r"(?:but|however|though|although|that\s+said)\s+(?:i\s+think|maybe|perhaps|it\s+(?:might|could|may)\s+be|possibly)",
    r"(?:but|however|though)\s+.*\d",  # abstain + then a number = hedged guess
    r"(?:but|however|though)\s+.*(?:was|is|were|had|did)",  # abstain + factual verb
    r"(?:my\s+guess|i\s+(?:would\s+)?guess|i\s+think|i\s+believe)\s+",
    r"(?:it\s+(?:was|is)\s+(?:probably|likely|possibly))",
    r"if\s+that\s+happened",  # speculative follow-up after abstention
]


def score_abstain(predicted: str) -> float:
    """Binary: 1.0 if system cleanly abstains, 0.0 otherwise.

    Requires:
      1. A valid abstention pattern is present
      2. No subsequent factual guess / hedged assertion
      3. No factual assertions after the abstention sentence
    """
    p_lower = predicted.strip().lower()

    # Step 1: Check for abstention signal
    has_abstention = False
    abstention_end = 0  # character position where the abstention pattern ends
    for pattern in _ABSTAIN_PATTERNS:
        m = re.search(pattern, p_lower)
        if m:
            has_abstention = True
            abstention_end = m.end()
            break

    if not has_abstention:
        return 0.0

    # Step 2: Check for hedged hallucination after the abstention
    for pattern in _HEDGE_PATTERNS:
        if re.search(pattern, p_lower):
            return 0.0

    # Step 3: Check for non-trivial content after the abstention sentence.
    # Any post-abstention clause that isn't itself another abstention → reject.
    remainder = p_lower[abstention_end:].strip()
    # Strip the rest of the abstention sentence (up to first sentence boundary)
    first_boundary = re.search(r"[.!?]\s+", remainder)
    if first_boundary:
        post_abstention = remainder[first_boundary.end():].strip()
    else:
        # No further sentences — clean abstention
        return 1.0

    if not post_abstention:
        return 1.0

    # Split post-abstention into clauses (sentence boundaries)
    clauses = re.split(r"[.!?]+\s*", post_abstention)
    clauses = [c.strip() for c in clauses if c.strip()]

    _STOPWORDS = {
        "i", "me", "my", "the", "a", "an", "is", "are", "was", "were", "it",
        "that", "this", "to", "of", "in", "on", "for", "and", "or", "but",
        "not", "no", "so", "if", "as", "at", "by", "from", "with", "about",
        "do", "does", "did", "be", "been", "being", "have", "has", "had",
        "you", "your", "we", "us", "our", "any", "how", "what", "when",
        "where", "which", "who", "there", "here", "can", "could", "would",
        "should", "will", "may", "might",
    }

    # Conversational closers: polite sign-offs that don't contain factual claims.
    # e.g., "feel free to share", "if you have questions", "let me know"
    _CLOSER_PATTERNS = [
        r"feel\s+free\s+to",
        r"(?:if|when)\s+you\s+(?:have|want|need|would\s+like|'d\s+like)",
        r"(?:would|do)\s+you\s+(?:like|want|need)\s+to",
        r"let\s+me\s+know",
        r"(?:happy|glad)\s+to\s+(?:help|assist)",
        r"(?:don'?t\s+)?hesitate\s+to",
        r"(?:i'?m|i\s+am)\s+here\s+(?:to\s+help|if)",
        r"how\s+(?:are|do)\s+you\s+(?:planning|want)",
        r"if\s+there'?s\s+anything",
        # Metadiscursive disclaimers about memory/context limitations (not factual claims)
        r"(?:it\s+)?(?:might|may|could)\s+not\s+be\s+in\s+(?:the\s+)?(?:memory|context)",
        r"if\s+this\s+is\s+(?:something\s+)?new",
        r"(?:it\s+)?(?:might|may|could)\s+not\s+(?:have\s+)?(?:been\s+)?(?:captured|recorded|stored|saved)",
        r"(?:this\s+)?(?:wasn'?t|was\s+not|isn'?t|is\s+not)\s+(?:captured|recorded|stored|mentioned)\s+in",
        r"(?:i\s+)?(?:don'?t|do\s+not)\s+(?:have\s+)?(?:any\s+)?(?:record|memory|context)\s+(?:of|about|for)",
    ]

    for clause in clauses:
        # Check if this clause is itself another abstention → allowed
        is_abstention = any(re.search(p, clause) for p in _ABSTAIN_PATTERNS)
        if is_abstention:
            continue

        # Check if this clause is a conversational closer → allowed
        is_closer = any(re.search(p, clause) for p in _CLOSER_PATTERNS)
        if is_closer:
            continue

        # Check if the clause has any non-stopword content
        tokens = re.findall(r"\w+", clause)
        non_stop = [t for t in tokens if t not in _STOPWORDS]
        if non_stop:
            # Non-trivial, non-abstention, non-closer clause → factual leak
            return 0.0

    return 1.0


def score_insight(expected: str, predicted: str) -> float:
    """Placeholder rubric scorer. Uses token-level F1 as proxy.
    Full GPT-4o judge implementation deferred to Phase 4.
    """
    # Phase 1E stub: use short_span F1 as rough proxy
    # In Phase 4 this becomes:
    #   correctness (0.4) + specificity (0.3) + cross-grounding (0.3)
    #   evaluated by GPT-4o judge with frozen prompt
    return score_short_span(expected, predicted)


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

SCORERS = {
    "entity": lambda e, p, at: score_entity(e, p, at),
    "date": lambda e, p, _: score_date(e, p),
    "number": lambda e, p, _: score_number(e, p),
    "set": lambda e, p, at: score_set(e, p, at),
    "short_span": lambda e, p, _: score_short_span(e, p),
    "abstain": lambda _, p, __: score_abstain(p),
    "insight": lambda e, p, _: score_insight(e, p),
}


def score_query(query: dict, prediction: dict, alias_table: dict[str, str]) -> dict:
    """Score a single query-prediction pair. Returns per-query result."""
    answer_type = query["answer_type"]
    expected = query["expected_answer"]
    predicted = prediction.get("system_answer", "")

    scorer = SCORERS.get(answer_type)
    if scorer is None:
        raise ValueError(f"Unknown answer type: {answer_type}")

    score = scorer(expected, predicted, alias_table)

    # Policy: for adversarial abstain queries, correctly contradicting a false
    # premise is as good as abstaining. If the query defines
    # accept_contradiction_patterns and any pattern matches, score 1.0 —
    # UNLESS reject_contradiction_patterns also match (mixed/wrong answers
    # that affirm the false premise alongside the contradiction).
    #
    # Step 0 Fix-1 (v1.3.1): when the prediction LEADS with a clear abstention
    # (first sentence matches _ABSTAIN_PATTERNS) AND matches an accept pattern,
    # the abstention takes priority over reject patterns. This prevents reject
    # regexes from penalizing subordinate clauses inside abstentions, e.g.,
    # "I don't have info on when Carlos closed the Series A" where "closed the"
    # fires a reject regex designed for affirmative false claims.
    _HAS_ACCEPT = (
        "accept_contradiction_patterns" in query
        or "accept_contradiction_regex" in query
    )
    if answer_type == "abstain" and score < 1.0 and _HAS_ACCEPT:
        p_lower = predicted.strip().lower()

        # Step A — is there a clear abstention at the START of the response?
        first_sentence_end = re.search(r"[.!?]", p_lower)
        leading_clause = p_lower[: first_sentence_end.start()] if first_sentence_end else p_lower
        has_leading_abstention = any(
            re.search(pat, leading_clause) for pat in _ABSTAIN_PATTERNS
        )

        # Step B — does the prediction match any accept pattern/regex?
        accepted = any(
            pat.lower() in p_lower
            for pat in query.get("accept_contradiction_patterns", [])
        )
        if not accepted:
            accepted = any(
                re.search(pat, p_lower)
                for pat in query.get("accept_contradiction_regex", [])
            )

        # Step C — does the prediction match any reject pattern/regex?
        rejected = any(
            pat.lower() in p_lower
            for pat in query.get("reject_contradiction_patterns", [])
        )
        if not rejected:
            rejected = any(
                re.search(pat, p_lower)
                for pat in query.get("reject_contradiction_regex", [])
            )

        # Step D — decision:
        # (1) Leading abstention + accept → abstention wins, even if reject
        #     also matches (the reject is inside the abstention clause).
        # (2) Not rejected + accept → original path (clean contradiction).
        if has_leading_abstention and accepted:
            score = 1.0
        elif not rejected and accepted:
            score = 1.0

    return {
        "query_id": query["query_id"],
        "task_family": query["task_family"],
        "answer_type": answer_type,
        "depth_band": query["depth_band"],
        "score": round(score, 4),
        "expected": expected,
        "predicted": predicted,
    }


def aggregate_scores(per_query: list[dict]) -> dict:
    """Compute per-task-family and per-depth-band aggregates."""
    # Per task family
    by_family: dict[str, list[float]] = defaultdict(list)
    for r in per_query:
        by_family[r["task_family"]].append(r["score"])

    per_task_family = {}
    for family, scores in by_family.items():
        per_task_family[family] = {
            "mean_score": round(sum(scores) / len(scores), 4),
            "count": len(scores),
        }

    # Per depth band
    by_depth: dict[str, list[float]] = defaultdict(list)
    for r in per_query:
        band = r["depth_band"]
        if band is not None:
            by_depth[band].append(r["score"])

    per_depth_band = {}
    for band, scores in by_depth.items():
        per_depth_band[band] = {
            "mean_score": round(sum(scores) / len(scores), 4),
            "count": len(scores),
        }

    # Top-level scores (per scope §6.5)
    def _family_mean(family: str) -> float:
        s = by_family.get(family, [])
        return round(sum(s) / len(s), 4) if s else 0.0

    scores = {
        "single_space_f1": _family_mean("single_space"),
        "cross_space_f1": _family_mean("cross_space"),
        "temporal_cross_space_f1": _family_mean("temporal_cross_space"),
        "adversarial_accuracy": _family_mean("adversarial"),
        "emergent_insight_score": _family_mean("emergent_insight"),
    }
    # Composite: weighted average (factual families get equal weight, adversarial + insight separate)
    factual_families = ["single_space", "cross_space", "temporal_cross_space"]
    factual_scores = [scores[f"{f}_f1"] for f in factual_families]
    factual_avg = sum(factual_scores) / len(factual_scores) if factual_scores else 0.0
    scores["composite_score"] = round(
        factual_avg * 0.5
        + scores["adversarial_accuracy"] * 0.25
        + scores["emergent_insight_score"] * 0.25,
        4,
    )

    return {
        "scores": scores,
        "per_task_family": per_task_family,
        "per_depth_band": per_depth_band,
    }


def build_report(
    per_query: list[dict],
    aggregates: dict,
    system: str = "dry_run",
    prediction_metadata: dict | None = None,
) -> dict:
    """Build report.json structure per scope §6.5.

    prediction_metadata: optional dict from prediction.json with system_model,
    system_config_version, system_config_hash, baseline_truncation, efficiency, etc.
    If None (dry-run), fields are populated with explicit stub defaults.
    """
    meta = prediction_metadata or {}

    # Compute depth_decay from per_depth_band
    depth_band = aggregates.get("per_depth_band", {})
    recent_f1 = depth_band.get("recent", {}).get("mean_score", 0.0)
    deep_f1 = depth_band.get("deep", {}).get("mean_score", 0.0)
    midrange_f1 = depth_band.get("mid", {}).get("mean_score", 0.0)
    cross_temporal_f1 = depth_band.get("cross_temporal", {}).get("mean_score", 0.0)
    # alpha = slope of decay from recent to deep (negative = decaying)
    alpha = round(deep_f1 - recent_f1, 4) if recent_f1 > 0 or deep_f1 > 0 else 0.0

    return {
        "benchmark_version": meta.get("benchmark_version", "full-v1"),
        "dataset_version": meta.get("dataset_version", "full-v1"),
        "scorer_version": "1.4.0",
        "run_id": meta.get("run_id", f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"),
        "system": system,
        "system_model": meta.get("system_model", "stub-default"),
        "system_config_version": meta.get("system_config_version", "stub-default"),
        "system_config_hash": meta.get("system_config_hash", "sha256:0000000000000000"),
        "generator_model": meta.get("generator_model", "gpt-4o"),
        "judge_model": meta.get("judge_model", "gpt-4o"),
        "seed": meta.get("seed", 42),
        "corpus_prompt_hash": meta.get("corpus_prompt_hash", "sha256:0000000000000000"),
        "query_prompt_hash": meta.get("query_prompt_hash", "sha256:0000000000000000"),
        "judge_prompt_hash": meta.get("judge_prompt_hash", "sha256:0000000000000000"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "scores": aggregates["scores"],
        "depth_decay": {
            "alpha": alpha,
            "recent_f1": recent_f1,
            "midrange_f1": midrange_f1,
            "deep_f1": deep_f1,
            "cross_temporal_f1": cross_temporal_f1,
        },
        "baseline_truncation": meta.get("baseline_truncation", {
            "truncated": False,
            "total_conversations": 0,
            "included_conversations": 0,
            "dropped_conversation_ids": [],
            "prompt_tokens": 0,
        }),
        "efficiency": meta.get("efficiency", {
            "avg_tokens_per_query": 0,
            "avg_cost_per_query": 0.0,
            "avg_latency_ms": 0,
            "total_cost": 0.0,
        }),
        "per_query": per_query,
        "per_task_family": aggregates["per_task_family"],
        "per_depth_band": aggregates["per_depth_band"],
    }


# ---------------------------------------------------------------------------
# Dry-run: generate perfect predictions to validate scoring
# ---------------------------------------------------------------------------

def generate_perfect_predictions(queries: list[dict]) -> list[dict]:
    """Generate predictions that should score 1.0 on every query."""
    predictions = []
    for q in queries:
        answer_type = q["answer_type"]
        expected = q["expected_answer"]

        if answer_type == "abstain":
            system_answer = "I don't have information about that in the conversations."
        elif answer_type == "set":
            if isinstance(expected, list):
                system_answer = ", ".join(str(x) for x in expected)
            else:
                system_answer = str(expected)
        else:
            system_answer = str(expected)

        predictions.append({
            "query_id": q["query_id"],
            "system_answer": system_answer,
        })
    return predictions


def generate_wrong_predictions(queries: list[dict]) -> list[dict]:
    """Generate predictions that should score 0.0 on every query."""
    predictions = []
    for q in queries:
        answer_type = q["answer_type"]

        if answer_type == "abstain":
            # Wrongly answers with a factual claim
            system_answer = "Carlos discussed his PhD thesis defense on March 15th."
        else:
            system_answer = "I have no idea about any of this."

        predictions.append({
            "query_id": q["query_id"],
            "system_answer": system_answer,
        })
    return predictions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EngramaBench Scorer")
    parser.add_argument("--queries", required=True, help="Path to queries.json")
    parser.add_argument("--predictions", help="Path to predictions.json")
    parser.add_argument("--registry", required=True, help="Path to entity_registry.json")
    parser.add_argument("--output", help="Path to write report.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with perfect + wrong predictions to validate scoring")
    parser.add_argument("--system", default="engrama", help="System name for report")
    args = parser.parse_args()

    with open(args.queries) as f:
        queries = json.load(f)
    with open(args.registry) as f:
        registry = json.load(f)

    alias_table = build_alias_table(registry)

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN: Validating scorer with perfect predictions")
        print("=" * 60)

        perfect_preds = generate_perfect_predictions(queries)
        pred_map = {p["query_id"]: p for p in perfect_preds}
        per_query = [score_query(q, pred_map[q["query_id"]], alias_table) for q in queries]
        aggregates = aggregate_scores(per_query)

        print(f"\nPerfect predictions — expected all 1.0:")
        for r in per_query:
            status = "✅" if r["score"] >= 0.99 else "❌"
            print(f"  {r['query_id']} ({r['answer_type']:12s}): {r['score']:.4f} {status}")

        all_perfect = all(r["score"] >= 0.99 for r in per_query)
        print(f"\n{'✅ ALL PERFECT' if all_perfect else '❌ SOME FAILED'}")
        print(f"\nScores: {json.dumps(aggregates['scores'], indent=2)}")

        # Also test wrong predictions
        print("\n" + "=" * 60)
        print("DRY RUN: Validating scorer with wrong predictions")
        print("=" * 60)

        wrong_preds = generate_wrong_predictions(queries)
        pred_map_w = {p["query_id"]: p for p in wrong_preds}
        per_query_w = [score_query(q, pred_map_w[q["query_id"]], alias_table) for q in queries]

        print(f"\nWrong predictions — expected all 0.0:")
        for r in per_query_w:
            status = "✅" if r["score"] <= 0.01 else "⚠️"
            print(f"  {r['query_id']} ({r['answer_type']:12s}): {r['score']:.4f} {status}")

        all_zero = all(r["score"] <= 0.01 for r in per_query_w)
        print(f"\n{'✅ ALL ZERO' if all_zero else '⚠️ SOME NON-ZERO (expected for token overlap)'}")

        # Write perfect-prediction report
        if args.output:
            report = build_report(per_query, aggregates, system="dry_run_perfect")
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nReport written to {args.output}")

        return 0 if all_perfect else 1

    else:
        # Real scoring mode
        if not args.predictions:
            print("ERROR: --predictions required (or use --dry-run)", file=sys.stderr)
            return 1

        with open(args.predictions) as f:
            pred_data = json.load(f)

        # Support both flat list and nested {predictions: [...]} format
        if isinstance(pred_data, list):
            predictions = pred_data
            prediction_metadata = {}
        else:
            predictions = pred_data.get("predictions", [])
            # Extract metadata from prediction.json top-level fields
            prediction_metadata = {
                k: v for k, v in pred_data.items()
                if k != "predictions"
            }

        pred_map = {p["query_id"]: p for p in predictions}

        per_query = []
        for q in queries:
            qid = q["query_id"]
            if qid not in pred_map:
                print(f"WARNING: No prediction for {qid}, scoring as 0.0")
                per_query.append({
                    "query_id": qid,
                    "task_family": q["task_family"],
                    "answer_type": q["answer_type"],
                    "depth_band": q["depth_band"],
                    "score": 0.0,
                    "expected": q["expected_answer"],
                    "predicted": "",
                })
                continue
            per_query.append(score_query(q, pred_map[qid], alias_table))

        aggregates = aggregate_scores(per_query)
        report = build_report(per_query, aggregates, system=args.system,
                              prediction_metadata=prediction_metadata)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Report written to {args.output}")

        # Print summary
        print(f"\n{'=' * 40}")
        print(f"EngramaBench Scorer Report")
        print(f"{'=' * 40}")
        print(f"System: {args.system}")
        print(f"\nScores:")
        for k, v in aggregates["scores"].items():
            print(f"  {k}: {v:.4f}")
        print(f"\nPer query:")
        for r in per_query:
            print(f"  {r['query_id']} ({r['task_family']:24s} {r['answer_type']:12s}): {r['score']:.4f}")

        return 0


if __name__ == "__main__":
    sys.exit(main())
