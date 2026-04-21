#!/usr/bin/env python3
"""
query_gpt4o.py — GPT-4o full-context baseline for EngramaBench.

For each query, concatenates ALL conversations (chronological) as context,
then asks GPT-4o to answer. This is the "brute-force context window" baseline:
no memory system, no retrieval — just dump everything into the prompt.

This is the primary baseline for the paper (docs/30-paper-strategy.md §6).
Supports checkpoint/resume (--resume) for crash recovery.

Usage:
    python3 scripts/engramabench/baselines/gpt4o_fullcontext/query_gpt4o.py \
        --queries datasets/engramabench/full_v1/queries_priya.json \
        --conversations datasets/engramabench/full_v1/conversations_priya.json \
        --output predictions_gpt4o_priya.json

Requires:
    pip install openai
    OPENAI_API_KEY env var set
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone

# Pinned model (docs/33-phase05-full-benchmark-spec.md §5.1, decision D3)
LLM_MODEL = "gpt-4o-2024-08-06"
LLM_TEMPERATURE = 0.3

# Pricing (GPT-4o-2024-08-06)
PRICING = {
    "gpt-4o-2024-08-06": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
}

SYSTEM_PROMPT = """You are a helpful assistant. You have access to the full history of conversations between a user and their assistant, organized by thematic spaces.

Answer the question based ONLY on the conversations provided below.
If the conversations do not contain enough information to answer, say "I don't have that information."
Be concise and specific. Do not invent facts not present in the conversations."""


def build_full_context(conversations: list) -> str:
    """Concatenate all conversations chronologically into a single context string."""
    sorted_convs = sorted(conversations, key=lambda c: c.get("depth_index", 0))

    sections = []
    for conv in sorted_convs:
        cid = conv["conversation_id"]
        space = conv["space_id"]
        timestamp = conv.get("timestamp", "")
        date_str = timestamp[:10] if timestamp else ""

        header = f"[Space: {space}] Conversation {cid}"
        if date_str:
            header += f" ({date_str})"

        messages = []
        for msg in conv["messages"]:
            role = "User" if msg["role"] == "user" else "Assistant"
            messages.append(f"{role}: {msg['content']}")

        section = header + "\n" + "\n".join(messages)
        sections.append(section)

    return "\n\n".join(sections)


def count_tokens_approx(text: str) -> int:
    """Rough token estimate (4 chars per token). For logging only."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _compute_file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return f"sha256:{h.hexdigest()[:16]}"


def _write_checkpoint_atomic(checkpoint_path: str, data: dict):
    dir_name = os.path.dirname(checkpoint_path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, checkpoint_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_checkpoint(checkpoint_path: str) -> dict | None:
    if not os.path.exists(checkpoint_path):
        return None
    with open(checkpoint_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="GPT-4o full-context baseline for EngramaBench"
    )
    parser.add_argument("--queries", required=True, help="Path to queries.json")
    parser.add_argument("--conversations", required=True, help="Path to conversations.json")
    parser.add_argument("--output", required=True, help="Output predictions JSON path")
    parser.add_argument("--filter", default=None,
                        help="Comma-separated query IDs (e.g., q_007,q_015)")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from last checkpoint")
    parser.add_argument("--checkpoint-path", default=None,
                        help="Checkpoint file path (default: <output>.checkpoint.json)")
    args = parser.parse_args()

    # Validate env
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Load data
    with open(args.queries) as f:
        all_queries = json.load(f)

    with open(args.conversations) as f:
        conversations = json.load(f)

    # Apply filter
    if args.filter:
        filter_ids = set(args.filter.split(","))
        queries = [q for q in all_queries if q["query_id"] in filter_ids]
        print(f"Loaded {len(queries)} queries (filtered from {len(all_queries)})")
    else:
        queries = all_queries
        print(f"Loaded {len(queries)} queries")

    # Config provenance
    queries_hash = _compute_file_hash(args.queries)
    conv_hash = _compute_file_hash(args.conversations)
    system_config_hash = hashlib.sha256(
        json.dumps({"model": LLM_MODEL, "temperature": LLM_TEMPERATURE,
                     "system_prompt": SYSTEM_PROMPT}, sort_keys=True).encode()
    ).hexdigest()[:16]
    system_config_version = f"gpt4o-fullctx:{LLM_MODEL}"

    print(f"Conversations: {len(conversations)}")
    print(f"Model: {LLM_MODEL} (temperature={LLM_TEMPERATURE})")

    # Build full context (same for all queries — this IS the baseline)
    full_context = build_full_context(conversations)
    approx_tokens = count_tokens_approx(full_context)
    print(f"Full context: ~{approx_tokens} tokens (approx)")

    # Check context fits in GPT-4o window (128K)
    if approx_tokens > 120_000:
        print(f"WARNING: Context may exceed GPT-4o window (~{approx_tokens} tokens)")

    # Checkpoint setup
    checkpoint_path = args.checkpoint_path or f"{args.output}.checkpoint.json"
    effective_query_ids = sorted(q["query_id"] for q in queries)
    predictions: list[dict] = []
    completed_ids: set[str] = set()

    if args.resume:
        ckpt = _load_checkpoint(checkpoint_path)
        if ckpt:
            # Validate identity
            for key, expected in [
                ("queries_hash", queries_hash),
                ("conversations_hash", conv_hash),
                ("system_config_hash", f"sha256:{system_config_hash}"),
                ("effective_query_ids", effective_query_ids),
            ]:
                if ckpt.get(key) != expected:
                    print(f"ERROR: Checkpoint mismatch on '{key}'. "
                          f"Delete {checkpoint_path} to start fresh.", file=sys.stderr)
                    sys.exit(1)

            predictions = ckpt.get("predictions", [])
            completed_ids = {p["query_id"] for p in predictions}
            print(f"Resumed from checkpoint: {len(completed_ids)} queries done")
        else:
            print("No checkpoint found, starting fresh")

    # Import OpenAI
    from openai import OpenAI
    client = OpenAI()

    # Run queries
    for i, query in enumerate(queries):
        qid = query["query_id"]
        question = query["question"]
        query_space = query.get("query_space", "")

        if qid in completed_ids:
            print(f"[{i+1}/{len(queries)}] {qid} — skipped (already completed)")
            continue

        user_message = f"""Conversations:
{full_context}

Question: {question}"""

        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            latency_ms = int((time.time() - t0) * 1000)

            choice = response.choices[0]
            system_answer = choice.message.content.strip()

            tokens_in = response.usage.prompt_tokens if response.usage else 0
            tokens_out = response.usage.completion_tokens if response.usage else 0

            prices = PRICING.get(LLM_MODEL, {"input": 0, "output": 0})
            cost_usd = tokens_in * prices["input"] + tokens_out * prices["output"]

        except Exception as e:
            latency_ms = int((time.time() - t0) * 1000)
            print(f"[{i+1}/{len(queries)}] {qid} ERROR: {e}")
            raise  # Crash so we can resume; don't silently record errors

        prediction = {
            "query_id": qid,
            "system_answer": system_answer,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
        }
        predictions.append(prediction)
        completed_ids.add(qid)

        print(f"[{i+1}/{len(queries)}] {qid} (space={query_space})")
        print(f"  Q: {question[:80]}...")
        print(f"  A: {system_answer[:80]}...")
        print(f"  Tokens: {tokens_in}in/{tokens_out}out, "
              f"cost=${cost_usd:.4f}, latency={latency_ms}ms")

        # Checkpoint after each successful query
        _write_checkpoint_atomic(checkpoint_path, {
            "run_type": "gpt4o_fullcontext_query",
            "queries_hash": queries_hash,
            "conversations_hash": conv_hash,
            "system_config_hash": f"sha256:{system_config_hash}",
            "effective_query_ids": effective_query_ids,
            "completed_query_ids": sorted(completed_ids),
            "predictions": predictions,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    # Reorder predictions to match original query order
    query_order = {q["query_id"]: idx for idx, q in enumerate(queries)}
    predictions.sort(key=lambda p: query_order.get(p["query_id"], float("inf")))

    # Build output
    run_id = f"gpt4o_fullctx_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    total_cost = sum(p["cost_usd"] for p in predictions)
    avg_latency = (
        sum(p["latency_ms"] for p in predictions) / len(predictions)
        if predictions else 0
    )

    output = {
        "run_id": run_id,
        "system": "gpt4o_fullcontext",
        "system_model": LLM_MODEL,
        "system_config_version": system_config_version,
        "system_config_hash": f"sha256:{system_config_hash}",
        "benchmark_version": "full-v1",
        "dataset_version": "full-v1",
        "temperature": LLM_TEMPERATURE,
        "context_conversations": len(conversations),
        "context_tokens_approx": approx_tokens,
        "predictions": predictions,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Clean up checkpoint on success
    if os.path.exists(checkpoint_path):
        os.unlink(checkpoint_path)

    print(f"\n{'='*60}")
    print("QUERY COMPLETE")
    print(f"{'='*60}")
    print(f"Queries: {len(predictions)}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Avg latency: {avg_latency:.0f}ms")
    print(f"Predictions written: {args.output}")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
