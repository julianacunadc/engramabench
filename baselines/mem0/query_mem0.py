#!/usr/bin/env python3
"""
query_mem0.py — Run EngramaBench queries against a seeded Mem0 instance.

For each query, searches Mem0 for relevant memories, then asks GPT-4o to
answer using the retrieved context. Outputs scorer-compatible predictions JSON.
Supports checkpoint/resume (--resume) for crash recovery.

Usage:
    python3 scripts/engramabench/baselines/mem0/query_mem0.py \
        --queries datasets/engramabench/full_v1/queries_priya.json \
        --mem0-dir mem0_data/priya \
        --output predictions_mem0_priya.json \
        --user-id benchmark_priya

Requires:
    pip install mem0ai openai
    OPENAI_API_KEY env var set
    Mem0 seeded via seed_mem0.py first
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
from pathlib import Path

# Pricing for cost estimation (GPT-4o-2024-08-06)
PRICING = {
    "gpt-4o-2024-08-06": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
}

ANSWER_SYSTEM_PROMPT = """You are a helpful assistant with access to a user's personal memories.
Answer the question based ONLY on the memories provided below.
If the memories do not contain enough information to answer, say "I don't have that information."
Be concise and specific. Do not invent facts not present in the memories."""

LLM_MODEL = "gpt-4o-2024-08-06"
LLM_TEMPERATURE = 0.3
TOP_K = 20


def get_mem0_config(mem0_dir: str) -> dict:
    """Same pinned config as seed_mem0.py — must match exactly."""
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-2024-08-06",
                "temperature": 0.1,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
                "embedding_dims": 1536,
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "engramabench_full_v1",
                "path": os.path.join(mem0_dir, "chroma"),
            },
        },
    }


def format_memories_as_context(memories: list) -> str:
    """Convert Mem0 search results into a context string for the LLM."""
    if not memories:
        return "(No relevant memories found.)"

    lines = []
    for i, mem in enumerate(memories, 1):
        if isinstance(mem, dict):
            content = mem.get("memory", mem.get("content", str(mem)))
            metadata = mem.get("metadata", {})
            space = metadata.get("space_id", "")
            space_tag = f" [{space}]" if space else ""
        else:
            content = str(mem)
            space_tag = ""

        lines.append(f"- {content}{space_tag}")

    return "\n".join(lines)


def ask_llm(context: str, question: str) -> dict:
    """Ask GPT-4o to answer using retrieved memories as context."""
    from openai import OpenAI

    client = OpenAI()

    user_message = f"""Memories:
{context}

Question: {question}"""

    t0 = time.time()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
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

    return {
        "system_answer": system_answer,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
    }


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
    parser = argparse.ArgumentParser(description="Query Mem0 baseline on EngramaBench")
    parser.add_argument("--queries", required=True, help="Path to queries.json")
    parser.add_argument("--mem0-dir", required=True, help="Directory with seeded Mem0 data")
    parser.add_argument("--output", required=True, help="Output predictions JSON path")
    parser.add_argument("--filter", default=None,
                        help="Comma-separated query IDs to run (e.g., q_007,q_015)")
    parser.add_argument("--user-id", default="carlos",
                        help="Mem0 user_id (must match seed_mem0.py --user-id)")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Number of memories to retrieve (default: {TOP_K})")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from last checkpoint")
    parser.add_argument("--checkpoint-path", default=None,
                        help="Checkpoint file path (default: <output>.checkpoint.json)")
    args = parser.parse_args()

    # Validate env
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Validate Mem0 data exists
    chroma_path = os.path.join(args.mem0_dir, "chroma")
    if not os.path.exists(chroma_path):
        print(f"ERROR: Mem0 data not found at {chroma_path}. Run seed_mem0.py first.",
              file=sys.stderr)
        sys.exit(1)

    # Load queries
    with open(args.queries) as f:
        all_queries = json.load(f)

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
    mem0_config = get_mem0_config(args.mem0_dir)
    system_config_hash = hashlib.sha256(
        json.dumps({"model": LLM_MODEL, "temperature": LLM_TEMPERATURE,
                     "top_k": args.top_k, "mem0_config": mem0_config,
                     "system_prompt": ANSWER_SYSTEM_PROMPT}, sort_keys=True).encode()
    ).hexdigest()[:16]
    system_config_version = f"mem0-oss:{LLM_MODEL}:topk{args.top_k}"

    # Store identity: hash the seed log to detect reseeds between resume runs
    seed_log_path = os.path.join(args.mem0_dir, "mem0_seed_log.json")
    if os.path.exists(seed_log_path):
        seed_log_hash = _compute_file_hash(seed_log_path)
    else:
        print("WARNING: mem0_seed_log.json not found — store identity cannot be validated",
              file=sys.stderr)
        seed_log_hash = "unknown"

    # Checkpoint setup
    checkpoint_path = args.checkpoint_path or f"{args.output}.checkpoint.json"
    effective_query_ids = sorted(q["query_id"] for q in queries)
    predictions: list[dict] = []
    completed_ids: set[str] = set()

    if args.resume:
        ckpt = _load_checkpoint(checkpoint_path)
        if ckpt:
            # Resolve mem0_dir to absolute for stable comparison
            mem0_dir_abs = os.path.abspath(args.mem0_dir)
            for key, expected in [
                ("queries_hash", queries_hash),
                ("system_config_hash", f"sha256:{system_config_hash}"),
                ("seed_log_hash", seed_log_hash),
                ("top_k", args.top_k),
                ("mem0_dir", mem0_dir_abs),
                ("effective_query_ids", effective_query_ids),
                ("user_id", args.user_id),
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

    # Initialize Mem0
    try:
        from mem0 import Memory
    except ImportError:
        print("ERROR: mem0ai not installed. Run: pip install mem0ai", file=sys.stderr)
        sys.exit(1)

    m = Memory.from_config(mem0_config)

    print(f"Mem0 dir: {args.mem0_dir}")
    print(f"LLM: {LLM_MODEL} (temperature={LLM_TEMPERATURE})")
    print(f"Top-K: {args.top_k}")

    # Run queries
    total_search_latency = 0
    total_llm_latency = 0

    for i, query in enumerate(queries):
        qid = query["query_id"]
        question = query["question"]
        query_space = query.get("query_space", "")

        if qid in completed_ids:
            print(f"[{i+1}/{len(queries)}] {qid} — skipped (already completed)")
            continue

        # Step 1: Search Mem0
        t0_search = time.time()
        try:
            results = m.search(
                question,
                filters={"user_id": args.user_id},
                top_k=args.top_k,
            )

            if isinstance(results, dict):
                memories = results.get("results", results.get("memories", []))
            elif isinstance(results, list):
                memories = results
            else:
                memories = []

        except Exception as e:
            print(f"[{i+1}/{len(queries)}] {qid} SEARCH ERROR: {e}")
            raise  # Crash to resume; don't silently record errors

        search_latency_ms = int((time.time() - t0_search) * 1000)
        total_search_latency += search_latency_ms

        # Step 2: Format context
        context = format_memories_as_context(memories)

        # Step 3: Ask LLM
        try:
            llm_result = ask_llm(context, question)
        except Exception as e:
            print(f"[{i+1}/{len(queries)}] {qid} LLM ERROR: {e}")
            raise  # Crash to resume

        total_llm_latency += llm_result["latency_ms"]
        total_latency_ms = search_latency_ms + llm_result["latency_ms"]

        prediction = {
            "query_id": qid,
            "system_answer": llm_result["system_answer"],
            "tokens_in": llm_result["tokens_in"],
            "tokens_out": llm_result["tokens_out"],
            "cost_usd": llm_result["cost_usd"],
            "latency_ms": total_latency_ms,
            "search_latency_ms": search_latency_ms,
            "llm_latency_ms": llm_result["latency_ms"],
            "memories_retrieved": len(memories),
            "memories_content": [
                m_item.get("memory", str(m_item)) if isinstance(m_item, dict) else str(m_item)
                for m_item in memories[:5]
            ],
        }
        predictions.append(prediction)
        completed_ids.add(qid)

        print(f"[{i+1}/{len(queries)}] {qid} (space={query_space})")
        print(f"  Q: {question[:80]}...")
        print(f"  A: {llm_result['system_answer'][:80]}...")
        print(f"  Memories: {len(memories)}, Search: {search_latency_ms}ms, "
              f"LLM: {llm_result['latency_ms']}ms, "
              f"Cost: ${llm_result['cost_usd']:.4f}")

        # Checkpoint after each successful query
        _write_checkpoint_atomic(checkpoint_path, {
            "run_type": "mem0_query",
            "queries_hash": queries_hash,
            "system_config_hash": f"sha256:{system_config_hash}",
            "seed_log_hash": seed_log_hash,
            "top_k": args.top_k,
            "mem0_dir": os.path.abspath(args.mem0_dir),
            "effective_query_ids": effective_query_ids,
            "user_id": args.user_id,
            "completed_query_ids": sorted(completed_ids),
            "predictions": predictions,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    # Reorder predictions to match original query order
    query_order = {q["query_id"]: idx for idx, q in enumerate(queries)}
    predictions.sort(key=lambda p: query_order.get(p["query_id"], float("inf")))

    # Build output (scorer-compatible schema)
    run_id = f"mem0_full_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    total_cost = sum(p["cost_usd"] for p in predictions)
    avg_latency = (
        sum(p["latency_ms"] for p in predictions) / len(predictions)
        if predictions else 0
    )

    output = {
        "run_id": run_id,
        "system": "mem0",
        "system_model": LLM_MODEL,
        "system_config_version": system_config_version,
        "system_config_hash": f"sha256:{system_config_hash}",
        "mem0_config": mem0_config,
        "benchmark_version": "full-v1",
        "dataset_version": "full-v1",
        "top_k": args.top_k,
        "temperature": LLM_TEMPERATURE,
        "predictions": predictions,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Clean up checkpoint on success
    if os.path.exists(checkpoint_path):
        os.unlink(checkpoint_path)

    n_preds = len(predictions)
    print(f"\n{'='*60}")
    print("QUERY COMPLETE")
    print(f"{'='*60}")
    print(f"Queries: {n_preds}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Avg total latency: {avg_latency:.0f}ms")
    print(f"Avg search latency: {total_search_latency/n_preds:.0f}ms" if n_preds else "")
    print(f"Avg LLM latency: {total_llm_latency/n_preds:.0f}ms" if n_preds else "")
    print(f"Predictions written: {args.output}")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
