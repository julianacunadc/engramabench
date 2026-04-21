#!/usr/bin/env python3
"""
seed_mem0.py — Ingest EngramaBench conversations into Mem0 OSS.

Reads conversations from the EngramaBench dataset, feeds them into
Mem0's Memory.add() with infer=True (Mem0 extracts facts naturally), and
persists the resulting memory store to disk.
Supports checkpoint/resume (--resume) for crash recovery.

Usage:
    python3 scripts/engramabench/baselines/mem0/seed_mem0.py \
        --conversations datasets/engramabench/full_v1/conversations_priya.json \
        --output-dir mem0_data/priya \
        --user-id benchmark_priya

Requires:
    pip install mem0ai
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
from pathlib import Path


def get_mem0_config(output_dir: str) -> dict:
    """Pinned Mem0 OSS config for reproducible benchmark runs.

    Overrides Mem0 defaults (Qdrant + SQLite) with explicit ChromaDB config.
    See docs/34-mem0-pilot-validation-plan.md §1 for rationale.
    """
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
                "path": os.path.join(output_dir, "chroma"),
            },
        },
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
    os.makedirs(dir_name, exist_ok=True)
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
    parser = argparse.ArgumentParser(description="Seed Mem0 with EngramaBench conversations")
    parser.add_argument("--conversations", required=True, help="Path to conversations.json (per-persona)")
    parser.add_argument("--output-dir", required=True, help="Directory for Mem0 persistent data")
    parser.add_argument("--user-id", default="carlos", help="Mem0 user_id (e.g. benchmark_priya)")
    parser.add_argument("--dry-run", action="store_true", help="Load and validate only, don't call Mem0")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from last checkpoint (skip completed conversations)")
    parser.add_argument("--checkpoint-path", default=None,
                        help="Checkpoint file path (default: <output-dir>/seed_checkpoint.json)")
    args = parser.parse_args()

    # Validate env
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Load conversations
    with open(args.conversations) as f:
        conversations = json.load(f)

    # Sort by depth_index (chronological ingestion order)
    conversations.sort(key=lambda c: c.get("depth_index", 0))

    conv_hash = _compute_file_hash(args.conversations)

    print(f"Loaded {len(conversations)} conversations")
    print(f"Output dir: {args.output_dir}")
    print(f"Ingestion order: by depth_index (chronological)")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        for conv in conversations:
            cid = conv["conversation_id"]
            space = conv["space_id"]
            msgs = len(conv["messages"])
            depth = conv.get("depth_index", "?")
            print(f"  {cid} | space={space} | messages={msgs} | depth_index={depth}")
        print(f"\nTotal: {len(conversations)} conversations, "
              f"{sum(len(c['messages']) for c in conversations)} messages")
        return

    # Import Mem0 (only when not dry-run)
    try:
        from mem0 import Memory
    except ImportError:
        print("ERROR: mem0ai not installed. Run: pip install mem0ai", file=sys.stderr)
        sys.exit(1)

    # Initialize Mem0
    os.makedirs(args.output_dir, exist_ok=True)
    config = get_mem0_config(args.output_dir)
    system_config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:16]
    m = Memory.from_config(config)

    print(f"\nMem0 initialized with config:")
    print(f"  LLM: {config['llm']['config']['model']} (temp={config['llm']['config']['temperature']})")
    print(f"  Embedder: {config['embedder']['config']['model']}")
    print(f"  Vector store: ChromaDB at {config['vector_store']['config']['path']}")

    # Checkpoint setup
    checkpoint_path = args.checkpoint_path or os.path.join(args.output_dir, "seed_checkpoint.json")
    completed_conv_ids: set[str] = set()
    seed_log_convs: list[dict] = []
    total_memories_extracted = 0
    total_latency_ms = 0

    if args.resume:
        ckpt = _load_checkpoint(checkpoint_path)
        if ckpt:
            # Validate identity (including config to prevent mixed-config stores)
            for key, expected in [
                ("conversations_hash", conv_hash),
                ("system_config_hash", f"sha256:{system_config_hash}"),
                ("user_id", args.user_id),
                ("output_dir", os.path.abspath(args.output_dir)),
            ]:
                if ckpt.get(key) != expected:
                    print(f"ERROR: Checkpoint mismatch on '{key}'. "
                          f"Delete {checkpoint_path} and reseed from scratch.",
                          file=sys.stderr)
                    sys.exit(1)

            # Detect crash-boundary: if pending_conversation_id is set, m.add()
            # may or may not have persisted. The store is in an ambiguous state.
            pending = ckpt.get("pending_conversation_id")
            if pending:
                print(
                    f"ERROR: Crash detected — conversation '{pending}' was in-flight "
                    f"when the process died. The Mem0 store at '{args.output_dir}' "
                    f"may contain partial or duplicate memories from this conversation.\n"
                    f"\n"
                    f"Cannot resume safely. To fix:\n"
                    f"  rm -rf '{args.output_dir}'\n"
                    f"Then reseed from scratch (without --resume).\n",
                    file=sys.stderr,
                )
                sys.exit(1)

            completed_conv_ids = set(ckpt.get("completed_conversation_ids", []))

            seed_log_convs = ckpt.get("seed_log_conversations", [])
            total_memories_extracted = ckpt.get("total_memories_extracted", 0)
            total_latency_ms = ckpt.get("total_latency_ms", 0)
            print(f"Resumed from checkpoint: {len(completed_conv_ids)} conversations done")
        else:
            print("No checkpoint found, starting fresh")

    # Ingest conversations
    print(f"\n{'='*60}")
    print("SEEDING MEM0")
    print(f"{'='*60}")

    for i, conv in enumerate(conversations):
        cid = conv["conversation_id"]
        space = conv["space_id"]
        timestamp = conv.get("timestamp", "")
        depth_index = conv.get("depth_index", 0)

        if cid in completed_conv_ids:
            print(f"[{i+1}/{len(conversations)}] {cid} — skipped (already completed)")
            continue

        # Format messages for Mem0
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in conv["messages"]
        ]

        # Write pending marker BEFORE m.add() — if we crash after m.add()
        # but before the completion checkpoint, resume can detect the boundary.
        _write_checkpoint_atomic(checkpoint_path, {
            "run_type": "mem0_seed",
            "conversations_hash": conv_hash,
            "system_config_hash": f"sha256:{system_config_hash}",
            "user_id": args.user_id,
            "output_dir": os.path.abspath(args.output_dir),
            "pending_conversation_id": cid,
            "completed_conversation_ids": sorted(completed_conv_ids),
            "seed_log_conversations": seed_log_convs,
            "total_memories_extracted": total_memories_extracted,
            "total_latency_ms": total_latency_ms,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        t0 = time.time()
        try:
            result = m.add(
                messages=messages,
                user_id=args.user_id,
                metadata={
                    "space_id": space,
                    "conversation_id": cid,
                    "timestamp": timestamp,
                    "depth_index": depth_index,
                },
            )
            latency_ms = int((time.time() - t0) * 1000)

            # Count memories extracted
            if isinstance(result, dict):
                memories = result.get("results", result.get("memories", []))
            elif isinstance(result, list):
                memories = result
            else:
                memories = []

            n_memories = len(memories) if memories else 0

            conv_log = {
                "conversation_id": cid,
                "space_id": space,
                "depth_index": depth_index,
                "messages_count": len(messages),
                "memories_extracted": n_memories,
                "latency_ms": latency_ms,
            }
            seed_log_convs.append(conv_log)
            total_memories_extracted += n_memories
            total_latency_ms += latency_ms
            completed_conv_ids.add(cid)

            print(f"[{i+1}/{len(conversations)}] {cid} (space={space})")
            print(f"  Messages: {len(messages)}, Memories extracted: {n_memories}, "
                  f"Latency: {latency_ms}ms")

            # Clear pending marker and record completion
            _write_checkpoint_atomic(checkpoint_path, {
                "run_type": "mem0_seed",
                "conversations_hash": conv_hash,
                "system_config_hash": f"sha256:{system_config_hash}",
                "user_id": args.user_id,
                "output_dir": os.path.abspath(args.output_dir),
                "pending_conversation_id": None,
                "completed_conversation_ids": sorted(completed_conv_ids),
                "seed_log_conversations": seed_log_convs,
                "total_memories_extracted": total_memories_extracted,
                "total_latency_ms": total_latency_ms,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

        except Exception as e:
            latency_ms = int((time.time() - t0) * 1000)
            print(f"[{i+1}/{len(conversations)}] {cid} ERROR: {e}")
            # pending_conversation_id is already set in checkpoint —
            # on resume, this conversation will be detected and skipped
            raise

    # Get total memory count
    try:
        all_memories = m.get_all(filters={"user_id": args.user_id})
        if isinstance(all_memories, dict):
            total = len(all_memories.get("results", all_memories.get("memories", [])))
        elif isinstance(all_memories, list):
            total = len(all_memories)
        else:
            total = "unknown"
    except Exception as e:
        total = f"error: {e}"

    # Write seed log
    seed_log = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "config": config,
        "user_id": args.user_id,
        "conversations": seed_log_convs,
        "total_memories_extracted": total_memories_extracted,
        "total_memories_in_store": total,
        "total_latency_ms": total_latency_ms,
    }

    log_path = os.path.join(args.output_dir, "mem0_seed_log.json")
    with open(log_path, "w") as f:
        json.dump(seed_log, f, indent=2)

    # Clean up checkpoint on success
    if os.path.exists(checkpoint_path):
        os.unlink(checkpoint_path)

    print(f"\n{'='*60}")
    print("SEED COMPLETE")
    print(f"{'='*60}")
    print(f"Conversations: {len(conversations)}")
    print(f"Memories extracted: {total_memories_extracted}")
    print(f"Total memories in store: {total}")
    print(f"Total latency: {total_latency_ms}ms")
    print(f"Seed log: {log_path}")


if __name__ == "__main__":
    main()
