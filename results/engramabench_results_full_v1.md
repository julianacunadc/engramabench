# EngramaBench full_v1 — Results (Paper-Ready)

**Setup:** 5 personas × 20 conversations × 30 queries = 150 queries. Scorer v1.4.0. Branch: `codex/full-benchmark-prep`. Date: April 20, 2026.

## Table 1: Main Results (Global — Unified Registry)

| System | single_space | cross_space | temporal | adversarial | emergent | **composite** | cost |
|---|---|---|---|---|---|---|---|
| GPT-4o full-context | 0.7339 | 0.6291 | 0.3902 | 0.9750 | **0.3305** | **0.6186** | ~$3.33 |
| Engrama full | 0.5997 | **0.6532** | 0.3230 | 0.8000 | 0.2963 | 0.5367 | ~$0.67 |
| Engrama no-L1 | **0.6330** | 0.6368 | **0.3397** | 0.8250 | 0.2972 | 0.5488 | ~$0.67 |
| Engrama no-L3 | 0.6322 | 0.6033 | 0.3104 | 0.8500 | 0.3197 | 0.5501 | ~$0.67 |
| Engrama no-L1-L3 | 0.6322 | 0.5986 | 0.3274 | 0.8000 | 0.3081 | 0.5367 | ~$0.67 |
| Mem0 full | 0.2848 | 0.5266 | 0.2356 | **1.0000** | 0.2255 | 0.4809 | ~$0.36 |

## Table 2: Per-Persona Composite

| Persona | Engrama full | GPT-4o full-ctx | Δ |
|---|---|---|---|
| Priya (PI / professor) | 0.5716 | 0.6066 | −0.0350 |
| Carlos (founder) | 0.5294 | 0.6397 | −0.1103 |
| Sam (artist / musician) | 0.5569 | 0.6405 | −0.0836 |
| Diane (working parent / product lead) | 0.5329 | 0.6052 | −0.0723 |
| Kai (PhD student) | 0.4928 | 0.6037 | −0.1109 |
| **Per-persona mean** | **0.5367** | **0.6191** | **−0.0824** |

Note: Per-persona mean differs slightly from unified registry composite for GPT-4o (0.6191 vs. 0.6186) due to scoring methodology. For Engrama full, the per-persona mean and unified registry composite both round to 0.5367.

## Table 3: Cost Comparison (150 queries)

| System | Total cost | Cost/query |
|---|---|---|
| GPT-4o full-context | ~$3.33 | ~$0.022 |
| Engrama full | ~$0.67 | ~$0.0045 |

Engrama is ~5× cheaper per query.

## Table 4: Ablation Summary (Δ vs Engrama full = 0.5367)

| Variant | Δ composite | Δ single_space | Δ cross_space | Interpretation |
|---|---|---|---|---|
| no-L1 | +0.0121 | +0.0333 | −0.0164 | L1 appears to trade global composite for better cross-space behavior |
| no-L3 | +0.0134 | +0.0325 | −0.0499 | L3 appears to trade global composite for better cross-space behavior |
| no-L1-L3 | +0.0000 | +0.0325 | −0.0546 | Removing both preserves single-space gains but erases most cross-space advantage |

**Ablation takeaway:** L1 and L3 improve cross-space behavior, but in their current implementations they do not improve global composite. Removing either one slightly improves overall score, while removing both eliminates that gain, suggesting a real interaction between the levers rather than simple additive overhead.

## Narrative Highlights

1. **Engrama wins cross_space** (0.6532 vs 0.6291, +2.4pp) — the only category where graph memory outperforms full-context. Validates the core hypothesis of associative retrieval across conversation boundaries.

2. **Mem0 achieves perfect adversarial abstention** (1.0000) — better than GPT-4o (0.9750) and Engrama (0.8000). With less context available, it consistently abstains on fabricated questions.

3. **Temporal is hard for everyone** — all three systems score below 0.40, suggesting time-based reasoning is fundamentally challenging regardless of architecture.

4. **Cost-efficiency** — Engrama achieves 86.7% of GPT-4o's composite at 20% of the cost. Mem0 is cheapest (~$0.36) but substantially weaker.

5. **Mem0 vs Engrama** — Mem0 underperforms Engrama by 5.6pp composite, with the largest gaps in single_space (−31.5pp) and cross_space (−12.7pp). This suggests that native memory extraction quality, not only retrieval over stored memories, is a major bottleneck for long-term memory systems.

## Limitations

- Benchmark is reproducible and hash-stable at the run/artifact level (snapshot hashes, run_ids, deterministic scoring).
- Runtime evidence at the exact `message_id` / `chunk_id` level has not been fully audited.
- Mem0 baseline showed at least one instance of factual drift during memory extraction (invented date). This is treated as baseline behavior, not benchmark error.
