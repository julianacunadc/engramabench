# EngramaBench

A benchmark for evaluating long-term conversational memory in LLM assistants.

EngramaBench tests whether a system can retain, connect, and reason over information accumulated across many sessions. It is built around **5 canonical personas**, **100 multi-session conversations**, and **150 queries** spanning factual recall, cross-space integration, temporal reasoning, adversarial abstention, and emergent synthesis.

**Paper:** [EngramaBench: Evaluating Long-Term Conversational Memory with Structured Graph Retrieval](https://zenodo.org/records/19697774) ([PDF](paper/engramabench_draft_v1.pdf))

> This repository releases EngramaBench, its scorer, public baselines, and canonical results. Engrama itself is a proprietary system and is not included.

## Benchmark structure

Each persona has 4-5 thematic **spaces** (e.g., a founder has investors, product, hiring, market, personal). Conversations are timestamped and trace a medium-horizon narrative spanning late 2025 into early 2026. Queries are distributed across five task families:

| Family | Count | Tests |
|---|---|---|
| `single_space` | 30 | Factual recall within one space |
| `cross_space` | 30 | Integration across 2+ spaces |
| `temporal_cross_space` | 30 | Temporal reasoning over distributed facts |
| `adversarial` | 40 | Abstention on fabricated premises |
| `emergent_insight` | 20 | Synthesis requiring multi-evidence reasoning |

## Main results

All systems use **GPT-4o** as the answering model, isolating the effect of memory architecture.

| System | composite | single | cross | temporal | advers. | emergent | cost |
|---|---|---|---|---|---|---|---|
| GPT-4o full-context | **0.6186** | **0.7339** | 0.6291 | **0.3902** | 0.9750 | **0.3305** | $3.33 |
| Engrama full | 0.5367 | 0.5997 | **0.6532** | 0.3230 | 0.8000 | 0.2963 | $0.67 |
| Mem0 | 0.4809 | 0.2848 | 0.5266 | 0.2356 | **1.0000** | 0.2255 | $0.36 |

Engrama is the only system to outperform full-context prompting on `cross_space` (0.6532 vs. 0.6291), the task family most diagnostic of structured long-term memory.

## Quick start

### Score predictions

The scorer requires only Python 3.8+ with no external dependencies:

```bash
python3 scorer/scorer.py \
    --queries datasets/full_v1/queries.json \
    --predictions <your_predictions.json> \
    --registry datasets/full_v1/entity_registry.json \
    --output report.json
```

### Run the GPT-4o full-context baseline

```bash
pip install openai
export OPENAI_API_KEY=<your-key>

# Run per persona (recommended for checkpoint/resume)
python3 baselines/gpt4o_fullcontext/query_gpt4o.py \
    --queries datasets/full_v1/queries_priya.json \
    --conversations datasets/full_v1/conversations_priya.json \
    --output predictions_gpt4o_priya.json
```

### Run the Mem0 baseline

```bash
pip install mem0ai openai
export OPENAI_API_KEY=<your-key>

# Step 1: Seed Mem0 with conversations
python3 baselines/mem0/seed_mem0.py \
    --conversations datasets/full_v1/conversations_priya.json \
    --mem0-dir mem0_data/priya \
    --user-id benchmark_priya

# Step 2: Run queries
python3 baselines/mem0/query_mem0.py \
    --queries datasets/full_v1/queries_priya.json \
    --mem0-dir mem0_data/priya \
    --output predictions_mem0_priya.json \
    --user-id benchmark_priya
```

### Prediction format

Each prediction file is a JSON array of objects:

```json
[
  {
    "query_id": "q_priya_001",
    "answer": "March 15, 2026",
    "raw_response": "The R01 deadline is March 15, 2026."
  }
]
```

The `query_id` must match the IDs in the queries file. The `answer` field is what gets scored.

## Scoring

The scorer supports seven answer types: `entity`, `date`, `number`, `set`, `short_span`, `abstain`, and `insight`. The composite score is:

```
F = (single_space_f1 + cross_space_f1 + temporal_cross_space_f1) / 3
composite = 0.5 * F + 0.25 * adversarial_accuracy + 0.25 * emergent_insight_score
```

## Canonical results

Pre-computed score files for all systems evaluated in the paper are in `results/`:

- `scores_engrama_full_all.json`
- `scores_engrama_noL1_all.json`
- `scores_engrama_noL3_all.json`
- `scores_engrama_noL1L3_all.json`
- `scores_gpt4o_fullcontext_all.json`
- `scores_mem0_full_all.json`

A human-readable summary is in `results/engramabench_results_full_v1.md`.

## Repository structure

```
engramabench/
  README.md
  LICENSE
  requirements.txt
  datasets/full_v1/          # Conversations, queries, personas, entity registries
  scorer/scorer.py           # Benchmark scorer (stdlib only)
  baselines/
    gpt4o_fullcontext/       # GPT-4o full-context baseline
    mem0/                    # Mem0 vector retrieval baseline
  results/                   # Canonical score artifacts from the paper
  paper/                     # Preprint PDF
```

## Personas

| Persona | Domain | Spaces |
|---|---|---|
| Priya Sharma | Academic research / PI | Lab, grants, publications, collaborations, personal |
| Carlos Mendoza | Startup founder | Investors, product, hiring, market, personal |
| Sam Rodriguez | Artist / musician | Music, visual art, commissions, teaching, personal |
| Diane Chen | Working parent / product lead | Product, family, career, health, finances |
| Kai Nakamura | PhD student | Research, coursework, advisor, social, wellness |

## Citation

```bibtex
@article{acuna2026engramabench,
  title={EngramaBench: Evaluating Long-Term Conversational Memory with Structured Graph Retrieval},
  author={Acu{\~n}a, Juli{\'a}n},
  year={2026},
  doi={10.5281/zenodo.19697774},
  url={https://zenodo.org/records/19697774}
}
```

## License

MIT. See [LICENSE](LICENSE).
