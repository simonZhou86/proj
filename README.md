# Self-Evolving Benchmark Generator

This project implements a self-evolving benchmark generator where **one OpenAI-compatible endpoint** is used for:
- question generation
- model answering
- answer evaluation

It is designed for practical benchmarking over time. This version assumes life-science or medical source text.

## Why this version is useful

- Questions are grounded in one or more local articles (`data/context/*.md|*.txt`) instead of generating from scratch.
- Each new item is checked for novelty (lexical + LLM judge).
- The curriculum evolves toward weak areas (reasoning/difficulty/task-type buckets with EMA-driven sampling).
- Low-quality items are flagged (`ambiguous_question`, `weak_gold`, `unsupported_by_context`).
- Built-in task types: `extractive_qa`, `abstractive_qa`, `claim_quality_check` (faithfulness/hallucination).
- Difficulty is prompt-controlled with levels `1/3/5` and task-specific generation rules.
- Run summaries include metrics for model comparison over time.

## Architecture

Loop per round:
1. Sample target profile (reasoning type, difficulty, task type) from policy.
2. Sample source chunks from provided articles.
3. Generate a question + ground truth answer grounded in source.
4. Reject duplicates via novelty checks.
5. Answer with the same endpoint.
6. Evaluate with the same endpoint.
7. Update EMA and bucket EMAs, then adapt future sampling.

Batching:
- Each round selects 12 buckets (task_type × reasoning_type × difficulty).
- Round 1 focuses on difficulty=1 buckets, round 2 focuses on difficulty=3 buckets.
- From round 3 onward, buckets are sampled by EMA with a difficulty prior.

## EMA scoring

Global score uses exponential moving average:

`EMA_t = alpha * score_t + (1 - alpha) * EMA_{t-1}`

`alpha` is configurable (`run.ema_alpha` in config).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set endpoint credentials in either:
- `configs/default.yaml` (`endpoint.api_key`), or
- environment variable `OPENAI_API_KEY`.

Run:

```bash
python -m src.main --config configs/default.yaml
```

## Config knobs

Main controls in `configs/default.yaml`:
- `context.context_dir`: folder of source articles (`.md`, `.txt`)
- `generation.task_types`: supports multiple task forms
- `generation.temperature`: generator temperature
- `generation.difficulty_prior`: optional weight per difficulty for bucket sampling
- `run.rounds`: number of accepted attempts to target
- `run.questions_per_round`: number of buckets sampled per round
- `run.bucket_gamma`: strength of EMA weighting in bucket sampling
- `run.max_generation_attempts`: retries per round for novelty
- `run.novelty_lexical_threshold`: lexical duplicate cutoff
- `run.ema_alpha`: EMA smoothing factor

## Output files

- `data/questions.jsonl`: per-question records (question, answer, eval, novelty metadata, EMA, sources)
- `data/runs.jsonl`: run-level summary metrics

Summary includes:
- accepted question count
- novelty rejection count
- failed item count (items that exhausted attempts or failed novelty)
- mean accuracy score
- EMA score
- quality flag rate
- task-type coverage

## Tests

```bash
python -m unittest discover -s tests
```
