from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import logging
import uuid
from typing import Any

import yaml

from .answerer import QuestionAnswerer
from .context import ContextLibrary
from .evaluator import AnswerEvaluator
from .generator import QuestionGenerator
from .llm_client import LLMConfig, OpenAICompatibleClient
from .novelty import is_novel
from .policy import CurriculumPolicy
from .storage import JsonlStore


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Run the benchmark generation loop and return a run summary."""
    logger = logging.getLogger(__name__)
    logger.info("Starting run with config.")
    endpoint = config["endpoint"]
    generation_cfg = config["generation"]
    run_cfg = config["run"]
    storage_cfg = config["storage"]
    context_cfg = config["context"]

    client = OpenAICompatibleClient(
        LLMConfig(
            base_url=endpoint["base_url"],
            api_key=endpoint.get("api_key", ""),
            model=endpoint["model"],
            temperature=float(endpoint.get("temperature", 0.2)),
            max_tokens=int(endpoint.get("max_tokens", 800)),
        )
    )
    generator = QuestionGenerator(client)
    answerer = QuestionAnswerer(client)
    evaluator = AnswerEvaluator(client)

    question_store = JsonlStore(storage_cfg["question_history_path"])
    run_store = JsonlStore(storage_cfg["run_summary_path"])
    history = question_store.load_all()

    context_library = ContextLibrary.from_dir(
        context_dir=context_cfg["context_dir"],
        max_chars_per_chunk=int(context_cfg.get("max_chars_per_chunk", 1800)),
        seed=run_cfg.get("seed"),
    )

    policy = CurriculumPolicy(
        alpha=float(run_cfg["ema_alpha"]),
        reasoning_types=list(generation_cfg["reasoning_types"]),
        difficulty_levels=list(generation_cfg["difficulty_levels"]),
        task_types=list(generation_cfg["task_types"]),
        epsilon=float(run_cfg.get("epsilon", 0.15)),
        seed=run_cfg.get("seed"),
    )
    policy.replay(history)
    logger.info("Loaded %d historical records.", len(history))

    rounds = int(run_cfg["rounds"])
    max_attempts = int(run_cfg.get("max_generation_attempts", 4))
    lexical_threshold = float(run_cfg.get("novelty_lexical_threshold", 0.82))
    llm_window = int(run_cfg.get("novelty_llm_window", 15))
    n_docs = int(context_cfg.get("sample_docs_per_round", 1))
    n_chunks = int(context_cfg.get("sample_chunks_per_doc", 2))
    run_id = str(uuid.uuid4())
    questions_per_round = int(run_cfg.get("questions_per_round", 12))
    generator_temperature = float(generation_cfg.get("temperature", 0.7))
    difficulty_prior = generation_cfg.get("difficulty_prior", {})
    bucket_gamma = float(run_cfg.get("bucket_gamma", 1.0))

    accepted_records: list[dict[str, Any]] = []
    novelty_rejections = 0
    failed_items = 0

    round_index = 0
    all_buckets = [
        (task_type, reasoning, difficulty)
        for task_type in generation_cfg["task_types"]
        for reasoning in generation_cfg["reasoning_types"]
        for difficulty in generation_cfg["difficulty_levels"]
    ]

    for round_number in range(1, rounds + 1):
        logger.info("Round %d/%d starting.", round_number, rounds)
        batch_records: list[dict[str, Any]] = []
        if round_number == 1:
            difficulty_filter = {"1"}
        elif round_number == 2:
            difficulty_filter = {"3"}
        else:
            difficulty_filter = set(generation_cfg["difficulty_levels"])

        candidate_buckets = [b for b in all_buckets if b[2] in difficulty_filter]
        selected_buckets = policy.sample_buckets(
            buckets=candidate_buckets,
            n=questions_per_round,
            difficulty_prior=difficulty_prior,
            gamma=bucket_gamma,
        )

        for task_type, reasoning, difficulty in selected_buckets:
            logger.info(
                "Target bucket: task_type=%s reasoning=%s difficulty=%s",
                task_type,
                reasoning,
                difficulty,
            )
            target = {
                "task_type": task_type,
                "reasoning_type": reasoning,
                "difficulty": difficulty,
            }
            contexts = context_library.sample(n_docs=n_docs)
            generated: dict[str, Any] | None = None
            novelty_info: dict[str, Any] = {"method": "not_run"}

            for attempt in range(1, max_attempts + 1):
                generated = generator.generate(
                    target=target,
                    recent_questions=[row.get("question", "") for row in history],
                    contexts=contexts,
                    temperature=generator_temperature,
                )
                question = str(generated.get("question", "")).strip()
                if not question:
                    logger.warning("Empty question generated (attempt %d/%d).", attempt, max_attempts)
                    generated = None
                    continue

                novel, novelty_info = is_novel(
                    candidate_question=question,
                    history=history,
                    client=client,
                    lexical_threshold=lexical_threshold,
                    llm_window=llm_window,
                )
                if novel:
                    logger.info("Accepted novel question on attempt %d/%d.", attempt, max_attempts)
                    break
                logger.info("Rejected non-novel question on attempt %d/%d.", attempt, max_attempts)
                novelty_rejections += 1
                generated = None

            if generated is None:
                logger.warning("Failed to generate a novel question for bucket after %d attempts.", max_attempts)
                failed_items += 1
                continue

            question = str(generated["question"])
            gold_answer = str(generated.get("gold_answer", ""))
            task_type = str(generated.get("task_type", target["task_type"]))
            expected_verdict = str(generated.get("expected_verdict", "NA"))
            model_answer = answerer.answer(question=question, contexts=contexts, task_type=task_type)
            evaluation = evaluator.evaluate(
                question=question,
                gold_answer=gold_answer,
                predicted_answer=model_answer,
                contexts=contexts,
                task_type=task_type,
                expected_verdict=expected_verdict,
                source_quote=str(generated.get("source_quote", "")),
            )
            logger.info("Evaluation score: %.3f", float(evaluation.get("score", 0.0)))

            record = {
                "id": str(uuid.uuid4()),
                "run_id": run_id,
                "round_index": round_index,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "question": question,
                "gold_answer": gold_answer,
                "model_answer": model_answer,
                "question_profile": {
                    "topic": "life_science",
                    "reasoning_type": generated.get("reasoning_type", target["reasoning_type"]),
                    "difficulty": generated.get("difficulty", target["difficulty"]),
                    "task_type": task_type,
                    "source_id": generated.get("source_id", ""),
                    "claim_text": generated.get("claim_text", ""),
                    "expected_verdict": expected_verdict,
                    "utility_rationale": generated.get("utility_rationale", ""),
                    "quality_risk": generated.get("quality_risk", "none"),
                },
                "sources": contexts,
                "novelty": novelty_info,
                "evaluation": evaluation,
            }

            batch_records.append(record)
            history.append(record)
            accepted_records.append(record)
            question_store.append(record)
            round_index += 1

        for record in batch_records:
            policy.update(record)
            record["ema_score"] = policy.global_ema
        logger.info(
            "Round %d complete. Accepted=%d, Failed=%d, Novelty rejects=%d, EMA=%.3f",
            round_number,
            len(batch_records),
            failed_items,
            novelty_rejections,
            policy.global_ema if policy.global_ema is not None else 0.0,
        )

    summary = build_summary(
        run_id=run_id,
        model_name=endpoint["model"],
        records=accepted_records,
        novelty_rejections=novelty_rejections,
        failed_items=failed_items,
        total_rounds=rounds,
        current_ema=policy.global_ema,
    )
    run_store.append(summary)
    logger.info("Run complete. Summary written to store.")
    return summary


def build_summary(
    run_id: str,
    model_name: str,
    records: list[dict[str, Any]],
    novelty_rejections: int,
    failed_items: int,
    total_rounds: int,
    current_ema: float | None,
) -> dict[str, Any]:
    """Aggregate run records into a summary metrics dictionary."""
    if not records:
        return {
            "run_id": run_id,
            "model": model_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "accepted_questions": 0,
            "total_rounds": total_rounds,
            "failed_items": failed_items,
            "novelty_rejections": novelty_rejections,
            "accuracy_mean": 0.0,
            "ema_score": current_ema if current_ema is not None else 0.0,
            "quality_flag_rate": 0.0,
            "task_coverage": {},
        }

    scores = [float(r["evaluation"]["score"]) for r in records]
    quality_flags = sum(1 for r in records if r["evaluation"].get("quality_flags"))
    task_counts: dict[str, int] = {}
    for r in records:
        task_type = str(r["question_profile"].get("task_type", "unknown"))
        task_counts[task_type] = task_counts.get(task_type, 0) + 1

    return {
        "run_id": run_id,
        "model": model_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_questions": len(records),
        "total_rounds": total_rounds,
        "failed_items": failed_items,
        "novelty_rejections": novelty_rejections,
        "accuracy_mean": sum(scores) / len(scores),
        "ema_score": current_ema if current_ema is not None else 0.0,
        "quality_flag_rate": quality_flags / len(records),
        "task_coverage": task_counts,
    }


def main() -> None:
    """CLI entrypoint for running the benchmark generator."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Self-evolving benchmark generator")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = run(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
