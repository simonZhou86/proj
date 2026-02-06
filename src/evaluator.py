from typing import Any
from textwrap import dedent

from .llm_client import OpenAICompatibleClient


class AnswerEvaluator:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        """Initialize the evaluator with an LLM client."""
        self.client = client

    def evaluate(
        self,
        question: str,
        gold_answer: str,
        predicted_answer: str,
        contexts: list[dict[str, str]],
        task_type: str = "",
        expected_verdict: str = "NA",
        source_quote: str = "",
    ) -> dict[str, Any]:
        '''
        Evaluate a model answer against ground truth and context.
        Args:
            question: The question being evaluated.
            gold_answer: The ground truth answer for the question.
            predicted_answer: The answer from answerer agent to evaluate.
            contexts: source chuncks, a list of dicts, each with keys 'source_id', 'chunk_id', 'title', 'text'.
            task_type: The type of task (e.g., "extractive_qa", "claim_quality_check") to inform evaluation criteria.
            expected_verdict: Only for claim_quality_check, the expected claim verdict label (FAITHFUL, HALLUCINATED, NOT_MENTIONED).
            source_quote: The specific quote or citation/reference from the generator that supports the ground truth answer.
        '''
        context_text = "\n\n".join(
            [
                (
                    f"[source={ctx['source_id']} chunk={ctx['chunk_id']} title={ctx['title']}]\n"
                    f"{ctx['text']}"
                )
                for ctx in contexts
            ]
        )
        result = self.client.chat_json(
            system_prompt=dedent(
                """\
                Your role: You are a strict judge that evaluates how accurate is the answer against provided source content.
                Your task: Judge the candidate answer relative to the question, ground truth answer, and context, focusing on factual correctness, completeness, and reasoning quality.
                Rules/constraints: Use only the provided context. Your output score must be in [0,1]. If task_type is *claim_quality_check*, prioritize verdict correctness and evidence grounding. Allowed claim verdict labels: FAITHFUL, HALLUCINATED, NOT_MENTIONED. If the question or ground truth answer is ambiguous, note it in ambiguity_note and reflect it in quality_flags. Avoid speculation and do not reference any external sources.
                Output format: Valid JSON parseable by json.loads() with keys: score, is_correct, feedback, error_type, quality_flags, ambiguity_note, claim_verdict_match.
                """
            ),
            user_prompt=dedent(
                f"""\
                Please evaluate the following inputs.

                Task type:
                {task_type}

                Question:
                {question}

                Ground truth answer:
                {gold_answer}

                Candidate answer:
                {predicted_answer}

                Expected claim verdict (if claim_quality_check):
                {expected_verdict}

                Generator source quote:
                {source_quote}

                Source context:
                {context_text}

                Notes:
                Evaluate factual correctness and reasoning quality.
                If task_type is claim_quality_check, prioritize verdict correctness and evidence grounding.
                quality_flags is a list of zero or more from: ambiguous_question, weak_gold, unsupported_by_context.
                """
            ),
            temperature=0.0,
        )

        score = float(result.get("score", 0.0))
        result["score"] = min(1.0, max(0.0, score))
        result["is_correct"] = bool(result.get("is_correct", score >= 0.8))
        result["feedback"] = str(result.get("feedback", ""))
        result["error_type"] = str(result.get("error_type", "none"))
        flags = result.get("quality_flags", [])
        if not isinstance(flags, list):
            flags = []
        result["quality_flags"] = [str(flag) for flag in flags]
        result["ambiguity_note"] = str(result.get("ambiguity_note", ""))
        result["claim_verdict_match"] = bool(result.get("claim_verdict_match", False))
        return result
