from typing import Any
from textwrap import dedent

from .llm_client import OpenAICompatibleClient


class QuestionGenerator:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        """Initialize the generator with an LLM client."""
        self.client = client

    def generate(
        self,
        target: dict[str, str],
        recent_questions: list[str],
        contexts: list[dict[str, str]],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        '''
        Generate a question and metadata for the given target profile.
        Args:
            target: A dict with keys 'reasoning_type', 'difficulty', 'task_type'.
            recent_questions: A list of recently generated questions to avoid duplication.
            contexts: source chuncks, a list of dicts, each with keys 'source_id', 'chunk_id', 'title', 'text'.
            temperature: Optional temperature setting for the LLM.
        '''
        recent_text = "\n".join([f"- {q}" for q in recent_questions[-10:]]) or "- None"
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
                Your role: You are an expert question generation agent that creates high-quality and novel questions for life sciences fields. You have deep knowledge of biology, medicine, and scientific research.
                Your task: Generate a novel, high-utility question grounded in the provided sources with the required reasoning type, difficulty level, and task type. Use your reasoning ability to think step by step before generating the question.
                Rules/constraints: Use only the provided sources. Produce questions that reveal model weaknesses and avoid duplicates of recent questions. Ensure the ground truth answer is unambiguous and supported by the sources. Prefer questions that test reasoning under constraints, comparisons, or subtle wording. Do not invent citations or facts.
                Output format: Valid JSON parseable by json.loads() with keys: question, gold_answer (ground truth answer), reasoning_type, difficulty, task_type, source_id, source_quote, utility_rationale, quality_risk, claim_text, expected_verdict.
                """
            ),
            user_prompt=dedent(
                f"""\
                Create one novel benchmark question.
                Target reasoning_type: {target['reasoning_type']}
                Target difficulty: {target['difficulty']}
                Target task_type: {target['task_type']}

                Constraints:
                1) Ground question and answer in Sources below.
                2) Include a clear ground truth answer for evaluation.
                3) Must not repeat any item in Recent questions.
                4) Favor questions that reveal model weaknesses.
                5) source_quote should be a short exact quote from Sources.
                6) quality_risk should be one of: none, ambiguous, weak_gold, unsupported.

                Task types:
                - extractive_qa: answer is an explicit span or fact directly from source.
                - abstractive_qa: answer requires concise synthesis from multiple source facts.
                - claim_quality_check: create a concrete claim and ask if it is FAITHFUL or HALLUCINATED relative to source.

                Difficulty policy (must strictly follow):
                For extractive_qa:
                - difficulty 1: single-sentence fact; answer appears verbatim in source.
                - difficulty 3: requires resolving reference/condition within the same paragraph.
                - difficulty 5: requires cross-sentence understanding within one document.
                For abstractive_qa:
                - difficulty 1: summarize 2-3 obvious points.
                - difficulty 3: summarize 4-5 points dispersed across the text.
                - difficulty 5: infer implicit motivation or design rationale.
                For claim_quality_check:
                - difficulty 1: claim is clearly supported or clearly refuted.
                - difficulty 3: verdict requires understanding experiment conditions or subgroup constraints.
                - difficulty 5: use NOT_MENTIONED-style uncertainty or subtle contradiction.

                For claim_quality_check:
                - Put the claim text in claim_text.
                - expected_verdict must be exactly one of: FAITHFUL, HALLUCINATED, NOT_MENTIONED.
                - ground truth answer (gold_answer) should start with 'Verdict: ...' then short evidence rationale.
                For non-claim tasks, set claim_text to empty string and expected_verdict to NA.

                Recent questions:
                {recent_text}

                Sources:
                {context_text}
                """
            ),
            temperature=0.7 if temperature is None else temperature,
        )
        return result
