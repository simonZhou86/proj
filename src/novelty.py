import re
from typing import Any
from textwrap import dedent

from .llm_client import OpenAICompatibleClient


def tokenize(text: str) -> set[str]:
    '''
    sample tokenizer
    '''
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def lexical_similarity(a: str, b: str) -> float:
    '''
    per-token similarity
    '''
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta.intersection(tb))
    union = len(ta.union(tb))
    return inter / union if union > 0 else 0.0


def is_novel(
    candidate_question: str,
    history: list[dict[str, Any]],
    client: OpenAICompatibleClient,
    lexical_threshold: float = 0.8,
    llm_window: int = 10,
) -> tuple[bool, dict[str, Any]]:
    '''
    llm as a judge approach for filtering the question further after a lexical similarity check.
    The LLM will be asked to determine whether the candidate question is materially and semantically novel relative to the provided history
    '''
    if not history:
        return True, {"method": "cold_start"}

    scored: list[tuple[float, str]] = []
    for row in history:
        old_question = row.get("question", "")
        if not old_question:
            continue
        sim = lexical_similarity(candidate_question, old_question)
        scored.append((sim, old_question))

    if not scored:
        return True, {"method": "cold_start_after_filter"}

    scored.sort(key=lambda x: x[0], reverse=True)
    best_sim, best_match = scored[0]
    if best_sim >= lexical_threshold:
        return False, {
            "method": "lexical",
            "max_similarity": round(best_sim, 4),
            "closest_question": best_match,
        }

    top_refs = [item[1] for item in scored[:llm_window]]
    verdict = client.chat_json(
        system_prompt=dedent(
            """\
            Your role: You are a strict judge that checkes the novelty of a candidate question comparing to historical questions.
            Your task: Determine whether the candidate is materially and semantically novel relative to the provided history, focusing on meaning and intent rather than surface wording.
            Rules/constraints: Use only the provided questions. Consider paraphrases, reordered phrasing, and minor wording changes as not novel. If the candidate is materially the same as any previous question, set is_novel=false. confidence must be in [0,1]. Do not assume domain facts not present in the questions.
            Output format: Valid JSON parseable by json.loads() with keys: is_novel (bool), confidence (0-1), rationale (string).
            """
        ),
        user_prompt=dedent(
            f"""\
            Here is the candidate question and the list of previous questions.

            Candidate question:
            {candidate_question}

            Previous questions:
            {"".join([f"- {ref}\n" for ref in top_refs]).rstrip()}
            """
        ),
        temperature=0.0,
    )

    return bool(verdict.get("is_novel", False)), {
        "method": "llm",
        "max_similarity": round(best_sim, 4),
        "verdict": verdict,
    }
