from textwrap import dedent

from .llm_client import OpenAICompatibleClient


class QuestionAnswerer:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        '''
        Initialize the answerer with an LLM client.
        '''
        self.client = client

    def answer(self, question: str, contexts: list[dict[str, str]], task_type: str = "") -> str:
        '''
        Generate an answer using the question and provided contexts.
        Args:
            question: The question to answer.
            contexts: A list of dicts, each with keys 'source_id', 'chunk_id', 'title', 'text'.
            task_type: The type of question, e.g., "claim_quality_check", which may affect the output format.
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
        output_rule = (
            "Output format: Verdict: FAITHFUL/HALLUCINATED/NOT_MENTIONED, then one short evidence rationale."
            if task_type == "claim_quality_check"
            else "Output format: A clear, concise answer grounded only in the provided context, and a citation of the content used in source. Format: Answer: <your answer here> Reference: <source citation here>."
        )
        return self.client.chat_text(
            system_prompt=dedent(
                f"""\
                Your role: You are a life sciences practitioner with broad knowledge in life sciences and medicine capable of answering a wide range of questions accurately.
                Your task: Answer the user's question using only the provided context, prioritizing factual accuracy and faithful grounding.
                Rules/constraints: Use only the provided context. Do not use outside knowledge or assumptions. If evidence is missing or ambiguous, respond with "insufficient evidence." Do not cite sources that are not present in the context. Do not speculate or invent data. Keep the answer focused on the question and relevant details only.
                {output_rule}
                """
            ),
            user_prompt=dedent(
                f"""\
                Question:
                {question}

                Context:
                {context_text}
                """
            ),
            temperature=0,
        )
