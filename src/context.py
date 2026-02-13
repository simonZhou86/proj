from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ContextDocument:
    source_id: str
    title: str
    text: str
    chunks: list[str]


class ContextLibrary:
    '''
    load documents and chunk them into passages.
    '''
    def __init__(self, docs: list[ContextDocument], seed: int | None = None) -> None:
        """Initialize with pre-chunked documents and a random seed."""
        if not docs:
            raise ValueError("No context documents found. Add .txt or .md files in context_dir.")
        self.docs = docs
        self.random = random.Random(seed)

    @classmethod
    def from_dir(cls, context_dir: str, max_chars_per_chunk: int = 1800, seed: int | None = None) -> "ContextLibrary":
        '''
        Load documents from a directory and build a chunked context library.
        '''
        root = Path(context_dir)
        if not root.exists():
            raise ValueError(f"context_dir does not exist: {context_dir}")
        docs: list[ContextDocument] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
                continue
            text = path.read_text(encoding="utf-8")
            chunks = _chunk_text(text=text, max_chars=max_chars_per_chunk)
            docs.append(
                ContextDocument(
                    source_id=str(path.relative_to(root)),
                    title=path.stem,
                    text=text,
                    chunks=chunks,
                )
            )
        return cls(docs=docs, seed=seed)

    def sample(self, n_docs: int) -> list[dict[str, str]]:
        '''
        we can also select random chunks
        '''
        selected_docs = self.docs if n_docs >= len(self.docs) else self.random.sample(self.docs, k=n_docs)
        contexts: list[dict[str, str]] = []
        for doc in selected_docs:
            chunks = doc.chunks
            for index, chunk in enumerate(chunks):
                contexts.append(
                    {
                        "source_id": doc.source_id,
                        "title": doc.title,
                        "chunk_id": str(index),
                        "text": chunk,
                    }
                )
        return contexts


def _chunk_text(text: str, max_chars: int) -> list[str]:
        '''
        chunk text into pieces of max_chars with overlap, trying to split on sentence boundaries to avoid cut-in word and
        lose semantic coherence.
        '''
        normalized = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        if not normalized:
            return [""]
        if max_chars <= 0:
            return [normalized]
        overlap = 100 if max_chars > 100 else max(0, max_chars - 1)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
        )
        chunks = splitter.split_text(normalized)
        return chunks if chunks else [""]


def _naive_chunk_text(text: str, max_chars: int) -> list[str]:
        '''
        chunk text into pieces of max_chars with overlap, trying to split on sentence boundaries to avoid cut-in word and
        lose semantic coherence.

        can be improved by using langchain
        '''
        normalized = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        if not normalized:
            return [""]
        if max_chars <= 0:
            return [normalized]
        # Split into paragraphs, then sentences using a simple punctuation-based heuristic.
        paragraphs = [p.strip() for p in normalized.split("\n") if p.strip()]
        sentences: list[str] = []
        for paragraph in paragraphs:
            parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
            if parts:
                sentences.extend(parts)
            else:
                sentences.append(paragraph)
        if not sentences:
            return [normalized[:max_chars]]
        # If any sentence exceeds max_chars, split it into fixed-size chunks.
        normalized_sentences: list[str] = []
        for sentence in sentences:
            if len(sentence) <= max_chars:
                normalized_sentences.append(sentence)
                continue
            index = 0
            while index < len(sentence):
                normalized_sentences.append(sentence[index : index + max_chars])
                index += max_chars
        sentences = normalized_sentences
        overlap_sentences = 1
        pieces: list[str] = []
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            sentence_len = len(sentence) + (1 if current else 0)
            if current and current_len + sentence_len > max_chars:
                pieces.append(" ".join(current))
                current = current[-overlap_sentences:] if overlap_sentences > 0 else []
                current_len = sum(len(s) for s in current) + (len(current) - 1 if len(current) > 1 else 0)
                if current and current_len + sentence_len > max_chars:
                    current = []
                    current_len = 0
            current.append(sentence)
            current_len += sentence_len
        if current:
            pieces.append(" ".join(current))
        return pieces
