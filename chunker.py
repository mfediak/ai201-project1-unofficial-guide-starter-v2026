"""
Stage 2 of the pipeline: splitting documents into chunks.

⚠️ THIS IS THE FILE YOU CHANGE IN MILESTONE 3.

`split_documents` below is deliberately plain. It cuts every document into
fixed-size pieces with a fixed overlap and pays no attention to where sentences
or paragraphs end. It works, and it is not good.

On a corpus of short posts it may not cut anything at all: `campus_life` comes
out as 88 documents and 88 chunks, because almost nothing in it reaches 800
characters. That is the baseline, not a bug — Milestone 3 is where you decide
whether one post should stay one chunk.

Your job in Milestone 3 is to replace the *body* of `split_documents` with a
strategy that fits the documents you actually read in Milestone 1. Keep the
name and the shape of what it returns — the rest of the pipeline calls it, and
your README has to name the function that produced your chunks.

If you get stuck for 30 minutes, `fallback_split` is the original. Switch back
to it, write down what you saw, and move on. That's a real observation about
your pipeline, not giving up.
"""

import re
from dataclasses import dataclass

import config
from ingest import Document

# Splits after a sentence-ending mark and the whitespace that follows it, so
# the punctuation stays with the sentence it closes.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """One piece of one document."""

    text: str
    source: str        # which file it came from
    index: int         # which chunk within that file, starting at 0
    produced_by: str   # the function that made it — cite this in your README

    @property
    def label(self) -> str:
        return f"{self.source}#{self.index}"


def fallback_split(
    documents: list[Document],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    The starter's original chunker. Fixed-size character windows with overlap.

    Keep this function. Milestone 3's stop rule points back at it, and having
    something to compare your own strategy against is useful in unit 2.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    if overlap >= chunk_size:
        raise ValueError("overlap has to be smaller than chunk_size")

    chunks: list[Chunk] = []
    for doc in documents:
        start = 0
        index = 0
        while start < len(doc.text):
            piece = doc.text[start : start + chunk_size].strip()
            if piece:
                chunks.append(
                    Chunk(
                        text=piece,
                        source=doc.source,
                        index=index,
                        produced_by="chunker.py::fallback_split",
                    )
                )
                index += 1
            start += chunk_size - overlap

    return chunks


def _split_paragraphs(text: str) -> list[str]:
    """Paragraphs, in order, as separated by blank lines."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Sentences, in order. Never used to cut inside a sentence."""
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def _carry_overlap(units: list[str], overlap: int) -> list[str]:
    """
    Trailing whole units (sentences or short paragraphs) from the end of a
    chunk, kept under `overlap` characters, to seed the next chunk with.

    Only ever takes whole units — the overlap is never a mid-sentence cut.
    """
    if overlap <= 0:
        return []
    carried: list[str] = []
    length = 0
    for unit in reversed(units):
        added = len(unit) + (1 if carried else 0)
        if carried and length + added > overlap:
            break
        carried.insert(0, unit)
        length += added
    return carried


def split_documents(documents: list[Document]) -> list[Chunk]:
    """
    Split documents on paragraph breaks first, falling back to sentences.

    Each document becomes a list of "units" — whole paragraphs where a
    paragraph fits inside chunk_size on its own, or its individual sentences
    where it doesn't. Units are then packed into chunks up to chunk_size,
    never splitting a unit apart, so every chunk holds at least one complete
    sentence and paragraph breaks are respected wherever they fit. Overlap
    carries whole trailing sentences (or paragraphs) into the next chunk
    rather than a raw character slice.

    campus_life's posts are all shorter than CHUNK_SIZE, so in practice this
    still yields one chunk per document — the strategy only starts doing real
    work on documents long enough to need it.
    """
    chunk_size = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP

    chunks: list[Chunk] = []
    for doc in documents:
        units: list[str] = []
        for paragraph in _split_paragraphs(doc.text):
            if len(paragraph) <= chunk_size:
                units.append(paragraph)
            else:
                units.extend(_split_sentences(paragraph))

        pieces: list[str] = []
        current: list[str] = []
        current_len = 0

        for unit in units:
            added_len = len(unit) + (1 if current else 0)
            if current and current_len + added_len > chunk_size:
                pieces.append("\n\n".join(current))
                current = _carry_overlap(current, overlap)
                current_len = len("\n\n".join(current)) if current else 0
                added_len = len(unit) + (1 if current else 0)
            current.append(unit)
            current_len += added_len

        if current:
            pieces.append("\n\n".join(current))

        for i, text in enumerate(pieces):
            chunks.append(
                Chunk(
                    text=text.strip(),
                    source=doc.source,
                    index=i,
                    produced_by="chunker.py::split_documents",
                )
            )

    return chunks


def describe(chunks: list[Chunk]) -> str:
    """A one-line summary, printed after indexing."""
    if not chunks:
        return "0 chunks"
    lengths = [len(c.text) for c in chunks]
    return (
        f"{len(chunks)} chunks, "
        f"{sum(lengths) // len(lengths)} characters on average "
        f"(shortest {min(lengths)}, longest {max(lengths)}), "
        f"produced by {chunks[0].produced_by}"
    )


if __name__ == "__main__":
    from ingest import load_documents

    chunks = split_documents(load_documents())
    print(describe(chunks))
