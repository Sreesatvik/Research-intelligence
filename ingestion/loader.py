from pathlib import Path

from core.schemas import Chunk
from ingestion.chunking import chunk_text


def load_corpus(folder: str, chunk_size: int = 100, overlap: int = 20) -> list[Chunk]:
    """Read every .txt file in `folder` and return all chunks."""
    chunks = []
    for path in sorted(Path(folder).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_text(path.stem, text, chunk_size, overlap))
    return chunks


def load_pdfs(folder: str, chunk_size: int = 100, overlap: int = 20) -> list[Chunk]:
    """Parse every PDF in `folder` into text and table chunks."""
    from ingestion.parsers.pdf import parse_pdf

    chunks = []
    for path in sorted(Path(folder).glob("*.pdf")):
        chunks.extend(parse_pdf(str(path), path.stem, chunk_size, overlap))
    return chunks
