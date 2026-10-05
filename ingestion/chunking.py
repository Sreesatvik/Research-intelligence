from core.schemas import Chunk


def chunk_text(document_id: str, text: str,
               chunk_size: int = 100, overlap: int = 20,
               page: int | None = None, start_position: int = 0) -> list[Chunk]:
    """Split text into overlapping windows of `chunk_size` words."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for i, start in enumerate(range(0, len(words), step)):
        position = start_position + i
        window = words[start:start + chunk_size]
        chunks.append(Chunk(
            chunk_id=f"{document_id}#{position}",
            document_id=document_id,
            text=" ".join(window),
            position=position,
            page=page,
        ))
        if start + chunk_size >= len(words):  # last window reached the end
            break
    return chunks
