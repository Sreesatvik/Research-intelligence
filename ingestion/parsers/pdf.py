import re

import pymupdf

from core.schemas import Chunk
from ingestion.chunking import chunk_text


CAPTION_RE = re.compile(r"^(table|figure|fig\.)\s*(\d+)\s*[:.]", re.I)


def clean_text(raw: str) -> str:
    """Repair PDF line-wrapping: join hyphenated words, turn line breaks into spaces."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)  # "lan-\nguage" -> "language"
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_pdf(path: str | bytes, document_id: str,
              chunk_size: int = 100, overlap: int = 20) -> list[Chunk]:
    """Turn one PDF into text chunks and table chunks, each tagged with its page.
    `path` may also be the file's raw bytes, so an upload can be checked before anything is written to disk."""
    chunks: list[Chunk] = []
    doc = pymupdf.open(stream=path, filetype="pdf") if isinstance(path, bytes) else pymupdf.open(path)
    for page_index, page in enumerate(doc):
        page_no = page_index + 1
        text = clean_text(page.get_text())
        if text:
            chunks.extend(chunk_text(document_id, text, chunk_size, overlap,
                                     page=page_no, start_position=len(chunks)))
        for table in page.find_tables().tables:
            markdown = table.to_markdown().strip()
            if len(markdown) < 40:  # skip empty or tiny false positives
                continue
            position = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{document_id}#{position}",
                document_id=document_id,
                text=markdown,
                position=position,
                content_type="table",
                page=page_no,
            ))
        for block in page.get_text("blocks"):  # captions: "Table 2: ...", "Figure 1: ..."
            caption = clean_text(block[4])
            m = CAPTION_RE.match(caption)
            if not m:
                continue
            kind = "figure" if m.group(1).lower().startswith("fig") else "table"
            position = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{document_id}#{position}",
                document_id=document_id,
                text=caption,
                position=position,
                content_type="caption",
                page=page_no,
                metadata={"label": f"{kind} {m.group(2)}"},
            ))
    return chunks


def render_page(path: str, page_no: int, dpi: int = 110) -> bytes:
    """Render one PDF page (1-based) to PNG bytes, for a vision model."""
    page = pymupdf.open(path)[page_no - 1]
    return page.get_pixmap(dpi=dpi).tobytes("png")


def render_table_region(path: str, page_no: int, label: str, dpi: int = 150) -> bytes | None:
    """Render only the table named by `label` (e.g. 'table 2'): its caption plus the
    detected table pieces right next to it. Returns None if we cannot locate it."""
    page = pymupdf.open(path)[page_no - 1]
    caption = None
    for block in page.get_text("blocks"):
        m = CAPTION_RE.match(clean_text(block[4]))
        if m and f"{'figure' if m.group(1).lower().startswith('fig') else 'table'} {m.group(2)}" == label:
            caption = pymupdf.Rect(block[:4])
            break
    if caption is None:
        return None
    pieces = [pymupdf.Rect(t.bbox) for t in page.find_tables().tables]
    # tables normally sit just above their caption; if none are, try just below it
    near = [r for r in pieces if caption.y0 - 300 <= r.y0 and r.y1 <= caption.y0 + 5]
    if not near:
        near = [r for r in pieces if caption.y1 - 5 <= r.y0 <= caption.y1 + 300]
    if not near:
        return None
    region = caption
    for r in near:
        region = region | r  # union of rectangles
    region = (region + (-10, -10, 10, 10)) & page.rect
    return page.get_pixmap(dpi=dpi, clip=region).tobytes("png")
