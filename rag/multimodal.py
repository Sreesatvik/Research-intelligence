from core.interfaces import LLM, VLM, Retriever
from core.schemas import Chunk, RAGAnswer
from ingestion.parsers.pdf import render_page, render_table_region
from rag.page_locator import LABEL_RE, PageLocator
from rag.pipeline import RAG
from rag.router import route


class MultimodalRAG:
    def __init__(self, retriever: Retriever, llm: LLM, vlm: VLM,
                 chunks: list[Chunk], pdf_dir: str = "data/pdfs"):
        self.locator = PageLocator(chunks)
        self.retriever = retriever
        self.text_rag = RAG(retriever, llm)
        self.vlm = vlm
        self.pdf_dir = pdf_dir

    def ask(self, question: str) -> tuple[str, RAGAnswer]:
        """Return (route used, answer)."""
        if route(question) == "text":
            return "text", self.text_rag.ask(question)

        results = self.retriever.search(question, k=5)
        located = self.locator.locate(question, results) if results else None
        if located is None or results[0].score < self.text_rag.min_score:
            return "text", self.text_rag.ask(question)  # nothing to look at: fall back
        document_id, page = located

        pdf_path = f"{self.pdf_dir}/{document_id}.pdf"
        image = None
        m = LABEL_RE.search(question)
        if m and not m.group(1).lower().startswith("fig"):  # a named table: crop to it
            image = render_table_region(pdf_path, page, f"table {m.group(2)}")
        if image is None:
            image = render_page(pdf_path, page)
        prompt = (f"Answer using only this page image. If the page does not contain the "
                  f"answer, say \"I don't know based on the provided sources.\"\n\n"
                  f"Question: {question}")
        answer = self.vlm.ask_image(image, prompt)
        cited = [f"{document_id}#page{page}"]  # source locator = the page
        return "visual", RAGAnswer(question, answer, cited, results)
