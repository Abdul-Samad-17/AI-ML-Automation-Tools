"""
Project 3: Document Question Answering System
Calder AI/ML Internship

Extracts text from a PDF, chunks it, and answers user queries
using TF-IDF keyword search with sentence-level scoring.
"""

import re
import math
import sys
from collections import Counter
from pathlib import Path


# ─────────────────────────────────────────────
# PDF Text Extraction
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file using PyMuPDF (fitz).
    Falls back to pypdf if fitz is not available.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        pages = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                pages.append(f"[Page {page_num}]\n{text.strip()}")
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        pass

    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {i}]\n{text.strip()}")
        return "\n\n".join(pages)
    except ImportError:
        raise ImportError(
            "No PDF library found. Install PyMuPDF: pip install pymupdf\n"
            "Or pypdf: pip install pypdf"
        )


# ─────────────────────────────────────────────
# Text Chunking
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[dict]:
    """
    Split text into overlapping word-level chunks.
    Returns list of {chunk_id, text, start_word, end_word}.
    """
    # Split into sentences first for better context
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_words = []
    chunk_id = 0

    for sentence in sentences:
        words = sentence.split()
        current_words.extend(words)

        if len(current_words) >= chunk_size:
            chunk_text_str = " ".join(current_words[:chunk_size])
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text_str,
            })
            chunk_id += 1
            current_words = current_words[chunk_size - overlap:]

    if current_words:
        chunks.append({
            "chunk_id": chunk_id,
            "text": " ".join(current_words),
        })

    return chunks


# ─────────────────────────────────────────────
# Text Preprocessing & TF-IDF
# ─────────────────────────────────────────────

STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "do", "i",
    "my", "me", "we", "our", "you", "your", "can", "will", "be", "am",
    "are", "was", "were", "has", "have", "had", "this", "that", "of",
    "for", "with", "about", "from", "or", "and", "not", "but", "so",
    "if", "as", "by", "its", "be", "been", "into", "such", "each",
}


def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]


def compute_tfidf(chunks: list[dict]) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Build TF-IDF vectors for all chunks. Returns (vectors, idf_map)."""
    tokenized = [preprocess(c["text"]) for c in chunks]
    N = len(tokenized)
    vocab = {w for toks in tokenized for w in toks}

    idf = {}
    for word in vocab:
        df = sum(1 for toks in tokenized if word in toks)
        idf[word] = math.log((N + 1) / (df + 1)) + 1

    vectors = []
    for toks in tokenized:
        tf = Counter(toks)
        total = len(toks) if toks else 1
        vec = {w: (c / total) * idf[w] for w, c in tf.items()}
        vectors.append(vec)

    return vectors, idf


def cosine_similarity(a: dict, b: dict) -> float:
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in b)
    na = math.sqrt(sum(v ** 2 for v in a.values()))
    nb = math.sqrt(sum(v ** 2 for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# ─────────────────────────────────────────────
# Document QA System
# ─────────────────────────────────────────────

class DocumentQASystem:
    SIMILARITY_THRESHOLD = 0.05
    TOP_K = 3  # number of relevant chunks to return

    def __init__(self):
        self.document_name: str = ""
        self.full_text: str = ""
        self.chunks: list[dict] = []
        self.vectors: list[dict] = []
        self.idf: dict[str, float] = {}
        self._loaded = False

    def load_document(self, pdf_path: str):
        """Load and index a PDF document."""
        print(f"Loading document: {pdf_path}")
        self.document_name = Path(pdf_path).name
        self.full_text = extract_text_from_pdf(pdf_path)
        self.chunks = chunk_text(self.full_text)
        self.vectors, self.idf = compute_tfidf(self.chunks)
        self._loaded = True
        print(f"✓ Loaded '{self.document_name}' | {len(self.chunks)} chunks indexed")

    def load_text(self, text: str, doc_name: str = "document"):
        """Load plain text directly (for testing without PDF)."""
        self.document_name = doc_name
        self.full_text = text
        self.chunks = chunk_text(text)
        self.vectors, self.idf = compute_tfidf(self.chunks)
        self._loaded = True
        print(f"✓ Loaded text document | {len(self.chunks)} chunks indexed")

    def answer(self, query: str) -> dict:
        """
        Answer a user query from the loaded document.

        Returns:
            dict with: answer, excerpts, scores, found
        """
        if not self._loaded:
            return {"found": False, "answer": "No document loaded. Call load_document() first.", "excerpts": []}

        query_tokens = preprocess(query)
        if not query_tokens:
            return {"found": False, "answer": "Please enter a valid query.", "excerpts": []}

        # Query TF-IDF vector
        tf = Counter(query_tokens)
        total = len(query_tokens)
        q_vec = {w: (c / total) * self.idf.get(w, 0) for w, c in tf.items()}

        # Score chunks
        scores = [(i, cosine_similarity(q_vec, vec)) for i, vec in enumerate(self.vectors)]
        scores.sort(key=lambda x: x[1], reverse=True)

        top = [(i, s) for i, s in scores[:self.TOP_K] if s >= self.SIMILARITY_THRESHOLD]

        if not top:
            return {
                "found": False,
                "answer": f"No relevant content found in '{self.document_name}' for your query.",
                "excerpts": [],
            }

        excerpts = []
        for rank, (idx, score) in enumerate(top, 1):
            excerpts.append({
                "rank": rank,
                "score": round(score, 4),
                "text": self.chunks[idx]["text"].strip(),
            })

        best_excerpt = excerpts[0]["text"]
        answer_summary = f"Based on '{self.document_name}':\n\n{best_excerpt}"

        return {
            "found": True,
            "answer": answer_summary,
            "excerpts": excerpts,
            "query": query,
        }

    def run_cli(self):
        """Interactive CLI for document QA."""
        if not self._loaded:
            pdf_path = input("Enter path to PDF document: ").strip()
            try:
                self.load_document(pdf_path)
            except Exception as e:
                print(f"Error loading document: {e}")
                return

        print(f"\n{'='*60}")
        print(f"  Document QA | Loaded: {self.document_name}")
        print(f"  Type 'exit' to quit")
        print(f"{'='*60}")

        while True:
            try:
                query = input("\nYour Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not query:
                continue
            if query.lower() in {"exit", "quit"}:
                break

            result = self.answer(query)
            if result["found"]:
                print(f"\n{'─'*50}")
                for ex in result["excerpts"]:
                    print(f"\n[Excerpt #{ex['rank']} | Score: {ex['score']}]")
                    print(ex["text"][:400] + ("..." if len(ex["text"]) > 400 else ""))
                print(f"{'─'*50}")
            else:
                print(f"\n{result['answer']}")

        print("\nGoodbye!")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    qa = DocumentQASystem()
    if len(sys.argv) > 1:
        try:
            qa.load_document(sys.argv[1])
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    qa.run_cli()
