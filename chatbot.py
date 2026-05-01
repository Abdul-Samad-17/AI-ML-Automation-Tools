"""
Project 1: FAQ Chatbot System
Calder AI/ML Internship

Uses TF-IDF vectorization + cosine similarity for query matching.
"""

import json
import re
import math
from collections import Counter
from pathlib import Path


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_faq_data(path: str = "faq_data.json") -> list[dict]:
    """Load FAQ dataset from JSON file."""
    faq_path = Path(__file__).parent / path
    with open(faq_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["faqs"]


# ─────────────────────────────────────────────
# Text Preprocessing
# ─────────────────────────────────────────────

STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "do",
    "i", "my", "me", "we", "our", "you", "your", "can", "how",
    "what", "when", "where", "why", "which", "who", "will", "be",
    "are", "was", "were", "has", "have", "had", "this", "that",
    "of", "for", "with", "about", "from", "or", "and", "not",
}


def preprocess(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop words, return tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens


# ─────────────────────────────────────────────
# TF-IDF Vectorizer (built from scratch)
# ─────────────────────────────────────────────

class TFIDFVectorizer:
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.corpus_tokens: list[list[str]] = []

    def fit(self, documents: list[str]):
        """Build vocabulary and IDF scores from documents."""
        self.corpus_tokens = [preprocess(doc) for doc in documents]
        # Build vocab
        all_words = {w for tokens in self.corpus_tokens for w in tokens}
        self.vocab = {w: i for i, w in enumerate(sorted(all_words))}
        # Compute IDF
        N = len(self.corpus_tokens)
        for word in self.vocab:
            df = sum(1 for tokens in self.corpus_tokens if word in tokens)
            self.idf[word] = math.log((N + 1) / (df + 1)) + 1  # smooth

    def transform(self, tokens: list[str]) -> dict[str, float]:
        """Return TF-IDF vector as sparse dict."""
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vector = {}
        for word, count in tf.items():
            if word in self.idf:
                vector[word] = (count / total) * self.idf[word]
        return vector

    def fit_transform_corpus(self) -> list[dict[str, float]]:
        """Return TF-IDF vectors for the fitted corpus."""
        return [self.transform(tokens) for tokens in self.corpus_tokens]


def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Compute cosine similarity between two sparse TF-IDF vectors."""
    dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in vec_b)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────
# FAQ Chatbot
# ─────────────────────────────────────────────

class FAQChatbot:
    SIMILARITY_THRESHOLD = 0.08  # minimum score to return a match

    def __init__(self, faq_path: str = "faq_data.json"):
        self.faqs = load_faq_data(faq_path)
        self._build_index()

    def _build_index(self):
        """Build TF-IDF index from FAQ questions + keywords."""
        # Combine question text + keywords into one searchable document per FAQ
        documents = []
        for faq in self.faqs:
            combined = faq["question"] + " " + " ".join(faq.get("keywords", []))
            documents.append(combined)

        self.vectorizer = TFIDFVectorizer()
        self.vectorizer.fit(documents)
        self.doc_vectors = self.vectorizer.fit_transform_corpus()

    def get_response(self, user_query: str) -> dict:
        """
        Find the best matching FAQ for a user query.

        Returns:
            dict with keys: answer, question, score, found
        """
        query_tokens = preprocess(user_query)

        if not query_tokens:
            return {
                "found": False,
                "answer": "I couldn't understand your query. Could you please rephrase it?",
                "question": None,
                "score": 0.0,
            }

        query_vector = self.vectorizer.transform(query_tokens)
        scores = [cosine_similarity(query_vector, dv) for dv in self.doc_vectors]

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_score = scores[best_idx]

        if best_score < self.SIMILARITY_THRESHOLD:
            return {
                "found": False,
                "answer": (
                    "I'm sorry, I couldn't find a relevant answer to your question. "
                    "Please contact our support team at support@calder.ai or call +92 349 5904995."
                ),
                "question": None,
                "score": round(best_score, 4),
            }

        matched_faq = self.faqs[best_idx]
        return {
            "found": True,
            "answer": matched_faq["answer"],
            "question": matched_faq["question"],
            "score": round(best_score, 4),
        }

    def run_cli(self):
        """Interactive command-line chatbot."""
        print("=" * 60)
        print("  Calder FAQ Chatbot  |  Type 'exit' to quit")
        print("=" * 60)
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "bye"}:
                print("Bot: Goodbye! Have a great day.")
                break

            result = self.get_response(user_input)
            if result["found"]:
                print(f"\nBot: {result['answer']}")
                print(f"     (Matched: \"{result['question']}\" | Score: {result['score']})")
            else:
                print(f"\nBot: {result['answer']}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    bot = FAQChatbot()
    bot.run_cli()
