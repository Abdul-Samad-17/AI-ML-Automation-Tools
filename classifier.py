"""
Project 2: Email Classification System
Calder AI/ML Internship

Uses TF-IDF + Naive Bayes classifier with training data.
Falls back to keyword/rule-based classification when confidence is low.
Categories: Inquiry, Complaint, Feedback, Spam, Support Request
"""

import re
import math
from collections import Counter, defaultdict


# ─────────────────────────────────────────────
# Categories
# ─────────────────────────────────────────────

CATEGORIES = ["Inquiry", "Complaint", "Feedback", "Spam", "Support Request"]

# Training data: (email_text, label)
TRAINING_DATA = [
    # Inquiry
    ("What are your pricing plans? I'd like to know more about the available options.", "Inquiry"),
    ("Can you tell me more about your services and features?", "Inquiry"),
    ("Do you offer a free trial? How long does it last?", "Inquiry"),
    ("I would like to know if you support integration with Slack.", "Inquiry"),
    ("What programming languages are supported by your API?", "Inquiry"),
    ("Is there a mobile app available for your platform?", "Inquiry"),
    ("How many users can be added to a single account?", "Inquiry"),
    ("What is the difference between the starter and professional plan?", "Inquiry"),
    ("Do you offer annual billing discounts?", "Inquiry"),
    ("Could you send me more information about enterprise pricing?", "Inquiry"),

    # Complaint
    ("Your service is completely broken and I cannot log in for the past two days.", "Complaint"),
    ("I am very frustrated. My account was charged twice this month and nobody is helping.", "Complaint"),
    ("This is unacceptable! The system has been down and I am losing business.", "Complaint"),
    ("I have been waiting for a response for a week and no one has replied to me.", "Complaint"),
    ("The product is terrible and not working as advertised. I want a refund immediately.", "Complaint"),
    ("Your customer service is awful. I have called three times with no resolution.", "Complaint"),
    ("I am extremely disappointed with the quality of your software.", "Complaint"),
    ("The export feature is broken and has been for weeks. This is unacceptable.", "Complaint"),
    ("I demand a full refund. This product does not work at all.", "Complaint"),
    ("Terrible experience. The dashboard crashes every time I open it.", "Complaint"),

    # Feedback
    ("I really enjoy using your platform. The interface is intuitive and clean.", "Feedback"),
    ("Great product overall! One suggestion: it would be nice to have dark mode.", "Feedback"),
    ("The onboarding process was smooth and the documentation is excellent.", "Feedback"),
    ("I think adding a bulk export feature would greatly improve the workflow.", "Feedback"),
    ("Your team's support was wonderful. Very helpful and quick to respond.", "Feedback"),
    ("I love the new dashboard update. Much better than before!", "Feedback"),
    ("The product is good but the mobile version could use improvement.", "Feedback"),
    ("Just wanted to say the recent update was a huge improvement. Well done!", "Feedback"),
    ("It would be great if you could add keyboard shortcuts to the editor.", "Feedback"),
    ("Very satisfied with the service. Would definitely recommend to others.", "Feedback"),

    # Spam
    ("Congratulations! You have won a $1000 gift card. Click here to claim now!", "Spam"),
    ("Make money fast from home! Earn $5000 per week with our proven system.", "Spam"),
    ("FREE OFFER LIMITED TIME ONLY!!! Act now and get rich quick!", "Spam"),
    ("Click this link to get a free iPhone. No strings attached. Hurry up!", "Spam"),
    ("Buy cheap medications online. No prescription needed. Best prices guaranteed.", "Spam"),
    ("You have been selected for a special offer. Claim your prize today!", "Spam"),
    ("Investment opportunity with 300% guaranteed returns. Join now!", "Spam"),
    ("URGENT: Your account has been compromised. Click here immediately to verify.", "Spam"),

    # Support Request
    ("I need help setting up two-factor authentication on my account.", "Support Request"),
    ("Please help me recover my account. I cannot access it anymore.", "Support Request"),
    ("How do I add a new team member to my workspace?", "Support Request"),
    ("I need assistance configuring the API keys for my integration.", "Support Request"),
    ("Can someone help me with importing data from my old system?", "Support Request"),
    ("I accidentally deleted a project. Is there a way to restore it?", "Support Request"),
    ("I need help understanding how to use the reporting feature.", "Support Request"),
    ("My payment failed but I was still charged. Can someone assist?", "Support Request"),
    ("Please help me set up the webhook for my application.", "Support Request"),
    ("I need technical support. The sync feature is not working properly.", "Support Request"),
]

# Keyword rules for fallback / boosting confidence
KEYWORD_RULES = {
    "Complaint": [
        "terrible", "awful", "broken", "unacceptable", "frustrated", "angry",
        "refund", "disappointed", "useless", "scam", "horrible", "worst",
        "never again", "demand", "complain", "issue", "problem", "not working",
        "crashed", "error", "failed", "charged twice",
    ],
    "Inquiry": [
        "what is", "how does", "can you tell", "do you offer", "is there",
        "what are", "how many", "pricing", "plans", "features", "trial",
        "how long", "difference between", "more information", "learn more",
    ],
    "Feedback": [
        "great", "love", "excellent", "suggest", "recommendation", "improve",
        "would be nice", "it would", "satisfied", "happy", "good", "fantastic",
        "appreciate", "enjoyed", "well done", "thumbs up",
    ],
    "Spam": [
        "click here", "free offer", "congratulations you won", "make money",
        "guaranteed", "claim your", "act now", "limited time", "earn",
        "prize", "gift card", "no prescription", "buy cheap",
    ],
    "Support Request": [
        "help me", "i need help", "can someone", "please assist", "how do i",
        "unable to", "cannot access", "not working", "technical support",
        "recover", "restore", "configure", "set up", "setup",
    ],
}


# ─────────────────────────────────────────────
# Text Preprocessing
# ─────────────────────────────────────────────

STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "do", "i",
    "my", "me", "we", "our", "you", "your", "can", "will", "be", "am",
    "are", "was", "were", "has", "have", "had", "this", "that", "of",
    "for", "with", "about", "from", "or", "and", "not", "but", "so",
    "if", "as", "by", "just", "more", "than",
}


def preprocess(text: str) -> list[str]:
    """Lowercase, remove special chars, filter stop words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return tokens


# ─────────────────────────────────────────────
# Naive Bayes Classifier
# ─────────────────────────────────────────────

class NaiveBayesClassifier:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha  # Laplace smoothing
        self.class_priors: dict[str, float] = {}
        self.word_probs: dict[str, dict[str, float]] = defaultdict(dict)
        self.vocab: set[str] = set()

    def fit(self, data: list[tuple[str, str]]):
        """Train on list of (text, label) pairs."""
        class_word_counts: dict[str, Counter] = defaultdict(Counter)
        class_counts: Counter = Counter()

        for text, label in data:
            tokens = preprocess(text)
            class_word_counts[label].update(tokens)
            class_counts[label] += 1
            self.vocab.update(tokens)

        total_docs = sum(class_counts.values())
        V = len(self.vocab)

        for label in CATEGORIES:
            count = class_counts.get(label, 0)
            self.class_priors[label] = math.log((count + self.alpha) / (total_docs + self.alpha * len(CATEGORIES)))
            total_words = sum(class_word_counts[label].values())
            for word in self.vocab:
                word_count = class_word_counts[label].get(word, 0)
                self.word_probs[label][word] = math.log(
                    (word_count + self.alpha) / (total_words + self.alpha * V)
                )

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Return predicted class and log-probability scores."""
        tokens = preprocess(text)
        scores = {}
        for label in CATEGORIES:
            score = self.class_priors[label]
            for token in tokens:
                if token in self.vocab:
                    score += self.word_probs[label][token]
            scores[label] = score

        # Convert log probs to relative probabilities (softmax-like)
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        probs = {k: round(v / total, 4) for k, v in exp_scores.items()}

        predicted = max(probs, key=probs.get)
        return predicted, probs


# ─────────────────────────────────────────────
# Rule-Based Boosting
# ─────────────────────────────────────────────

def keyword_boost(text: str, ml_prediction: str, ml_probs: dict) -> tuple[str, dict]:
    """
    Boost ML prediction confidence using keyword rules.
    If keywords strongly indicate a different category, override.
    """
    text_lower = text.lower()
    keyword_hits: dict[str, int] = Counter()

    for category, keywords in KEYWORD_RULES.items():
        for kw in keywords:
            if kw in text_lower:
                keyword_hits[category] += 1

    if not keyword_hits:
        return ml_prediction, ml_probs

    top_keyword_cat = keyword_hits.most_common(1)[0][0]
    top_hits = keyword_hits[top_keyword_cat]

    # Override ML if keyword evidence is strong (3+ hits) and different from ML
    if top_hits >= 3 and top_keyword_cat != ml_prediction:
        ml_probs[top_keyword_cat] = min(ml_probs.get(top_keyword_cat, 0) + 0.25, 0.99)
        # Re-normalize
        total = sum(ml_probs.values())
        ml_probs = {k: round(v / total, 4) for k, v in ml_probs.items()}
        return top_keyword_cat, ml_probs

    return ml_prediction, ml_probs


# ─────────────────────────────────────────────
# Email Classifier (main interface)
# ─────────────────────────────────────────────

class EmailClassifier:
    def __init__(self):
        self.model = NaiveBayesClassifier()
        self.model.fit(TRAINING_DATA)

    def classify(self, subject: str, body: str) -> dict:
        """
        Classify an email given subject and body.

        Returns:
            dict with: category, confidence, all_scores, subject, snippet
        """
        full_text = f"{subject} {body}"
        predicted, probs = self.model.predict(full_text)
        predicted, probs = keyword_boost(full_text, predicted, probs)

        confidence = probs[predicted]

        return {
            "category": predicted,
            "confidence": confidence,
            "confidence_pct": f"{confidence * 100:.1f}%",
            "all_scores": probs,
            "subject": subject,
            "snippet": body[:100] + ("..." if len(body) > 100 else ""),
        }

    def classify_batch(self, emails: list[dict]) -> list[dict]:
        """Classify multiple emails at once. Each email: {subject, body}"""
        return [self.classify(e.get("subject", ""), e.get("body", "")) for e in emails]

    def run_cli(self):
        """Interactive CLI for email classification."""
        print("=" * 60)
        print("  Calder Email Classifier  |  Type 'exit' to quit")
        print("=" * 60)
        while True:
            try:
                subject = input("\nEmail Subject: ").strip()
                if subject.lower() in {"exit", "quit"}:
                    break
                body = input("Email Body: ").strip()
                if not subject and not body:
                    print("Please provide at least a subject or body.")
                    continue

                result = self.classify(subject, body)
                print(f"\n{'─'*40}")
                print(f"Category   : {result['category']}")
                print(f"Confidence : {result['confidence_pct']}")
                print(f"All Scores : {result['all_scores']}")
                print(f"{'─'*40}")
            except (EOFError, KeyboardInterrupt):
                break
        print("\nGoodbye!")


# ─────────────────────────────────────────────
# Demo emails for testing
# ─────────────────────────────────────────────

DEMO_EMAILS = [
    {
        "subject": "Question about pricing",
        "body": "Hi, I wanted to know more about your pricing plans. Do you offer a free trial for the professional plan?"
    },
    {
        "subject": "System is broken!",
        "body": "I am extremely frustrated. Your platform has been down for 2 days and I cannot access my data. This is completely unacceptable. I demand a refund immediately."
    },
    {
        "subject": "Love the new update!",
        "body": "Just wanted to say the recent dashboard update is fantastic. It's much more intuitive now. One suggestion: could you add dark mode? That would be a great improvement."
    },
    {
        "subject": "YOU WON A PRIZE!!!",
        "body": "Congratulations! You have been selected for a free gift card worth $500. Click here to claim your prize now! Limited time offer. Act immediately!"
    },
    {
        "subject": "Need help with API setup",
        "body": "Hi support team, I need help configuring the API keys for my integration. I cannot figure out how to set up the webhook. Can someone assist me?"
    },
]

if __name__ == "__main__":
    classifier = EmailClassifier()
    print("=" * 60)
    print("  Email Classification Demo")
    print("=" * 60)
    for i, email in enumerate(DEMO_EMAILS, 1):
        result = classifier.classify(email["subject"], email["body"])
        print(f"\n[Email {i}]")
        print(f"  Subject   : {email['subject']}")
        print(f"  Category  : {result['category']}")
        print(f"  Confidence: {result['confidence_pct']}")
