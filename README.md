# Calder AI/ML Internship — Project Submissions

> **Applicant submission for the Calder AI/ML Internship**  
> All 5 projects implemented in Python with clean architecture, tests, and documentation.

---

## Repository Structure

```
calder_ai_ml/
├── requirements.txt
├── README.md
├── project1_faq_chatbot/
│   ├── chatbot.py          # TF-IDF FAQ retrieval chatbot
│   ├── faq_data.json       # Structured FAQ dataset (12 entries)
│   └── test_chatbot.py     # Unit tests
├── project2_email_classifier/
│   ├── classifier.py       # Naive Bayes + rule-based email classifier
│   └── test_classifier.py  # Unit tests
├── project3_doc_qa/
│   ├── doc_qa.py           # PDF extraction + TF-IDF document QA
│   └── test_doc_qa.py      # Unit tests
├── project4_rag_chatbot/
│   ├── rag_chatbot.py      # RAG pipeline with in-memory vector store
│   └── test_rag_chatbot.py # Unit tests
└── project5_workflow_automation/
    ├── main.py             # FastAPI workflow automation system
    ├── test_main.py        # Unit tests
    └── workflow.log        # Auto-generated log file
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/calder-ai-ml-internship.git
cd calder-ai-ml-internship

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

# (Optional) Set Claude API key for LLM features
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Python version:** 3.11+

---

## Projects

---

### Project 1 — FAQ Chatbot System

**Approach:** TF-IDF vectorization with cosine similarity (built from scratch, no external ML libraries).

| Feature | Detail |
|---|---|
| Dataset | 12 FAQs in `faq_data.json` (JSON format) |
| Retrieval | TF-IDF + cosine similarity |
| Preprocessing | Lowercasing, punctuation removal, stop-word filtering |
| Fallback | Returns contact info when similarity < threshold |

**Run:**
```bash
cd project1_faq_chatbot
python chatbot.py
```

**Test:**
```bash
python test_chatbot.py
```

**API usage:**
```python
from chatbot import FAQChatbot
bot = FAQChatbot()
result = bot.get_response("How do I reset my password?")
print(result["answer"])  # → "To reset your password..."
print(result["score"])   # → 0.38 (similarity score)
```

---

### Project 2 — Email Classification System

**Approach:** Multinomial Naive Bayes classifier trained on 50 labeled examples, with keyword-rule boosting for high-confidence cases.

| Feature | Detail |
|---|---|
| Categories | Inquiry, Complaint, Feedback, Spam, Support Request (5 classes) |
| Model | Naive Bayes with Laplace smoothing |
| Boosting | Keyword rule layer corrects low-confidence predictions |
| Output | Predicted category + confidence % + all class scores |

**Run:**
```bash
cd project2_email_classifier
python classifier.py          # Runs demo emails
```

**Test:**
```bash
python test_classifier.py
```

**API usage:**
```python
from classifier import EmailClassifier
clf = EmailClassifier()

result = clf.classify(
    subject="System is broken!",
    body="I'm frustrated, the platform has been down for 2 days. I want a refund!"
)
print(result["category"])     # → "Complaint"
print(result["confidence_pct"])  # → "87.3%"

# Batch mode
results = clf.classify_batch([
    {"subject": "Pricing question", "body": "What plans do you offer?"},
    {"subject": "Bug report", "body": "The export feature is broken."},
])
```

---

### Project 3 — Document Question Answering System

**Approach:** PDF text extraction → word-level chunking with overlap → TF-IDF index → cosine similarity search → top-K excerpts returned.

| Feature | Detail |
|---|---|
| PDF extraction | PyMuPDF (fitz) with pypdf fallback |
| Chunking | 200-word chunks with 50-word overlap |
| Search | TF-IDF cosine similarity over chunks |
| Output | Ranked excerpts with relevance scores |

**Run:**
```bash
cd project3_doc_qa
python doc_qa.py path/to/document.pdf
# Or interactive mode (prompts for PDF path):
python doc_qa.py
```

**Test:**
```bash
python test_doc_qa.py
```

**API usage:**
```python
from doc_qa import DocumentQASystem

qa = DocumentQASystem()
qa.load_document("report.pdf")            # Load PDF
# qa.load_text("plain text...", "name")  # Or load plain text

result = qa.answer("What are the security certifications?")
print(result["answer"])                   # Best matching excerpt
for ex in result["excerpts"]:
    print(f"Score {ex['score']}: {ex['text'][:200]}")
```

---

### Project 4 — Smart FAQ Chatbot (RAG-Based System)

**Approach:** Full RAG pipeline — knowledge base is chunked, TF-IDF embedded, and stored in an in-memory vector store. Queries retrieve top-K context chunks, which are passed to Claude for answer generation. Falls back to retrieval-only mode if no API key is set.

| Feature | Detail |
|---|---|
| Embeddings | TF-IDF (dense-equivalent, no external API) |
| Vector store | Custom in-memory store with cosine similarity search |
| Generation | Claude `claude-sonnet-4-20250514` via Anthropic API |
| Fallback | Returns best retrieved document if LLM unavailable |
| Knowledge base | 21 curated documents covering all Calder topics |

**Run:**
```bash
cd project4_rag_chatbot
export ANTHROPIC_API_KEY="sk-ant-..."   # Optional: enables LLM generation
python rag_chatbot.py
```

**Test:**
```bash
python test_rag_chatbot.py
```

**API usage:**
```python
from rag_chatbot import RAGChatbot

bot = RAGChatbot(api_key="sk-ant-...")  # or set ANTHROPIC_API_KEY env var
result = bot.chat("What security certifications does Calder have?")
print(result["answer"])
print(f"Retrieved {len(result['retrieved_docs'])} docs | LLM: {result['used_llm']}")

# Add custom documents at runtime
bot.add_documents(["New feature: Calder now supports Notion integration as of May 2025."])
```

---

### Project 5 — AI Workflow Automation System

**Approach:** End-to-end support ticket processing pipeline. Rule-based classification + LLM enrichment → structured output with routing decision and auto-response.

```
Input (customer email/message)
    ↓
[Rule Engine] → Category + Priority + Sentiment + Entity Extraction
    ↓
[LLM Enrichment] → Summary + Auto-Response Draft  (if API key set)
    ↓
[Output] → Structured ProcessedTicket (JSON)
```

| Feature | Detail |
|---|---|
| Categories | Billing, Technical, Account, Inquiry, Complaint, Feature Request, Security, Other |
| Priority | Critical / High / Medium / Low (with Enterprise tier escalation) |
| Teams | Billing Support, Technical Support, Account Management, Sales, Security, General |
| API | FastAPI REST with `/process` and `/batch` endpoints |
| Logging | Structured JSON logs to `workflow.log` + console |
| LLM | Claude generates polished summaries and auto-responses |

**Run demo (no server required):**
```bash
cd project5_workflow_automation
python main.py
```

**Run as API server:**
```bash
pip install fastapi uvicorn pydantic
python main.py --serve
# → API available at http://localhost:8000
# → Docs at http://localhost:8000/docs
```

**API endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/v1/tickets/process` | Process single ticket |
| POST | `/api/v1/tickets/batch` | Process up to 20 tickets |
| GET | `/api/v1/categories` | List categories and routing |

**Example request:**
```bash
curl -X POST http://localhost:8000/api/v1/tickets/process \
  -H "Content-Type: application/json" \
  -d '{
    "customer_name": "Ali Hassan",
    "customer_email": "ali@company.com",
    "subject": "Double charge on my account",
    "body": "I was charged $29 twice this month. Please refund immediately.",
    "account_tier": "Professional"
  }'
```

**Test:**
```bash
python test_main.py
```

---

## Running All Tests

```bash
# From the root directory
cd project1_faq_chatbot && python test_chatbot.py && cd ..
cd project2_email_classifier && python test_classifier.py && cd ..
cd project3_doc_qa && python test_doc_qa.py && cd ..
cd project4_rag_chatbot && python test_rag_chatbot.py && cd ..
cd project5_workflow_automation && python test_main.py && cd ..
```

Or with pytest:
```bash
pytest . -v
```

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| ML library | None (stdlib only) | Demonstrates understanding of fundamentals; no dependency on sklearn/numpy |
| Embeddings | TF-IDF | No external API costs; fast; interpretable |
| LLM | Claude claude-sonnet-4-20250514 | Best quality; Calder-adjacent ecosystem |
| API framework | FastAPI | Async, auto-docs (Swagger), modern Python |
| Data models | Pydantic | Type safety, validation, serialization |
| Fallbacks | Rule-based | Every system degrades gracefully without API key |

---

## Contact

Submitted for the **Calder AI/ML Internship** — NASTP, Rawalpindi  
LinkedIn: [linkedin.com/company/calderr](https://linkedin.com/company/calderr)
