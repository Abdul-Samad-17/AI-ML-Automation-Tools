"""
Project 5: AI Workflow Automation System
Calder AI/ML Internship

Use case: Automated Support Ticket Processing Pipeline
  Input  → Raw email/text from a customer
  Processing → Classify intent, extract key info, assign priority, route to team
  Output → Structured ticket with routing decision + auto-response draft

Architecture:
  FastAPI REST API → Workflow Engine → LLM Decision Maker → Structured Output
  Logging → JSON log file + console
"""

import json
import logging
import os
import re
import sys
import time
import uuid
import urllib.request
import urllib.error
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


# ─────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────

LOG_FILE = Path(__file__).parent / "workflow.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("calder.workflow")


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

class Priority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class TicketCategory(str, Enum):
    BILLING = "Billing"
    TECHNICAL = "Technical"
    ACCOUNT = "Account"
    INQUIRY = "Inquiry"
    COMPLAINT = "Complaint"
    FEATURE_REQUEST = "Feature Request"
    SECURITY = "Security"
    OTHER = "Other"


class Team(str, Enum):
    BILLING_SUPPORT = "Billing Support"
    TECHNICAL_SUPPORT = "Technical Support"
    ACCOUNT_MANAGEMENT = "Account Management"
    SALES = "Sales"
    SECURITY = "Security Team"
    GENERAL_SUPPORT = "General Support"


# Input schema
class TicketInput(BaseModel):
    customer_name: str = Field(..., description="Name of the customer")
    customer_email: str = Field(..., description="Email of the customer")
    subject: str = Field(..., description="Email/ticket subject")
    body: str = Field(..., description="Full message body")
    account_tier: Optional[str] = Field("Starter", description="Customer account tier")


# Output schema
class ProcessedTicket(BaseModel):
    ticket_id: str
    timestamp: str
    customer_name: str
    customer_email: str
    subject: str
    category: TicketCategory
    priority: Priority
    assigned_team: Team
    sentiment: str
    key_entities: list[str]
    summary: str
    auto_response: str
    processing_time_ms: float
    llm_used: bool


# ─────────────────────────────────────────────
# Rule-Based Decision Engine
# ─────────────────────────────────────────────

PRIORITY_SIGNALS = {
    Priority.CRITICAL: [
        "data breach", "security incident", "account hacked", "unauthorized access",
        "system down", "complete outage", "production down", "cannot access at all",
        "legal action", "lawsuit",
    ],
    Priority.HIGH: [
        "urgent", "asap", "immediately", "cannot work", "blocking", "broken",
        "refund", "charged wrong", "double charged", "frustrated", "angry",
        "critical bug", "data lost",
    ],
    Priority.MEDIUM: [
        "issue", "problem", "not working", "help", "need assistance",
        "incorrect", "error", "failed",
    ],
}

CATEGORY_SIGNALS = {
    TicketCategory.BILLING: [
        "bill", "billing", "invoice", "charge", "payment", "refund", "subscription",
        "price", "pricing", "cost", "charged", "overcharged", "credit card",
    ],
    TicketCategory.TECHNICAL: [
        "bug", "crash", "error", "broken", "not working", "api", "integration",
        "performance", "slow", "timeout", "sync", "export", "import", "webhook",
    ],
    TicketCategory.ACCOUNT: [
        "login", "password", "account", "access", "two-factor", "mfa", "profile",
        "settings", "email change", "delete account", "team member",
    ],
    TicketCategory.SECURITY: [
        "hacked", "security", "breach", "unauthorized", "suspicious",
        "phishing", "fraud", "compromised",
    ],
    TicketCategory.FEATURE_REQUEST: [
        "feature", "suggestion", "would be great", "would love", "please add",
        "enhancement", "improvement", "request",
    ],
    TicketCategory.COMPLAINT: [
        "terrible", "awful", "worst", "unacceptable", "disappointed", "disgusted",
        "horrible", "never again", "waste of money",
    ],
    TicketCategory.INQUIRY: [
        "how do i", "can you", "what is", "do you offer", "is there",
        "information about", "learn more", "question",
    ],
}

TEAM_ROUTING: dict[TicketCategory, Team] = {
    TicketCategory.BILLING: Team.BILLING_SUPPORT,
    TicketCategory.TECHNICAL: Team.TECHNICAL_SUPPORT,
    TicketCategory.ACCOUNT: Team.ACCOUNT_MANAGEMENT,
    TicketCategory.INQUIRY: Team.SALES,
    TicketCategory.COMPLAINT: Team.GENERAL_SUPPORT,
    TicketCategory.FEATURE_REQUEST: Team.GENERAL_SUPPORT,
    TicketCategory.SECURITY: Team.SECURITY,
    TicketCategory.OTHER: Team.GENERAL_SUPPORT,
}


def detect_priority(text: str, account_tier: str) -> Priority:
    text_lower = text.lower()
    for priority in [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM]:
        if any(sig in text_lower for sig in PRIORITY_SIGNALS.get(priority, [])):
            # Enterprise customers get bumped up one level
            if account_tier == "Enterprise" and priority == Priority.MEDIUM:
                return Priority.HIGH
            return priority
    return Priority.LOW


def detect_category(text: str) -> TicketCategory:
    text_lower = text.lower()
    scores: dict[TicketCategory, int] = {cat: 0 for cat in TicketCategory}
    for category, signals in CATEGORY_SIGNALS.items():
        for sig in signals:
            if sig in text_lower:
                scores[category] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else TicketCategory.OTHER


def detect_sentiment(text: str) -> str:
    text_lower = text.lower()
    negative = ["angry", "frustrated", "terrible", "awful", "horrible", "worst",
                "unacceptable", "disappointed", "broken", "failed", "worst"]
    positive = ["great", "love", "excellent", "wonderful", "thank", "appreciate",
               "fantastic", "good", "happy", "satisfied"]
    neg = sum(1 for w in negative if w in text_lower)
    pos = sum(1 for w in positive if w in text_lower)
    if neg > pos: return "Negative"
    if pos > neg: return "Positive"
    return "Neutral"


def extract_entities(text: str) -> list[str]:
    """Extract key entities: emails, URLs, version numbers, error codes, amounts."""
    entities = []
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    entities.extend(emails)
    urls = re.findall(r'https?://\S+', text)
    entities.extend(urls)
    amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
    entities.extend(amounts)
    errors = re.findall(r'\b(?:error|err)\s*(?:code\s*)?[:\-]?\s*[A-Z0-9\-]{3,}\b', text, re.IGNORECASE)
    entities.extend(errors)
    versions = re.findall(r'v\d+\.\d+(?:\.\d+)?', text, re.IGNORECASE)
    entities.extend(versions)
    return list(set(entities))[:10]  # cap at 10


# ─────────────────────────────────────────────
# LLM Integration (Claude API)
# ─────────────────────────────────────────────

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-20250514"


def call_claude(prompt: str, api_key: str, system: str = "", max_tokens: int = 600) -> str:
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        CLAUDE_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["content"][0]["text"]
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")


def llm_summarize(ticket_input: TicketInput, api_key: str) -> tuple[str, str]:
    """Use LLM to generate ticket summary and auto-response."""
    system = (
        "You are a support workflow assistant for Calder, a business productivity platform. "
        "Respond ONLY with a JSON object. No preamble or markdown. "
        "Keys: summary (1-2 sentences), auto_response (polite, professional reply to customer, 2-4 sentences)."
    )
    prompt = (
        f"Customer: {ticket_input.customer_name} ({ticket_input.customer_email})\n"
        f"Subject: {ticket_input.subject}\n"
        f"Message: {ticket_input.body}\n\n"
        f"Generate a JSON with 'summary' and 'auto_response' keys."
    )
    try:
        raw = call_claude(prompt, api_key, system=system)
        # Strip any accidental markdown
        raw = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(raw)
        return data.get("summary", ""), data.get("auto_response", "")
    except Exception as e:
        logger.warning(f"LLM summarization failed: {e}")
        return "", ""


def rule_based_summary(ticket_input: TicketInput, category: TicketCategory, priority: Priority) -> str:
    return (
        f"Customer {ticket_input.customer_name} submitted a {category.value} ticket "
        f"regarding: {ticket_input.subject}. Priority assessed as {priority.value}."
    )


def rule_based_response(customer_name: str, category: TicketCategory, priority: Priority) -> str:
    eta = {"Critical": "1 hour", "High": "4 hours", "Medium": "24 hours", "Low": "48 hours"}
    return (
        f"Dear {customer_name},\n\n"
        f"Thank you for contacting Calder Support. We have received your {category.value} request "
        f"and assigned it {priority.value} priority. Our team will respond within {eta.get(priority.value, '24 hours')}.\n\n"
        f"Best regards,\nCalder Support Team"
    )


# ─────────────────────────────────────────────
# Workflow Engine
# ─────────────────────────────────────────────

class WorkflowEngine:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        logger.info("WorkflowEngine initialized | LLM=%s", bool(self.api_key))

    def process(self, ticket_input: TicketInput) -> ProcessedTicket:
        """Run the full ticket processing pipeline."""
        start = time.perf_counter()
        ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
        logger.info("Processing ticket %s for %s", ticket_id, ticket_input.customer_email)

        full_text = f"{ticket_input.subject} {ticket_input.body}"

        # Step 1: Rule-based classification
        category = detect_category(full_text)
        priority = detect_priority(full_text, ticket_input.account_tier or "Starter")
        sentiment = detect_sentiment(full_text)
        entities = extract_entities(full_text)
        team = TEAM_ROUTING[category]

        logger.info("%s | Category=%s | Priority=%s | Team=%s",
                    ticket_id, category.value, priority.value, team.value)

        # Step 2: LLM enrichment (if API key available)
        summary, auto_response = "", ""
        used_llm = False
        if self.api_key:
            try:
                summary, auto_response = llm_summarize(ticket_input, self.api_key)
                used_llm = bool(summary)
            except Exception as e:
                logger.error("LLM enrichment failed for %s: %s", ticket_id, e)

        # Fallback to rule-based if LLM not available
        if not summary:
            summary = rule_based_summary(ticket_input, category, priority)
        if not auto_response:
            auto_response = rule_based_response(ticket_input.customer_name, category, priority)

        elapsed = (time.perf_counter() - start) * 1000
        logger.info("Ticket %s processed in %.1fms | LLM=%s", ticket_id, elapsed, used_llm)

        return ProcessedTicket(
            ticket_id=ticket_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            customer_name=ticket_input.customer_name,
            customer_email=ticket_input.customer_email,
            subject=ticket_input.subject,
            category=category,
            priority=priority,
            assigned_team=team,
            sentiment=sentiment,
            key_entities=entities,
            summary=summary,
            auto_response=auto_response,
            processing_time_ms=round(elapsed, 2),
            llm_used=used_llm,
        )


# ─────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Calder AI Workflow Automation API",
        description=(
            "AI-driven support ticket processing pipeline. "
            "Classifies, prioritizes, routes, and drafts responses for incoming customer messages."
        ),
        version="1.0.0",
    )

    engine = WorkflowEngine()

    @app.get("/health")
    async def health_check():
        """Check API health."""
        return {
            "status": "ok",
            "service": "Calder Workflow Automation",
            "llm_enabled": bool(engine.api_key),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    @app.post("/api/v1/tickets/process", response_model=ProcessedTicket)
    async def process_ticket(ticket: TicketInput, request: Request):
        """
        Process a support ticket through the full AI workflow pipeline.

        Returns a structured ticket with classification, routing, and auto-response.
        """
        logger.info("POST /api/v1/tickets/process | client=%s", request.client.host if request.client else "unknown")
        try:
            result = engine.process(ticket)
            return result
        except Exception as e:
            logger.error("Ticket processing failed: %s", str(e))
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    @app.post("/api/v1/tickets/batch")
    async def process_batch(tickets: list[TicketInput]):
        """Process multiple tickets in one request (max 20)."""
        if len(tickets) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 tickets per batch.")
        results = []
        errors = []
        for i, ticket in enumerate(tickets):
            try:
                results.append(engine.process(ticket).model_dump())
            except Exception as e:
                errors.append({"index": i, "error": str(e)})
                logger.error("Batch ticket %d failed: %s", i, e)
        return {"processed": len(results), "errors": errors, "tickets": results}

    @app.get("/api/v1/categories")
    async def list_categories():
        """List all supported ticket categories and routing."""
        return {
            "categories": [c.value for c in TicketCategory],
            "routing": {cat.value: team.value for cat, team in TEAM_ROUTING.items()},
            "priorities": [p.value for p in Priority],
        }


# ─────────────────────────────────────────────
# Standalone Demo (no FastAPI)
# ─────────────────────────────────────────────

DEMO_TICKETS = [
    TicketInput(
        customer_name="Sarah Ahmed",
        customer_email="sarah@acme.com",
        subject="URGENT: Account hacked - unauthorized access!",
        body="Someone has accessed my account without my permission. I see logins from a foreign country. This is a security breach. Please help immediately!",
        account_tier="Professional",
    ),
    TicketInput(
        customer_name="Ali Hassan",
        customer_email="ali@startup.pk",
        subject="Question about Enterprise pricing",
        body="Hi, we are a team of 50 and interested in the Enterprise plan. Can you tell me more about custom pricing and what features are included?",
        account_tier="Starter",
    ),
    TicketInput(
        customer_name="Maria Chen",
        customer_email="maria@corp.com",
        subject="Double charged this month - $58 taken!",
        body="I was charged $29 twice this month. My invoice shows two charges for the Professional plan. Please refund $29 immediately. This is very frustrating.",
        account_tier="Professional",
    ),
]


def run_demo():
    """Run the workflow engine against demo tickets."""
    engine = WorkflowEngine()
    print("\n" + "=" * 65)
    print("  Calder AI Workflow Automation - Demo Run")
    print("=" * 65)
    for ticket in DEMO_TICKETS:
        result = engine.process(ticket)
        print(f"\n{'─'*65}")
        print(f"Ticket ID  : {result.ticket_id}")
        print(f"Customer   : {result.customer_name} ({result.customer_email})")
        print(f"Subject    : {result.subject}")
        print(f"Category   : {result.category.value}")
        print(f"Priority   : {result.priority.value}")
        print(f"Team       : {result.assigned_team.value}")
        print(f"Sentiment  : {result.sentiment}")
        print(f"Summary    : {result.summary}")
        print(f"Entities   : {result.key_entities}")
        print(f"Time       : {result.processing_time_ms}ms | LLM: {result.llm_used}")
        print(f"\nAuto-Response:\n{result.auto_response}")
    print(f"\n{'='*65}")


if __name__ == "__main__":
    if "--serve" in sys.argv:
        if not FASTAPI_AVAILABLE:
            print("FastAPI not installed. Run: pip install fastapi uvicorn")
            sys.exit(1)
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        run_demo()
