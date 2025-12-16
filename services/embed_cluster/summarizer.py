"""
Ticket Summarizer Service
Uses a large LLM to create detailed summaries for embedding.
Produces comprehensive summaries that fit the embedding model's context window.
Keeps model loaded for efficient batch processing.

Features:
- Streaming output for live progress visibility
- Performance metrics (tokens/sec, latency, throughput)
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional
import httpx
import structlog

logger = structlog.get_logger()

# Environment-based defaults
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
DEFAULT_SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-oss:120b")


@dataclass
class SummaryMetrics:
    """Metrics for a single summary generation."""
    ticket_id: str
    input_chars: int
    output_tokens: int
    output_chars: int
    latency_seconds: float
    tokens_per_second: float
    time_to_first_token: Optional[float] = None


@dataclass
class BatchMetrics:
    """Aggregate metrics for batch summarization."""
    total_tickets: int = 0
    completed: int = 0
    failed: int = 0
    total_input_chars: int = 0
    total_output_tokens: int = 0
    total_output_chars: int = 0
    total_latency_seconds: float = 0.0
    start_time: float = field(default_factory=time.time)

    @property
    def avg_tokens_per_second(self) -> float:
        if self.total_latency_seconds > 0:
            return self.total_output_tokens / self.total_latency_seconds
        return 0.0

    @property
    def avg_latency(self) -> float:
        if self.completed > 0:
            return self.total_latency_seconds / self.completed
        return 0.0

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def tickets_per_minute(self) -> float:
        if self.elapsed_seconds > 0:
            return (self.completed / self.elapsed_seconds) * 60
        return 0.0

    @property
    def eta_seconds(self) -> float:
        if self.completed > 0:
            remaining = self.total_tickets - self.completed
            avg_time = self.elapsed_seconds / self.completed
            return remaining * avg_time
        return 0.0

    def log_progress(self):
        """Log current progress with metrics."""
        eta_min = self.eta_seconds / 60
        logger.info(
            "Summarization progress",
            progress=f"{self.completed}/{self.total_tickets}",
            tps=f"{self.avg_tokens_per_second:.1f}",
            avg_latency=f"{self.avg_latency:.1f}s",
            throughput=f"{self.tickets_per_minute:.1f}/min",
            eta=f"{eta_min:.1f}min",
            total_tokens=self.total_output_tokens,
        )


# Anti-hallucination prompt - extracts ONLY information present in ticket
# Structured format for consistent parsing and embedding
# Always outputs in English (translates if necessary)
SUMMARIZE_PROMPT = """You are a support ticket analyst. Extract a structured summary from the ticket below.

IMPORTANT: Output ONLY the structured summary in ENGLISH. Do NOT include any reasoning, thinking, or explanation. Start directly with "**TICKET SUMMARY**". Translate all non-English content to English.

CRITICAL RULES:
1. ONLY include information EXPLICITLY stated in the ticket
2. If a section has no information, write "[Not provided]"
3. Do NOT infer or fabricate details
4. Quote exact error messages verbatim (translate to English if needed)
5. ALL OUTPUT MUST BE IN ENGLISH - translate any non-English content

TICKET METADATA:
- Ticket ID: {ticket_id}
- Subject: {subject}
- Status: {status}
- Priority: {priority}
- Created: {created_at}
- Requester: {requester_name} ({requester_email})
- Agent: {agent_name} ({agent_email})
- Organization: {organization}
- Region: {region}
- Tags: {tags}

CUSTOM FIELDS:
{custom_fields}

CONVERSATION ({comment_count} comments, {word_count} words):
{conversation}

OUTPUT (ALL IN ENGLISH):

**TICKET SUMMARY**
- ID: {ticket_id}
- Agent: {agent_name}
- Comments: {comment_count}
- Density: [sparse/moderate/detailed]

**ISSUE**
- Type: [license/bug/feature/how-to/integration/other]
- Category: [from tags]
- Severity: [stated or "not specified"]

**PROBLEM**
[1-2 sentence English summary of the customer's issue]

**TECHNICAL**
- Error: [exact quote in English or "none"]
- Component: [affected component or "not specified"]
- Version: [software version or "not specified"]
- License Keys: [if any or "none"]
- Hardware ID: [if any or "none"]

**RESOLUTION**
- Agent Action: [what agent did/advised, in English]
- Customer Response: [customer's reply or "no response"]
- Status: [resolved/pending/closed without response]

**IMAGE ATTACHMENTS** (if any):
[For each image, list: "filename.jpg: description from vision analysis"]
[If no images: "None"]

**KEYWORDS**: [5-7 English terms for clustering]"""


def build_prompt_from_ticket(ticket: dict) -> str:
    """Build the anti-hallucination prompt from a full ticket context."""
    # Extract organization info
    org = ticket.get("organization", {}) or {}
    org_fields = org.get("organization_fields", {}) if org else {}

    # Build conversation thread
    thread_parts = []
    for comment in ticket.get("comments", []):
        role = "[AGENT]" if comment.get("author_role") == "agent" else "[CUSTOMER]"
        name = comment.get("author_name", "Unknown")
        body = comment.get("plain_body", comment.get("body", ""))
        thread_parts.append(f"{role} {name}: {body}")

    conversation = "\n\n".join(thread_parts) if thread_parts else ticket.get("description", "No content")

    # Count words
    word_count = sum(len(c.get("plain_body", "").split()) for c in ticket.get("comments", []))
    if word_count == 0:
        word_count = len(ticket.get("description", "").split())

    # Build custom fields (non-null only)
    custom_fields_parts = []
    for cf in ticket.get("custom_fields", []):
        val = cf.get("value")
        if val and str(val).lower() not in ["null", "false", "none", ""]:
            custom_fields_parts.append(f"  - {cf.get('name')}: {str(val)[:200]}")
    custom_fields_str = "\n".join(custom_fields_parts) if custom_fields_parts else "  None provided"

    # Build image descriptions if present (from vision model processing)
    image_descriptions = ticket.get("_image_descriptions", [])
    if image_descriptions:
        image_parts = []
        for img in image_descriptions:
            image_parts.append(f"  - {img.get('filename')}: {img.get('description')}")
        conversation += "\n\n[IMAGE ATTACHMENTS - Descriptions from vision analysis]:\n" + "\n".join(image_parts)

    return SUMMARIZE_PROMPT.format(
        ticket_id=ticket.get("id", ticket.get("ticket_id", "unknown")),
        subject=ticket.get("subject", "No subject"),
        status=ticket.get("status", "unknown"),
        priority=ticket.get("priority") or "Not set",
        created_at=ticket.get("created_at", "unknown"),
        requester_name=ticket.get("requester_name", "Unknown"),
        requester_email=ticket.get("requester_email", ""),
        agent_name=ticket.get("assignee_name", "Unassigned"),
        agent_email=ticket.get("assignee_email", ""),
        organization=org.get("name", "Unknown") if org else "Unknown",
        region=org_fields.get("region", "Unknown"),
        tags=", ".join(ticket.get("tags", [])),
        custom_fields=custom_fields_str,
        comment_count=len(ticket.get("comments", [])),
        word_count=word_count,
        conversation=conversation,
    )


# Legacy prompt for simple ticket format (subject + description only)
SIMPLE_SUMMARIZE_PROMPT = """You are a support ticket analyst. Extract a structured summary from this ticket.

CRITICAL RULES:
1. ONLY include information EXPLICITLY stated in the ticket
2. If a section has no information, write "[Not provided]"
3. Do NOT infer, assume, or fabricate ANY details
4. Do NOT add example commands, API paths, KB articles, or version numbers unless explicitly mentioned

OUTPUT FORMAT:

**ISSUE**
- Type: [bug/feature/how-to/license/etc or "unclear"]
- Severity: [stated or "not specified"]

**PROBLEM**
[1-2 sentence summary using customer's exact words where possible]

**TECHNICAL**
- Error: [exact quote or "none provided"]
- Component: [specific or "not specified"]
- Version: [if mentioned or "not specified"]

**KEYWORDS** (for clustering):
[5-7 relevant terms from the ticket]

---
TICKET SUBJECT: {subject}
CONTENT: {description}
---

SUMMARY:"""


class TicketSummarizer:
    """Summarizes tickets using a large LLM before embedding.

    Features:
    - Streaming output for live visibility
    - Performance metrics tracking (TPS, latency, throughput)
    - Efficient batch processing with model kept loaded
    """

    DEFAULT_MAX_OUTPUT_TOKENS = 16000

    def __init__(
        self,
        ollama_url: str = None,
        model: str = None,
        timeout: float = 600.0,
        max_input_chars: int = 100000,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        keep_alive: str = "30m",
        stream: bool = True,  # Enable streaming by default
    ):
        # Use environment defaults if not specified
        if ollama_url is None:
            ollama_url = DEFAULT_OLLAMA_URL
        if model is None:
            model = DEFAULT_SUMMARY_MODEL
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_input_chars = max_input_chars
        self.max_output_tokens = max_output_tokens
        self.keep_alive = keep_alive
        self.stream = stream
        self._client = None
        self.batch_metrics: Optional[BatchMetrics] = None

    def _get_client(self) -> httpx.Client:
        """Get or create a persistent HTTP client for connection reuse."""
        if self._client is None:
            self._client = httpx.Client(timeout=httpx.Timeout(self.timeout, connect=30.0))
        return self._client

    def _preload_model(self):
        """Preload the model before batch processing."""
        logger.info("Preloading model", model=self.model)
        try:
            client = self._get_client()
            response = client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "",
                    "keep_alive": self.keep_alive,
                },
            )
            response.raise_for_status()
            logger.info("Model preloaded successfully", model=self.model)
        except Exception as e:
            logger.warning("Model preload failed, will load on first request", error=str(e))

    def summarize_full_ticket(
        self,
        ticket: dict,
        is_last: bool = False,
        show_live: bool = True,
    ) -> tuple[str, SummaryMetrics]:
        """
        Summarize a ticket with full context (comments, custom fields, agent info).
        Uses anti-hallucination prompt.

        Args:
            ticket: Full ticket dict with comments, organization, custom_fields, etc.

        Returns:
            Tuple of (summary_text, metrics)
        """
        ticket_id = str(ticket.get("id", ticket.get("ticket_id", "")))
        subject = ticket.get("subject", "No subject")

        # Build anti-hallucination prompt
        prompt = build_prompt_from_ticket(ticket)
        input_chars = len(prompt)

        return self._run_summarization(
            prompt=prompt,
            ticket_id=ticket_id,
            subject=subject,
            input_chars=input_chars,
            is_last=is_last,
            show_live=show_live,
        )

    def summarize_streaming(
        self,
        subject: str,
        description: str,
        ticket_id: str = "",
        is_last: bool = False,
        show_live: bool = True,
    ) -> tuple[str, SummaryMetrics]:
        """
        Summarize a ticket with simple format (subject + description only).
        Uses simple anti-hallucination prompt.

        Returns:
            Tuple of (summary_text, metrics)
        """
        input_chars = len(description)

        # Truncate if needed
        if len(description) > self.max_input_chars:
            half = self.max_input_chars // 2
            description = description[:half] + "\n\n[...content truncated...]\n\n" + description[-half:]

        prompt = SIMPLE_SUMMARIZE_PROMPT.format(
            subject=subject or "No subject",
            description=description or "No description provided",
        )

        return self._run_summarization(
            prompt=prompt,
            ticket_id=ticket_id,
            subject=subject,
            input_chars=input_chars,
            is_last=is_last,
            show_live=show_live,
        )

    def _run_summarization(
        self,
        prompt: str,
        ticket_id: str,
        subject: str,
        input_chars: int,
        is_last: bool = False,
        show_live: bool = True,
    ) -> tuple[str, SummaryMetrics]:
        """
        Internal method to run summarization with streaming.
        """

        start_time = time.time()
        time_to_first_token = None
        output_tokens = 0
        summary_parts = []

        try:
            # Use streaming request
            with self._get_client().stream(
                "POST",
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "keep_alive": "0" if is_last else self.keep_alive,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": self.max_output_tokens,
                    },
                },
            ) as response:
                response.raise_for_status()

                if show_live:
                    # Print header for this ticket
                    print(f"\n{'='*60}", flush=True)
                    print(f"📋 Ticket {ticket_id}: {subject[:60]}...", flush=True)
                    print(f"{'='*60}", flush=True)

                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            if time_to_first_token is None:
                                time_to_first_token = time.time() - start_time
                            summary_parts.append(token)
                            output_tokens += 1
                            if show_live:
                                print(token, end="", flush=True)

                        # Check if done
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

                if show_live:
                    print("\n", flush=True)

            total_time = time.time() - start_time
            summary = "".join(summary_parts).strip()

            metrics = SummaryMetrics(
                ticket_id=ticket_id,
                input_chars=input_chars,
                output_tokens=output_tokens,
                output_chars=len(summary),
                latency_seconds=total_time,
                tokens_per_second=output_tokens / total_time if total_time > 0 else 0,
                time_to_first_token=time_to_first_token,
            )

            if not summary:
                logger.warning("Empty summary returned, using prompt excerpt")
                return f"Subject: {subject}\n\n[Summary generation failed - no output]", metrics

            return summary, metrics

        except Exception as e:
            total_time = time.time() - start_time
            logger.warning("Summarization failed", error=str(e), ticket_subject=subject[:100])
            metrics = SummaryMetrics(
                ticket_id=ticket_id,
                input_chars=input_chars,
                output_tokens=0,
                output_chars=0,
                latency_seconds=total_time,
                tokens_per_second=0,
            )
            return f"Subject: {subject}\n\n[Summary generation failed: {str(e)[:200]}]", metrics

    def summarize_batch(
        self,
        tickets: list[dict],
        subject_key: str = "subject",
        description_key: str = "description",
        show_live: bool = True,
        existing_summaries: dict[str, str] | None = None,
    ) -> tuple[list[str], BatchMetrics]:
        """
        Summarize a batch of tickets with streaming output and metrics.
        Skips tickets that already have summaries.

        Args:
            tickets: List of ticket dictionaries
            subject_key: Key for subject field
            description_key: Key for description field
            show_live: If True, stream output to stdout
            existing_summaries: Dict of ticket_id -> summary for already processed tickets

        Returns:
            Tuple of (list of summaries, batch metrics)
        """
        existing_summaries = existing_summaries or {}
        summaries = []
        total = len(tickets)
        skipped = 0

        self.batch_metrics = BatchMetrics(total_tickets=total)

        if total == 0:
            return summaries, self.batch_metrics

        # Count how many we'll skip
        for ticket in tickets:
            ticket_id = str(ticket.get("ticket_id") or ticket.get("id") or "")
            if ticket_id in existing_summaries:
                skipped += 1

        to_process = total - skipped

        print(f"\n🚀 Starting summarization with {self.model}", flush=True)
        print(f"   Total tickets: {total}", flush=True)
        print(f"   Already summarized (skipping): {skipped}", flush=True)
        print(f"   To process: {to_process}", flush=True)
        print(f"   Max output tokens: {self.max_output_tokens}", flush=True)

        if to_process > 0:
            self._preload_model()

        for i, ticket in enumerate(tickets):
            ticket_id = str(ticket.get("ticket_id") or ticket.get("id") or str(i))
            subject = ticket.get(subject_key, "")

            # Skip if already summarized
            if ticket_id in existing_summaries:
                summaries.append(existing_summaries[ticket_id])
                continue

            description = ticket.get(description_key, "") or ticket.get("ticket_fulltext", "")
            is_last = (i == total - 1)

            summary, metrics = self.summarize_streaming(
                subject=subject,
                description=description,
                ticket_id=ticket_id,
                is_last=is_last,
                show_live=show_live,
            )

            summaries.append(summary)

            # Update batch metrics
            self.batch_metrics.completed += 1
            self.batch_metrics.total_input_chars += metrics.input_chars
            self.batch_metrics.total_output_tokens += metrics.output_tokens
            self.batch_metrics.total_output_chars += metrics.output_chars
            self.batch_metrics.total_latency_seconds += metrics.latency_seconds

            # Calculate rates and ETA
            tickets_per_hour = self.batch_metrics.tickets_per_minute * 60
            remaining = to_process - self.batch_metrics.completed
            eta_hours = remaining / tickets_per_hour if tickets_per_hour > 0 else 0

            # Show compact KPIs
            if show_live:
                print(f"\n   📊 [{self.batch_metrics.completed}/{to_process}] "
                      f"Rate: {tickets_per_hour:.0f}/hr | "
                      f"TPS: {self.batch_metrics.avg_tokens_per_second:.1f} | "
                      f"ETA: {eta_hours:.1f}h", flush=True)

            if (self.batch_metrics.completed) % 10 == 0:
                self.batch_metrics.log_progress()

        # Final summary
        print(f"\n{'='*60}", flush=True)
        print(f"✅ Summarization complete!", flush=True)
        print(f"   Processed: {self.batch_metrics.completed} | Skipped: {skipped}", flush=True)
        print(f"   Total tokens: {self.batch_metrics.total_output_tokens:,}", flush=True)
        print(f"   Total time: {self.batch_metrics.elapsed_seconds:.1f}s", flush=True)
        print(f"   Avg TPS: {self.batch_metrics.avg_tokens_per_second:.1f}", flush=True)
        print(f"   Throughput: {self.batch_metrics.tickets_per_minute:.1f} tickets/min", flush=True)
        print(f"{'='*60}\n", flush=True)

        return summaries, self.batch_metrics

    def summarize_batch_full_context(
        self,
        tickets: list[dict],
        show_live: bool = True,
        existing_summaries: dict[str, str] | None = None,
    ) -> tuple[list[str], BatchMetrics]:
        """
        Summarize a batch of tickets with FULL context (comments, custom fields, agent info).
        Uses anti-hallucination prompt.

        Args:
            tickets: List of full ticket dicts (with comments, organization, custom_fields)
            show_live: If True, stream output to stdout
            existing_summaries: Dict of ticket_id -> summary for already processed tickets

        Returns:
            Tuple of (list of summaries, batch metrics)
        """
        existing_summaries = existing_summaries or {}
        summaries = []
        total = len(tickets)
        skipped = 0

        self.batch_metrics = BatchMetrics(total_tickets=total)

        if total == 0:
            return summaries, self.batch_metrics

        # Count how many we'll skip
        for ticket in tickets:
            ticket_id = str(ticket.get("ticket_id") or ticket.get("id") or "")
            if ticket_id in existing_summaries:
                skipped += 1

        to_process = total - skipped

        print(f"\n🚀 Starting FULL CONTEXT summarization with {self.model}", flush=True)
        print(f"   Anti-hallucination prompt: ENABLED", flush=True)
        print(f"   Total tickets: {total}", flush=True)
        print(f"   Already summarized (skipping): {skipped}", flush=True)
        print(f"   To process: {to_process}", flush=True)
        print(f"   Max output tokens: {self.max_output_tokens}", flush=True)

        if to_process > 0:
            self._preload_model()

        for i, ticket in enumerate(tickets):
            ticket_id = str(ticket.get("ticket_id") or ticket.get("id") or str(i))

            # Skip if already summarized
            if ticket_id in existing_summaries:
                summaries.append(existing_summaries[ticket_id])
                continue

            is_last = (i == total - 1)

            summary, metrics = self.summarize_full_ticket(
                ticket=ticket,
                is_last=is_last,
                show_live=show_live,
            )

            summaries.append(summary)

            # Update batch metrics
            self.batch_metrics.completed += 1
            self.batch_metrics.total_input_chars += metrics.input_chars
            self.batch_metrics.total_output_tokens += metrics.output_tokens
            self.batch_metrics.total_output_chars += metrics.output_chars
            self.batch_metrics.total_latency_seconds += metrics.latency_seconds

            # Calculate rates and ETA
            tickets_per_hour = self.batch_metrics.tickets_per_minute * 60
            remaining = to_process - self.batch_metrics.completed
            eta_hours = remaining / tickets_per_hour if tickets_per_hour > 0 else 0

            # Show compact KPIs
            if show_live:
                print(f"\n   📊 [{self.batch_metrics.completed}/{to_process}] "
                      f"Rate: {tickets_per_hour:.0f}/hr | "
                      f"TPS: {self.batch_metrics.avg_tokens_per_second:.1f} | "
                      f"ETA: {eta_hours:.1f}h", flush=True)

            if (self.batch_metrics.completed) % 10 == 0:
                self.batch_metrics.log_progress()

        # Final summary
        print(f"\n{'='*60}", flush=True)
        print(f"✅ Full context summarization complete!", flush=True)
        print(f"   Processed: {self.batch_metrics.completed} | Skipped: {skipped}", flush=True)
        print(f"   Total tokens: {self.batch_metrics.total_output_tokens:,}", flush=True)
        print(f"   Total time: {self.batch_metrics.elapsed_seconds:.1f}s", flush=True)
        print(f"   Avg TPS: {self.batch_metrics.avg_tokens_per_second:.1f}", flush=True)
        print(f"   Throughput: {self.batch_metrics.tickets_per_minute:.1f} tickets/min", flush=True)
        print(f"{'='*60}\n", flush=True)

        return summaries, self.batch_metrics

    def __del__(self):
        """Cleanup HTTP client on destruction."""
        if self._client:
            self._client.close()

