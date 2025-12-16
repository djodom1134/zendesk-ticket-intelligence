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
import sys
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional
import httpx
import structlog

logger = structlog.get_logger()


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


# Detailed summary prompt - produces rich summaries for better embedding quality
# Target: ~2000-4000 tokens output to fit qwen3-embedding's 40K context window
SUMMARIZE_PROMPT = """You are a support ticket analyst. Create a DETAILED technical summary of this support ticket for semantic clustering and search.

Your summary should capture ALL of the following (when present):

1. **ISSUE CLASSIFICATION**
   - Primary issue type (bug, feature request, how-to, integration, performance, security, billing, etc.)
   - Severity/impact level
   - Whether this is a regression or new issue

2. **TECHNICAL DETAILS**
   - Exact error messages, codes, and stack traces
   - Affected product/feature/component/module
   - API endpoints or functions involved
   - Configuration settings mentioned

3. **ENVIRONMENT**
   - Platform (Windows/Mac/Linux/iOS/Android)
   - Version numbers (product version, OS version, browser, SDK)
   - Deployment type (cloud, on-prem, hybrid, container, serverless)
   - Scale/load context if mentioned

4. **REPRODUCTION**
   - Steps to reproduce if described
   - Frequency (always, intermittent, specific conditions)
   - Workarounds attempted or known

5. **TIMELINE & CONTEXT**
   - When the issue started
   - Any recent changes (updates, deployments, config changes)
   - Related tickets or incidents mentioned

6. **CUSTOMER CONTEXT**
   - Use case being attempted
   - Business impact described
   - Urgency indicators

7. **RESOLUTION HINTS**
   - Any solutions proposed in the thread
   - What was tried and failed
   - Documentation or KB articles referenced

Write a comprehensive paragraph-form summary. Include specific technical terms, error codes, version numbers, and keywords that would help match this ticket with similar issues. Do NOT omit details - more context is better for clustering accuracy.

---
TICKET SUBJECT: {subject}

FULL TICKET CONTENT:
{description}
---

DETAILED SUMMARY:"""


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
        ollama_url: str = "http://ollama:11434",
        model: str = "gpt-oss:120b",
        timeout: float = 600.0,
        max_input_chars: int = 100000,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        keep_alive: str = "30m",
        stream: bool = True,  # Enable streaming by default
    ):
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

    def summarize_streaming(
        self,
        subject: str,
        description: str,
        ticket_id: str = "",
        is_last: bool = False,
        show_live: bool = True,
    ) -> tuple[str, SummaryMetrics]:
        """
        Summarize a ticket with streaming output and metrics.

        Returns:
            Tuple of (summary_text, metrics)
        """
        input_chars = len(description)

        # Truncate if needed
        if len(description) > self.max_input_chars:
            half = self.max_input_chars // 2
            description = description[:half] + "\n\n[...content truncated...]\n\n" + description[-half:]

        prompt = SUMMARIZE_PROMPT.format(
            subject=subject or "No subject",
            description=description or "No description provided",
        )

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
                logger.warning("Empty summary returned, using original text")
                return f"Subject: {subject}\n\n{description[:8000]}", metrics

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
            return f"Subject: {subject}\n\n{description[:8000]}", metrics

    def summarize_batch(
        self,
        tickets: list[dict],
        subject_key: str = "subject",
        description_key: str = "description",
        show_live: bool = True,
    ) -> tuple[list[str], BatchMetrics]:
        """
        Summarize a batch of tickets with streaming output and metrics.

        Args:
            tickets: List of ticket dictionaries
            subject_key: Key for subject field
            description_key: Key for description field
            show_live: If True, stream output to stdout

        Returns:
            Tuple of (list of summaries, batch metrics)
        """
        summaries = []
        total = len(tickets)

        self.batch_metrics = BatchMetrics(total_tickets=total)

        if total == 0:
            return summaries, self.batch_metrics

        # Preload model before batch
        self._preload_model()

        print(f"\n🚀 Starting summarization of {total} tickets with {self.model}", flush=True)
        print(f"   Streaming: {'enabled' if show_live else 'disabled'}", flush=True)
        print(f"   Max output tokens: {self.max_output_tokens}", flush=True)

        for i, ticket in enumerate(tickets):
            ticket_id = ticket.get("ticket_id") or ticket.get("id") or str(i)
            subject = ticket.get(subject_key, "")
            description = ticket.get(description_key, "") or ticket.get("ticket_fulltext", "")
            is_last = (i == total - 1)

            summary, metrics = self.summarize_streaming(
                subject=subject,
                description=description,
                ticket_id=str(ticket_id),
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

            # Log progress every 10 tickets or show compact metrics
            if show_live:
                print(f"   ⏱️  {metrics.latency_seconds:.1f}s | "
                      f"📊 {metrics.tokens_per_second:.1f} tok/s | "
                      f"📝 {metrics.output_tokens} tokens", flush=True)

            if (i + 1) % 10 == 0:
                self.batch_metrics.log_progress()

        # Final summary
        print(f"\n{'='*60}", flush=True)
        print(f"✅ Summarization complete!", flush=True)
        print(f"   Total tickets: {self.batch_metrics.completed}", flush=True)
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

