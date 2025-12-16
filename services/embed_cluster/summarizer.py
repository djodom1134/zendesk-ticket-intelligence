"""
Ticket Summarizer Service
Uses a large LLM to create detailed summaries for embedding.
Produces comprehensive summaries that fit the embedding model's context window.
Keeps model loaded for efficient batch processing.
"""

import httpx
import structlog

logger = structlog.get_logger()


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

    Produces detailed summaries that maximize the embedding model's 40K token
    context window for better clustering quality.

    Uses Ollama's keep_alive parameter to keep model loaded during batch processing,
    avoiding expensive load/unload cycles per ticket.
    """

    # Allow up to 16K tokens output - embedding model can handle 40K total
    # This gives us room for very detailed summaries while leaving headroom
    DEFAULT_MAX_OUTPUT_TOKENS = 16000

    def __init__(
        self,
        ollama_url: str = "http://ollama:11434",
        model: str = "gpt-oss:120b",
        timeout: float = 600.0,  # Longer timeout for detailed summaries
        max_input_chars: int = 100000,  # Allow very long tickets
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        keep_alive: str = "30m",  # Keep model loaded for 30 minutes
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_input_chars = max_input_chars
        self.max_output_tokens = max_output_tokens
        self.keep_alive = keep_alive
        self._client = None

    def _get_client(self) -> httpx.Client:
        """Get or create a persistent HTTP client for connection reuse."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _preload_model(self):
        """Preload the model before batch processing."""
        logger.info("Preloading model", model=self.model)
        try:
            # Send empty prompt with long keep_alive to load model
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

    def summarize(self, subject: str, description: str, is_last: bool = False) -> str:
        """
        Create a detailed summary of a single ticket.

        Args:
            subject: Ticket subject line
            description: Full ticket description/body (including all comments)
            is_last: If True, allow model to unload after this request

        Returns:
            Detailed summary text suitable for embedding (~2000 tokens)
        """
        # Truncate description if too long for LLM context
        if len(description) > self.max_input_chars:
            # Keep beginning and end - often resolution info is at the end
            half = self.max_input_chars // 2
            description = description[:half] + "\n\n[...content truncated...]\n\n" + description[-half:]

        prompt = SUMMARIZE_PROMPT.format(
            subject=subject or "No subject",
            description=description or "No description provided",
        )

        try:
            client = self._get_client()
            response = client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "0" if is_last else self.keep_alive,
                    "options": {
                        "temperature": 0.2,  # Low temp for factual extraction
                        "num_predict": self.max_output_tokens,  # Allow long detailed output
                    },
                },
            )
            response.raise_for_status()
            result = response.json()
            summary = result.get("response", "").strip()

            if not summary:
                # Fallback: return original text (will be truncated at embedding stage if needed)
                logger.warning("Empty summary returned, using original text")
                return f"Subject: {subject}\n\n{description[:8000]}"

            return summary

        except Exception as e:
            logger.warning("Summarization failed, using fallback", error=str(e), ticket_subject=subject[:100])
            # Fallback: return original text
            return f"Subject: {subject}\n\n{description[:8000]}"

    def summarize_batch(
        self,
        tickets: list[dict],
        subject_key: str = "subject",
        description_key: str = "description",
    ) -> list[str]:
        """
        Summarize a batch of tickets efficiently.

        Keeps the model loaded for the entire batch, then unloads after completion.

        Args:
            tickets: List of ticket dictionaries
            subject_key: Key for subject field
            description_key: Key for description field

        Returns:
            List of summarized texts
        """
        summaries = []
        total = len(tickets)

        if total == 0:
            return summaries

        # Preload model before batch
        self._preload_model()

        for i, ticket in enumerate(tickets):
            if i % 25 == 0:
                logger.info("Summarizing tickets", progress=f"{i}/{total}")

            subject = ticket.get(subject_key, "")
            description = ticket.get(description_key, "") or ticket.get("ticket_fulltext", "")

            is_last = (i == total - 1)
            summary = self.summarize(subject, description, is_last=is_last)
            summaries.append(summary)

        logger.info("Summarization complete", total=total)
        return summaries

    def __del__(self):
        """Cleanup HTTP client on destruction."""
        if self._client:
            self._client.close()

