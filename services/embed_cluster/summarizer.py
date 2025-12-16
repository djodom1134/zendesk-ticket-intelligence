"""
Ticket Summarizer Service
Uses a large LLM to summarize tickets before embedding.
Keeps model loaded for efficient batch processing.
"""

import httpx
import structlog

logger = structlog.get_logger()


SUMMARIZE_PROMPT = """Summarize this support ticket for clustering purposes.
Focus on: the core issue, error messages, affected product/feature, and environment.
Keep it concise (2-4 sentences max).

TICKET:
Subject: {subject}
Description: {description}

SUMMARY:"""


class TicketSummarizer:
    """Summarizes tickets using a large LLM before embedding.

    Uses Ollama's keep_alive parameter to keep model loaded during batch processing,
    avoiding expensive load/unload cycles per ticket.
    """

    def __init__(
        self,
        ollama_url: str = "http://ollama:11434",
        model: str = "gpt-oss:120b",
        timeout: float = 300.0,
        max_input_chars: int = 8000,
        keep_alive: str = "30m",  # Keep model loaded for 30 minutes
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_input_chars = max_input_chars
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
        Summarize a single ticket.

        Args:
            subject: Ticket subject line
            description: Full ticket description/body
            is_last: If True, allow model to unload after this request

        Returns:
            Summarized text suitable for embedding
        """
        # Truncate description if too long
        if len(description) > self.max_input_chars:
            description = description[:self.max_input_chars] + "..."

        prompt = SUMMARIZE_PROMPT.format(
            subject=subject or "No subject",
            description=description or "No description",
        )

        try:
            client = self._get_client()
            response = client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "0" if is_last else self.keep_alive,  # Unload after last
                    "options": {"temperature": 0.1, "num_predict": 200},
                },
            )
            response.raise_for_status()
            result = response.json()
            summary = result.get("response", "").strip()

            if not summary:
                # Fallback: return truncated original
                return f"{subject}. {description[:500]}"

            return summary

        except Exception as e:
            logger.warning("Summarization failed, using fallback", error=str(e))
            # Fallback: return truncated original
            return f"{subject}. {description[:500]}"

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

