"""
Cluster Labeling Service
Uses LLM to generate human-readable labels and summaries for ticket clusters
"""

import json
from dataclasses import dataclass

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class ClusterSummary:
    """Generated summary for a cluster"""
    cluster_id: int
    label: str  # Short descriptive label
    issue_description: str  # What is the core issue?
    common_symptoms: list[str]  # How users describe it
    environment: str  # Common environment/context
    recommended_response: str  # Suggested first response
    deflection_path: str  # Self-service or doc link suggestion
    confidence: float  # 0-1 confidence in summary


CLUSTER_SUMMARY_PROMPT = """You are a support ticket analyst. Analyze these representative tickets from a cluster and generate a structured summary.

REPRESENTATIVE TICKETS:
{tickets}

CLUSTER KEYWORDS: {keywords}

Generate a JSON response with:
{{
  "label": "Short 3-5 word label for this issue category",
  "issue_description": "1-2 sentence description of the core problem",
  "common_symptoms": ["symptom 1", "symptom 2", "symptom 3"],
  "environment": "Common environment, product version, or context",
  "recommended_response": "Suggested first response to users with this issue",
  "deflection_path": "Documentation link or self-service suggestion if applicable"
}}

Respond ONLY with valid JSON, no markdown or explanation."""


class ClusterLabeler:
    """Generates labels and summaries for ticket clusters using LLM"""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        timeout: float = 120.0,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def summarize_cluster(
        self,
        cluster_id: int,
        representative_texts: list[str],
        keywords: list[str] = None,
    ) -> ClusterSummary:
        """
        Generate a summary for a cluster based on representative tickets.

        Args:
            cluster_id: The cluster identifier
            representative_texts: Text content of representative tickets
            keywords: Optional list of cluster keywords

        Returns:
            ClusterSummary with generated fields
        """
        if not representative_texts:
            return self._empty_summary(cluster_id)

        # Truncate texts to fit context
        tickets_text = ""
        for i, text in enumerate(representative_texts[:5]):  # Max 5 tickets
            truncated = text[:2000]  # 2k chars per ticket
            tickets_text += f"\n--- TICKET {i+1} ---\n{truncated}\n"

        prompt = CLUSTER_SUMMARY_PROMPT.format(
            tickets=tickets_text,
            keywords=", ".join(keywords or []),
        )

        try:
            response = self._call_ollama(prompt)
            parsed = self._parse_response(response)

            return ClusterSummary(
                cluster_id=cluster_id,
                label=parsed.get("label", f"Cluster-{cluster_id}"),
                issue_description=parsed.get("issue_description", ""),
                common_symptoms=parsed.get("common_symptoms", []),
                environment=parsed.get("environment", ""),
                recommended_response=parsed.get("recommended_response", ""),
                deflection_path=parsed.get("deflection_path", ""),
                confidence=0.8 if parsed else 0.3,
            )
        except Exception as e:
            logger.error("Failed to summarize cluster", cluster_id=cluster_id, error=str(e))
            return self._empty_summary(cluster_id)

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for completion"""
        response = httpx.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3},
            },
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")

    def _parse_response(self, response: str) -> dict:
        """Parse JSON from LLM response"""
        try:
            # Try direct JSON parse
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        logger.warning("Failed to parse LLM response as JSON", response=response[:200])
        return {}

    def _empty_summary(self, cluster_id: int) -> ClusterSummary:
        """Return empty summary for failed cases"""
        return ClusterSummary(
            cluster_id=cluster_id,
            label=f"Cluster-{cluster_id}",
            issue_description="",
            common_symptoms=[],
            environment="",
            recommended_response="",
            deflection_path="",
            confidence=0.0,
        )

    def summarize_clusters_batch(
        self,
        clusters: list,  # list of ClusterResult
        ticket_texts: dict[str, str],  # ticket_id -> text mapping
    ) -> list[ClusterSummary]:
        """
        Generate summaries for multiple clusters.

        Args:
            clusters: List of ClusterResult objects
            ticket_texts: Dictionary mapping ticket_id to text content

        Returns:
            List of ClusterSummary objects
        """
        summaries = []
        for cluster in clusters:
            # Get texts for representative tickets
            rep_texts = []
            for tid in cluster.representative_ids:
                if tid in ticket_texts:
                    rep_texts.append(ticket_texts[tid])

            logger.info("Summarizing cluster",
                       cluster_id=cluster.cluster_id,
                       size=cluster.size,
                       representatives=len(rep_texts))

            summary = self.summarize_cluster(
                cluster_id=cluster.cluster_id,
                representative_texts=rep_texts,
                keywords=cluster.keywords,
            )
            summaries.append(summary)

        return summaries


def summary_to_dict(summary: ClusterSummary) -> dict:
    """Convert ClusterSummary to dictionary for JSON serialization"""
    return {
        "cluster_id": summary.cluster_id,
        "label": summary.label,
        "issue_description": summary.issue_description,
        "common_symptoms": summary.common_symptoms,
        "environment": summary.environment,
        "recommended_response": summary.recommended_response,
        "deflection_path": summary.deflection_path,
        "confidence": summary.confidence,
    }

