"""
Zendesk A2A Protocol Client
Communicates with the Zendesk agent using Google's A2A protocol
"""

import json
import uuid
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


class ZendeskA2AClient:
    """Client for Zendesk A2A agent"""

    def __init__(self, agent_url: str, timeout: float = 300.0):
        self.agent_url = agent_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def check_health(self) -> bool:
        """Check if agent is reachable"""
        try:
            response = await self._client.get(
                f"{self.agent_url}/.well-known/agent-card.json"
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning("Agent health check failed", error=str(e))
            return False

    async def get_agent_card(self) -> dict:
        """Get agent capabilities"""
        response = await self._client.get(
            f"{self.agent_url}/.well-known/agent-card.json"
        )
        response.raise_for_status()
        return response.json()

    async def send_message(self, text: str) -> dict:
        """
        Send a message to the agent using A2A protocol

        Protocol: JSON-RPC 2.0
        Method: message/send
        """
        message_id = str(uuid.uuid4())

        payload = {
            "jsonrpc": "2.0",
            "id": message_id,
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": message_id,
                    "role": "user",
                    "parts": [{"text": text}],
                }
            },
        }

        logger.debug("Sending A2A message", message_id=message_id, text=text[:100])

        response = await self._client.post(
            self.agent_url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        result = response.json()

        if "error" in result:
            raise Exception(f"A2A error: {result['error']}")

        return result.get("result", {})

    async def fetch_tickets(self, days: int = 365) -> list[dict]:
        """
        Fetch tickets from Zendesk agent

        Args:
            days: Number of days of history to fetch

        Returns:
            List of raw ticket dictionaries
        """
        logger.info("Fetching tickets from agent", days=days)

        # Request ticket export from agent
        prompt = f"Export all tickets from the last {days} days as JSON. Include ticket ID, subject, description, status, priority, created_at, updated_at, comments, tags, and custom fields."

        result = await self.send_message(prompt)

        # Parse the response
        tickets = self._parse_ticket_response(result)

        logger.info("Fetched tickets", count=len(tickets))
        return tickets

    def _parse_ticket_response(self, result: dict) -> list[dict]:
        """Parse ticket data from agent response"""
        tickets = []

        # Extract text parts from response
        parts = result.get("parts", [])
        for part in parts:
            if part.get("kind") == "text":
                text = part.get("text", "")
                # Try to parse as JSON
                tickets.extend(self._extract_tickets_from_text(text))
            elif part.get("kind") == "data":
                # Direct data response
                data = part.get("data", {})
                if isinstance(data, list):
                    tickets.extend(data)
                elif isinstance(data, dict) and "tickets" in data:
                    tickets.extend(data["tickets"])

        return tickets

    def _extract_tickets_from_text(self, text: str) -> list[dict]:
        """Extract ticket JSON from text response"""
        tickets = []

        # Try to find JSON in the response
        try:
            # First try: entire response is JSON
            data = json.loads(text)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "tickets" in data:
                return data["tickets"]
        except json.JSONDecodeError:
            pass

        # Second try: find JSON array in text
        try:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                if isinstance(data, list):
                    return data
        except json.JSONDecodeError:
            pass

        # Third try: find JSON objects line by line
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    tickets.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return tickets

    async def close(self):
        """Close the client"""
        await self._client.aclose()

