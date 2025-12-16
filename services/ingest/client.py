"""
Zendesk MCP Client
Communicates with the Zendesk MCP server using SSE transport
"""

import asyncio
import json
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx
import structlog

logger = structlog.get_logger()


class ZendeskMCPClient:
    """
    Client for Zendesk MCP server using SSE transport

    MCP SSE Protocol:
    1. GET /sse to establish SSE connection and receive session_id
    2. POST /messages/?session_id=xxx with JSON-RPC requests
    3. Responses arrive via SSE stream
    """

    def __init__(self, mcp_url: str, timeout: float = 300.0):
        self.mcp_url = mcp_url.rstrip("/").replace("/sse", "")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._session_id: Optional[str] = None
        self._responses: deque = deque()
        self._sse_task: Optional[asyncio.Task] = None
        self._request_id = 0
        # MCP requires Host: localhost for local servers
        self._headers = {"Host": "localhost:10005"}

    async def connect(self) -> bool:
        """Establish SSE connection and get session ID"""
        try:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._sse_task = asyncio.create_task(self._sse_reader())

            # Wait for session ID
            for _ in range(50):  # 5 seconds max
                if self._session_id:
                    break
                await asyncio.sleep(0.1)

            if not self._session_id:
                logger.error("Failed to get MCP session ID")
                return False

            # Initialize MCP
            await self._initialize()
            logger.info("Connected to MCP server", session=self._session_id[:8])
            return True

        except Exception as e:
            logger.error("Failed to connect to MCP server", error=str(e))
            return False

    async def _sse_reader(self):
        """Background task to read SSE events"""
        try:
            async with self._client.stream(
                "GET", f"{self.mcp_url}/sse", headers=self._headers
            ) as response:
                async for line in response.aiter_lines():
                    line = line.strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if "session_id=" in data:
                            self._session_id = data.split("session_id=")[1]
                        else:
                            try:
                                msg = json.loads(data)
                                self._responses.append(msg)
                            except json.JSONDecodeError:
                                pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("SSE reader error", error=str(e))

    async def _initialize(self):
        """Initialize MCP session"""
        self._request_id += 1
        await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "zti-ingest", "version": "1.0.0"},
            },
        )
        await asyncio.sleep(0.5)

    async def _send_request(self, method: str, params: dict = None) -> int:
        """Send JSON-RPC request via POST"""
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        await self._client.post(
            f"{self.mcp_url}/messages/?session_id={self._session_id}",
            json=request,
            headers=self._headers,
        )
        return self._request_id

    async def _wait_for_response(self, request_id: int, timeout: float = 60.0) -> dict:
        """Wait for response with matching request ID"""
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            for i, resp in enumerate(self._responses):
                if resp.get("id") == request_id:
                    self._responses.remove(resp)
                    if "error" in resp:
                        raise Exception(f"MCP error: {resp['error']}")
                    return resp.get("result", {})
            await asyncio.sleep(0.1)
        raise TimeoutError(f"Timeout waiting for response {request_id}")

    async def call_tool(self, name: str, arguments: dict = None) -> Any:
        """Call an MCP tool and wait for result"""
        request_id = await self._send_request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
        )
        return await self._wait_for_response(request_id, timeout=self.timeout)

    async def check_health(self) -> bool:
        """Check if MCP server is reachable"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.mcp_url}/sse",
                    headers=self._headers,
                )
                return response.status_code == 200
        except Exception as e:
            logger.warning("MCP health check failed", error=str(e))
            return False

    async def fetch_tickets(
        self,
        days: int = 365,
        include_comments: bool = True,
        chunk_size: int = 30,
    ) -> list[dict]:
        """
        Fetch tickets using MCP tools with automatic chunking for large ranges

        Args:
            days: Number of days of history to fetch
            include_comments: Whether to include ticket comments
            chunk_size: Days per chunk (default 30)

        Returns:
            List of raw ticket dictionaries
        """
        logger.info("Fetching tickets via MCP", days=days, chunk_size=chunk_size)

        all_tickets = []

        # For small ranges, use crawl_tickets directly
        if days <= chunk_size:
            result = await self.call_tool(
                "crawl_tickets",
                {
                    "days_back": days,
                    "status": ["new", "open", "pending", "hold", "solved", "closed"],
                    "include_description": True,
                    "include_comments": include_comments,
                },
            )
            tickets = self._parse_crawl_result(result)
            logger.info("Fetched tickets", count=len(tickets))
            return tickets

        # For large ranges, chunk by month and use bulk_export
        from datetime import datetime, timedelta

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        current_start = start_date
        chunk_num = 0

        while current_start < end_date:
            current_end = min(current_start + timedelta(days=chunk_size), end_date)
            chunk_num += 1

            logger.info(
                "Fetching chunk",
                chunk=chunk_num,
                start=current_start.strftime("%Y-%m-%d"),
                end=current_end.strftime("%Y-%m-%d"),
            )

            try:
                result = await self.call_tool(
                    "bulk_export_tickets",
                    {
                        "start_date": current_start.strftime("%Y-%m-%d"),
                        "end_date": current_end.strftime("%Y-%m-%d"),
                        "include_description": True,
                        "chunk_days": 7,
                        "max_tickets": 10000,
                    },
                )
                tickets = self._parse_crawl_result(result)
                all_tickets.extend(tickets)
                logger.info("Chunk complete", chunk=chunk_num, tickets=len(tickets))
            except Exception as e:
                logger.warning("Chunk failed, continuing", chunk=chunk_num, error=str(e))

            current_start = current_end

        # Deduplicate by ticket ID
        seen_ids = set()
        unique_tickets = []
        for ticket in all_tickets:
            ticket_id = ticket.get("id")
            if ticket_id and ticket_id not in seen_ids:
                seen_ids.add(ticket_id)
                unique_tickets.append(ticket)

        logger.info("Fetched all tickets", total=len(unique_tickets), chunks=chunk_num)
        return unique_tickets

    async def fetch_tickets_bulk(
        self,
        start_date: str,
        end_date: str,
        include_description: bool = True,
    ) -> list[dict]:
        """
        Fetch tickets using bulk_export_tickets tool (for large ranges)

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            include_description: Include ticket description

        Returns:
            List of ticket dictionaries
        """
        logger.info("Bulk exporting tickets", start=start_date, end=end_date)

        result = await self.call_tool(
            "bulk_export_tickets",
            {
                "start_date": start_date,
                "end_date": end_date,
                "include_description": include_description,
                "chunk_days": 7,  # Auto-chunk by week
                "max_tickets": 100000,
            },
        )

        tickets = self._parse_bulk_result(result)
        logger.info("Bulk exported tickets", count=len(tickets))
        return tickets

    def _parse_crawl_result(self, result: Any) -> list[dict]:
        """Parse result from crawl_tickets tool"""
        if isinstance(result, list):
            # Check if it's a list of content items (MCP tool response format)
            for item in result:
                if isinstance(item, dict) and item.get("type") == "text":
                    try:
                        parsed = json.loads(item.get("text", "[]"))
                        # Handle nested {"tickets": [...]} structure
                        if isinstance(parsed, dict) and "tickets" in parsed:
                            return parsed["tickets"]
                        return parsed
                    except json.JSONDecodeError:
                        pass
            return result
        elif isinstance(result, dict):
            if "tickets" in result:
                return result["tickets"]
            if "content" in result:
                return self._parse_crawl_result(result["content"])
        return []

    def _parse_bulk_result(self, result: Any) -> list[dict]:
        """Parse result from bulk_export_tickets tool"""
        return self._parse_crawl_result(result)

    async def close(self):
        """Close the client and cleanup"""
        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()

