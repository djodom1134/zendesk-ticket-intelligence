#!/usr/bin/env python3
"""
Re-summarize all tickets using MCP for full context and anti-hallucination prompt.
Uses parallel fetching: fetches next ticket while current one is being summarized.
Processes image attachments using vision model (ministral-3).

Usage:
    python scripts/resummarize_all.py --mcp-url http://192.168.87.79:10005/sse --ollama-url http://localhost:11434

This script:
1. Fetches all ticket IDs from ArangoDB
2. Parallel pipeline: fetch next ticket while summarizing current one
3. Downloads and processes image attachments with vision model
4. Generates summaries using anti-hallucination prompt (includes image descriptions)
5. Stores summaries back to ArangoDB
"""
import argparse
import asyncio
import base64
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.ingest.client import ZendeskMCPClient
from services.embed_cluster.summarizer import TicketSummarizer

# Load from environment
DEFAULT_MCP_URL = os.getenv("MCP_URL", "http://192.168.87.79:10005/sse")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-oss:120b")
DEFAULT_VISION_MODEL = os.getenv("VISION_MODEL", "ministral-3:8b")


async def fetch_ticket_full_context(client: ZendeskMCPClient, ticket_id: int) -> dict | None:
    """Fetch a single ticket with full context from MCP."""
    try:
        result = await client.call_tool("get_ticket_full_context", {"ticket_id": ticket_id})
        if result.get("isError"):
            return None

        # Parse the result
        if "content" in result:
            for item in result["content"]:
                if item.get("type") == "text":
                    return json.loads(item["text"])
        return result
    except Exception as e:
        print(f"  ⚠️ Exception fetching {ticket_id}: {e}")
        return None


async def download_image(url: str, timeout: float = 30.0) -> bytes | None:
    """Download an image from a Zendesk attachment URL."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, follow_redirects=True)
            if response.status_code == 200:
                return response.content
            return None
    except Exception as e:
        print(f"    ⚠️ Failed to download image: {e}")
        return None


def process_image_with_vision(
    image_data: bytes,
    filename: str,
    ollama_url: str,
    vision_model: str,
) -> str:
    """
    Process an image using the vision model to get a description.
    Returns a text description of the image content.
    """
    # Encode image as base64
    image_b64 = base64.b64encode(image_data).decode("utf-8")

    prompt = """Describe this image from a support ticket. Focus on:
1. What the image shows (screenshot, error message, hardware, document, etc.)
2. Any visible text (error messages, UI elements, filenames)
3. Technical details relevant to troubleshooting

Keep description concise (2-4 sentences). Output ONLY the description, no preamble."""

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": vision_model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {"num_predict": 200},
                },
            )
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
    except Exception as e:
        print(f"    ⚠️ Vision processing failed: {e}")

    return f"[Image: {filename} - could not process]"


async def process_ticket_images(
    ticket: dict,
    ollama_url: str,
    vision_model: str,
    max_images: int = 5,
) -> list[dict]:
    """
    Download and process image attachments from a ticket.
    Returns list of {"filename": str, "description": str}.
    """
    image_descriptions = []

    # Get image attachments from ticket
    image_attachments = ticket.get("image_attachments", [])
    if not image_attachments:
        return image_descriptions

    # Limit number of images to process
    images_to_process = image_attachments[:max_images]

    for attachment in images_to_process:
        filename = attachment.get("file_name", "unknown.jpg")
        content_url = attachment.get("content_url")

        if not content_url:
            continue

        print(f"    📷 Processing image: {filename}")

        # Download image
        image_data = await download_image(content_url)
        if not image_data:
            image_descriptions.append({
                "filename": filename,
                "description": f"[Image: {filename} - download failed]",
            })
            continue

        # Process with vision model
        description = process_image_with_vision(
            image_data=image_data,
            filename=filename,
            ollama_url=ollama_url,
            vision_model=vision_model,
        )

        image_descriptions.append({
            "filename": filename,
            "description": description,
        })

    return image_descriptions


class ParallelPipeline:
    """
    Parallel pipeline with 3 stages running concurrently:
    1. Fetch tickets from MCP
    2. Process images with vision model (ministral-3)
    3. Summarize with text model (gpt-oss)

    This allows image processing for ticket N+1 while ticket N is being summarized.
    """

    def __init__(
        self,
        mcp_client: ZendeskMCPClient,
        summarizer: TicketSummarizer,
        ollama_url: str,
        vision_model: str = DEFAULT_VISION_MODEL,
        prefetch_count: int = 3,
        process_images: bool = True,
    ):
        self.mcp_client = mcp_client
        self.summarizer = summarizer
        self.ollama_url = ollama_url
        self.vision_model = vision_model
        self.prefetch_count = prefetch_count
        self.process_images = process_images

        # Two-stage queue: fetch -> image processing -> summarization
        self.fetch_queue = asyncio.Queue(maxsize=prefetch_count)
        self.summary_queue = asyncio.Queue(maxsize=prefetch_count)

        self.results = {}
        self.fetch_errors = 0
        self.summarize_errors = 0
        self.images_processed = 0
        self._stop_signal = object()  # Sentinel for stopping workers

    async def fetch_worker(self, ticket_ids: list[int]):
        """Stage 1: Fetch tickets from MCP and put in fetch_queue."""
        for ticket_id in ticket_ids:
            ticket = await fetch_ticket_full_context(self.mcp_client, ticket_id)
            if ticket:
                await self.fetch_queue.put((ticket_id, ticket))
            else:
                self.fetch_errors += 1
                await self.fetch_queue.put((ticket_id, None))

        # Signal end of fetching
        await self.fetch_queue.put((None, self._stop_signal))

    async def image_worker(self, show_live: bool = True):
        """
        Stage 2: Process images with vision model.
        Runs in parallel with summarization - while gpt-oss summarizes one ticket,
        ministral-3 processes images for the next ticket.
        """
        while True:
            ticket_id, ticket = await self.fetch_queue.get()

            # Check for stop signal
            if ticket is self._stop_signal:
                await self.summary_queue.put((None, self._stop_signal, []))
                break

            if ticket is None:
                # Fetch failed, pass through
                await self.summary_queue.put((ticket_id, None, []))
                continue

            # Process images if enabled
            image_descriptions = []
            image_attachments = ticket.get("image_attachments", [])

            if self.process_images and image_attachments:
                num_images = len(image_attachments)
                if show_live:
                    print(f"    🖼️  Processing {num_images} images for ticket {ticket_id}...")

                image_descriptions = await process_ticket_images(
                    ticket=ticket,
                    ollama_url=self.ollama_url,
                    vision_model=self.vision_model,
                    max_images=5,
                )
                self.images_processed += len(image_descriptions)

                # Add to ticket for summarizer
                if image_descriptions:
                    ticket["_image_descriptions"] = image_descriptions

            # Pass to summary queue
            await self.summary_queue.put((ticket_id, ticket, image_descriptions))

    async def process_tickets(
        self,
        ticket_ids: list[int],
        db_collection,
        dry_run: bool = False,
        show_live: bool = True,
    ) -> tuple[int, int]:
        """
        Process tickets with 3-stage parallel pipeline:
        - Stage 1: Fetch tickets from MCP
        - Stage 2: Process images with vision model (ministral-3)
        - Stage 3: Summarize with text model (gpt-oss)

        This allows image processing for ticket N+1 while ticket N is being summarized.

        Returns:
            Tuple of (processed_count, failed_count)
        """
        total = len(ticket_ids)
        processed = 0
        failed = 0
        start_time = time.time()

        print(f"\n🚀 Starting 3-STAGE PARALLEL pipeline (prefetch={self.prefetch_count})")
        print(f"   Stage 1: Fetch tickets from MCP")
        print(f"   Stage 2: Process images with {self.vision_model}")
        print(f"   Stage 3: Summarize with {self.summarizer.model}")
        print(f"   Images enabled: {self.process_images}")
        print(f"\n   ⚡ Vision + Summary run in PARALLEL on different models!")

        # Start Stage 1: Fetch worker
        fetch_task = asyncio.create_task(self.fetch_worker(ticket_ids))

        # Start Stage 2: Image processing worker
        image_task = asyncio.create_task(self.image_worker(show_live=show_live))

        # Stage 3: Summarization (main loop)
        while True:
            # Get next ticket from summary queue (already has images processed)
            ticket_id, ticket, image_descriptions = await self.summary_queue.get()

            # Check for end signal
            if ticket is self._stop_signal:
                break

            idx = processed + failed + 1

            if ticket is None:
                # Fetch failed
                failed += 1
                print(f"\n[{idx}/{total}] ❌ Failed to fetch ticket {ticket_id}")
                continue

            # Show ticket info
            num_images = len(image_descriptions)
            if show_live:
                img_info = f" | 📷 {num_images} images analyzed" if num_images > 0 else ""
                print(f"\n[{idx}/{total}] 📋 Summarizing ticket {ticket_id}{img_info}")

            try:
                # Summarize (gpt-oss:120b)
                summary, metrics = self.summarizer.summarize_full_ticket(
                    ticket=ticket,
                    is_last=(idx == total),
                    show_live=show_live,
                )

                # Store summary and image info
                if not dry_run and db_collection:
                    update_doc = {
                        "_key": str(ticket_id),
                        "summary": summary,
                        "summary_version": "v3_with_images",
                    }
                    if image_descriptions:
                        update_doc["image_descriptions"] = image_descriptions
                    db_collection.update(update_doc)

                processed += 1

            except Exception as e:
                print(f"  ⚠️ Summarization error: {e}")
                failed += 1

            # Progress stats
            if idx % 10 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed * 3600 if elapsed > 0 else 0
                eta = (total - idx) / (processed / elapsed) if processed > 0 else 0
                print(f"\n📊 Progress: {idx}/{total} | "
                      f"Rate: {rate:.0f}/hr | "
                      f"ETA: {eta/60:.1f}min | "
                      f"Images: {self.images_processed} | "
                      f"Fetch Q: {self.fetch_queue.qsize()} | "
                      f"Summary Q: {self.summary_queue.qsize()}")

        # Wait for workers to complete
        await fetch_task
        await image_task

        return processed, failed


async def main():
    parser = argparse.ArgumentParser(description="Re-summarize all tickets with parallel fetch + summarize + vision")
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL, help="MCP server URL")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama URL")
    parser.add_argument("--model", default=DEFAULT_SUMMARY_MODEL, help="Summarization model")
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL, help="Vision model for images")
    parser.add_argument("--no-images", action="store_true", help="Disable image processing")
    parser.add_argument("--arango-host", default="localhost", help="ArangoDB host")
    parser.add_argument("--arango-port", type=int, default=8529, help="ArangoDB port")
    parser.add_argument("--prefetch", type=int, default=3, help="Number of tickets to prefetch")
    parser.add_argument("--start-from", type=int, default=0, help="Start from ticket index")
    parser.add_argument("--max-tickets", type=int, default=0, help="Max tickets to process (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Don't save summaries, just test")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    args = parser.parse_args()

    print("=" * 60)
    print("🔄 PARALLEL RE-SUMMARIZATION PIPELINE WITH VISION")
    print("=" * 60)
    print(f"MCP URL: {args.mcp_url}")
    print(f"Ollama URL: {args.ollama_url}")
    print(f"Summary Model: {args.model}")
    print(f"Vision Model: {args.vision_model}")
    print(f"Process Images: {not args.no_images}")
    print(f"Prefetch: {args.prefetch} tickets")
    print(f"ArangoDB: {args.arango_host}:{args.arango_port}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Connect to ArangoDB to get ticket IDs
    from arango import ArangoClient
    arango = ArangoClient(hosts=f"http://{args.arango_host}:{args.arango_port}")
    db = arango.db("zti", username="root", password="zti_dev_password")

    # Get all ticket IDs
    tickets_col = db.collection("tickets")
    cursor = db.aql.execute("FOR t IN tickets RETURN t.ticket_id")
    ticket_ids = [int(tid) for tid in cursor if tid]

    print(f"📋 Found {len(ticket_ids)} tickets in ArangoDB")

    # Apply limits
    if args.start_from > 0:
        ticket_ids = ticket_ids[args.start_from:]
        print(f"   Starting from index {args.start_from}")
    if args.max_tickets > 0:
        ticket_ids = ticket_ids[:args.max_tickets]
        print(f"   Limited to {args.max_tickets} tickets")

    print(f"   Processing: {len(ticket_ids)} tickets")
    print()

    # Connect to MCP
    print("🔌 Connecting to MCP server...")
    mcp_client = ZendeskMCPClient(args.mcp_url)
    if not await mcp_client.connect():
        print("❌ Failed to connect to MCP server")
        return 1
    print("✅ Connected to MCP")

    # Initialize summarizer
    summarizer = TicketSummarizer(
        ollama_url=args.ollama_url,
        model=args.model,
        max_output_tokens=2000,
    )

    # Create parallel pipeline with vision support
    pipeline = ParallelPipeline(
        mcp_client=mcp_client,
        summarizer=summarizer,
        ollama_url=args.ollama_url,
        vision_model=args.vision_model,
        prefetch_count=args.prefetch,
        process_images=not args.no_images,
    )

    start_time = time.time()

    try:
        processed, failed = await pipeline.process_tickets(
            ticket_ids=ticket_ids,
            db_collection=tickets_col if not args.dry_run else None,
            dry_run=args.dry_run,
            show_live=not args.quiet,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        processed, failed = 0, 0
    finally:
        await mcp_client.close()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ Complete!")
    print(f"   Processed: {processed}")
    print(f"   Failed: {failed}")
    print(f"   Images processed: {pipeline.images_processed}")
    print(f"   Time: {elapsed/60:.1f} minutes")
    print(f"   Rate: {processed/elapsed*3600:.0f} tickets/hour" if elapsed > 0 else "")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

