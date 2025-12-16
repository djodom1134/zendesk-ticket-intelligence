#!/usr/bin/env python3
"""
Re-summarize all tickets with MAXIMUM GPU UTILIZATION.

Architecture:
- Fetch Thread: Constantly fetching tickets from MCP (I/O bound)
- Vision Thread: Dedicated thread constantly processing images with ministral-3 (GPU)
- Summary Thread: Dedicated thread constantly summarizing with gpt-oss:120b (GPU)

Both GPU threads run independently with thread-safe queues, keeping both models busy.

Usage:
    python scripts/resummarize_all.py --mcp-url http://192.168.87.79:10005/sse --ollama-url http://localhost:11434
"""
import argparse
import asyncio
import base64
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.ingest.client import ZendeskMCPClient
from services.embed_cluster.summarizer import TicketSummarizer

# Load from environment
DEFAULT_MCP_URL = os.getenv("MCP_URL", "http://192.168.87.79:10005/sse")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-oss:120b")
DEFAULT_VISION_MODEL = os.getenv("VISION_MODEL", "ministral-3:8b")

# Sentinel for stopping threads
STOP_SIGNAL = object()


async def fetch_ticket_full_context(client: ZendeskMCPClient, ticket_id: int) -> dict | None:
    """Fetch a single ticket with full context from MCP."""
    try:
        result = await client.call_tool("get_ticket_full_context", {"ticket_id": ticket_id})
        print(f"    [FETCH DEBUG] Raw result type: {type(result)}")
        print(f"    [FETCH DEBUG] Raw result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")

        if result.get("isError"):
            print(f"    [FETCH DEBUG] isError=True for ticket {ticket_id}")
            return None

        # Parse the result
        if "content" in result:
            for item in result["content"]:
                if item.get("type") == "text":
                    text_content = item["text"]
                    parsed = json.loads(text_content)

                    # Check for MCP error response
                    if "error" in parsed:
                        print(f"    ❌ [MCP ERROR] {parsed.get('error')}")
                        print(f"    ❌ [MCP ERROR] Code: {parsed.get('code')}")
                        return None

                    # Log key fields
                    subject = parsed.get("subject", "NO SUBJECT")
                    images = parsed.get("image_attachments", [])
                    comments = parsed.get("comments", [])
                    print(f"    [FETCH] Ticket {ticket_id}: subject='{(subject[:50] if subject else 'None')}', images={len(images)}, comments={len(comments)}")
                    return parsed

        print(f"    [FETCH DEBUG] No content in result, returning raw: {str(result)[:500]}")
        return result
    except Exception as e:
        print(f"  ⚠️ Exception fetching {ticket_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


async def download_image(url: str, timeout: float = 30.0) -> bytes | None:
    """Download an image from a Zendesk attachment URL (async)."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, follow_redirects=True)
            if response.status_code == 200:
                return response.content
            return None
    except Exception as e:
        print(f"    ⚠️ Failed to download image: {e}")
        return None


async def process_image_with_vision_async(
    image_data: bytes,
    filename: str,
    ollama_url: str,
    vision_model: str,
) -> str:
    """
    Process an image using the vision model (ASYNC).
    Uses httpx.AsyncClient so it doesn't block the event loop.
    """
    image_b64 = base64.b64encode(image_data).decode("utf-8")

    prompt = """Describe this image from a support ticket. Focus on:
1. What the image shows (screenshot, error message, hardware, document, etc.)
2. Any visible text (error messages, UI elements, filenames)
3. Technical details relevant to troubleshooting

Keep description concise (2-4 sentences). Output ONLY the description, no preamble."""

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
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


@dataclass
class TicketWorkItem:
    """Work item passed between pipeline stages."""
    ticket_id: int
    ticket: dict | None
    image_descriptions: list[dict]


class ThreadedGPUPipeline:
    """
    Maximum GPU utilization pipeline with DEDICATED THREADS for each GPU operation.

    Architecture:
    - Fetch Thread: Constantly fetching tickets from MCP (I/O bound)
    - Vision Thread: DEDICATED thread constantly processing images with ministral-3
    - Summary Thread: DEDICATED thread constantly summarizing with gpt-oss:120b

    Both GPU threads run independently with thread-safe queues between them.
    This keeps BOTH models busy simultaneously on the GPU.
    """

    def __init__(
        self,
        summarizer: TicketSummarizer,
        ollama_url: str,
        vision_model: str = DEFAULT_VISION_MODEL,
        fetch_buffer_size: int = 10,
        vision_buffer_size: int = 10,
        process_images: bool = True,
    ):
        self.summarizer = summarizer
        self.ollama_url = ollama_url
        self.vision_model = vision_model
        self.process_images = process_images

        # Thread-safe queues between stages
        # Fetch -> Vision: raw tickets
        self.fetch_queue: queue.Queue = queue.Queue(maxsize=fetch_buffer_size)
        # Vision -> Summary: tickets with processed images
        self.vision_queue: queue.Queue = queue.Queue(maxsize=vision_buffer_size)
        # Summary -> Results: completed summaries
        self.result_queue: queue.Queue = queue.Queue()

        # Metrics (thread-safe via GIL for simple operations)
        self.fetch_count = 0
        self.vision_count = 0
        self.summary_count = 0
        self.images_processed = 0
        self.failed_count = 0

        # Control
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def _download_image_sync(self, url: str, timeout: float = 30.0) -> bytes | None:
        """Download image synchronously."""
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(url, follow_redirects=True)
                if response.status_code == 200:
                    return response.content
        except Exception as e:
            pass
        return None

    def _process_image_vision_sync(self, image_data: bytes, filename: str) -> str:
        """Process image with vision model (synchronous, runs in vision thread)."""
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        prompt = """Describe this image from a support ticket. Focus on:
1. What the image shows (screenshot, error message, hardware, document, etc.)
2. Any visible text (error messages, UI elements, filenames)
3. Technical details relevant to troubleshooting

Keep description concise (2-4 sentences). Output ONLY the description."""

        try:
            with httpx.Client(timeout=90.0) as client:
                response = client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.vision_model,
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
            print(f"    ⚠️ Vision error: {e}")

        return f"[Image: {filename} - processing failed]"

    def vision_thread_worker(self, show_live: bool = True):
        """
        DEDICATED VISION THREAD - constantly processes images.

        This thread runs independently, keeping ministral-3 busy.
        Pulls from fetch_queue, processes images, pushes to vision_queue.
        """
        print(f"    [VISION] Started, waiting for tickets...")
        while not self._stop_event.is_set():
            try:
                wait_start = time.time()
                item = self.fetch_queue.get(timeout=5.0)
                wait_time = time.time() - wait_start
                if wait_time > 0.5:
                    print(f"    [VISION] ⏱️ WAITED {wait_time:.1f}s for fetch_queue")
            except queue.Empty:
                continue

            if item is STOP_SIGNAL:
                print(f"    [VISION] Received STOP_SIGNAL, exiting")
                self.vision_queue.put(STOP_SIGNAL)
                break

            ticket_id, ticket = item

            if ticket is None:
                self.vision_queue.put(TicketWorkItem(ticket_id, None, []))
                continue

            # Process images
            image_descriptions = []
            image_attachments = ticket.get("image_attachments", [])
            num_images = len(image_attachments)

            if self.process_images and image_attachments:
                process_count = min(num_images, 5)
                print(f"    🖼️  [VISION] Ticket {ticket_id}: {process_count} images to process...")

                vision_start = time.time()
                for i, attachment in enumerate(image_attachments[:5]):
                    filename = attachment.get("file_name", "unknown.jpg")
                    content_url = attachment.get("content_url")

                    if not content_url:
                        continue

                    # Download (I/O)
                    dl_start = time.time()
                    image_data = self._download_image_sync(content_url)
                    dl_time = time.time() - dl_start

                    if not image_data:
                        image_descriptions.append({
                            "filename": filename,
                            "description": f"[Image: {filename} - download failed]",
                        })
                        continue

                    # Process with vision model (GPU)
                    gpu_start = time.time()
                    description = self._process_image_vision_sync(image_data, filename)
                    gpu_time = time.time() - gpu_start

                    print(f"    [VISION] Image {i+1}: dl={dl_time:.1f}s, gpu={gpu_time:.1f}s")
                    image_descriptions.append({
                        "filename": filename,
                        "description": description,
                    })

                    with self._lock:
                        self.images_processed += 1

                vision_total = time.time() - vision_start
                if image_descriptions:
                    ticket["_image_descriptions"] = image_descriptions
                    print(f"    ✅ [VISION] Ticket {ticket_id}: {len(image_descriptions)} images in {vision_total:.1f}s")
            else:
                print(f"    [VISION] Ticket {ticket_id}: 0 images (pass-through)")

            with self._lock:
                self.vision_count += 1

            self.vision_queue.put(TicketWorkItem(ticket_id, ticket, image_descriptions))

    def summary_thread_worker(self, db_collection, dry_run: bool, show_live: bool, total: int):
        """
        DEDICATED SUMMARY THREAD - constantly summarizes tickets.

        This thread runs independently, keeping gpt-oss:120b busy.
        Pulls from vision_queue, summarizes, pushes to result_queue.
        """
        print(f"    [SUMMARY] Started, waiting for tickets...")
        while not self._stop_event.is_set():
            try:
                wait_start = time.time()
                item = self.vision_queue.get(timeout=5.0)
                wait_time = time.time() - wait_start
                if wait_time > 0.5:
                    print(f"    [SUMMARY] ⏱️ WAITED {wait_time:.1f}s for vision_queue")
            except queue.Empty:
                continue

            if item is STOP_SIGNAL:
                print(f"    [SUMMARY] Received STOP_SIGNAL, exiting")
                self.result_queue.put(STOP_SIGNAL)
                break

            work_item: TicketWorkItem = item

            if work_item.ticket is None:
                with self._lock:
                    self.failed_count += 1
                self.result_queue.put(("failed", work_item.ticket_id, None))
                continue

            with self._lock:
                idx = self.summary_count + self.failed_count + 1
                vision_q = self.vision_queue.qsize()

            img_info = f" | 📷 {len(work_item.image_descriptions)}" if work_item.image_descriptions else ""
            print(f"\n[{idx}/{total}] 📋 [SUMMARY] Ticket {work_item.ticket_id}{img_info} [VQ: {vision_q}]")

            try:
                # Summarize with gpt-oss:120b (GPU)
                gpu_start = time.time()
                summary, metrics = self.summarizer.summarize_full_ticket(
                    ticket=work_item.ticket,
                    is_last=(idx == total),
                    show_live=show_live,
                )
                gpu_time = time.time() - gpu_start
                print(f"    [SUMMARY] ⏱️ Ticket {work_item.ticket_id}: GPU={gpu_time:.1f}s, len={len(summary)}")

                # Store to DB
                if not dry_run and db_collection:
                    update_doc = {
                        "_key": str(work_item.ticket_id),
                        "summary": summary,
                        "summary_version": "v3_with_images",
                    }
                    if work_item.image_descriptions:
                        update_doc["image_descriptions"] = work_item.image_descriptions
                    db_collection.update(update_doc)

                with self._lock:
                    self.summary_count += 1

                self.result_queue.put(("success", work_item.ticket_id, summary))

            except Exception as e:
                print(f"  ⚠️ [SUMMARY] Error: {e}")
                import traceback
                traceback.print_exc()
                with self._lock:
                    self.failed_count += 1
                self.result_queue.put(("failed", work_item.ticket_id, str(e)))

    async def process_tickets(
        self,
        mcp_client: ZendeskMCPClient,
        ticket_ids: list[int],
        db_collection,
        dry_run: bool = False,
        show_live: bool = True,
    ) -> tuple[int, int]:
        """
        Process tickets with MAXIMUM GPU UTILIZATION.

        Three dedicated threads run in parallel:
        1. Async fetch loop feeds fetch_queue
        2. Vision thread constantly processes images (ministral-3)
        3. Summary thread constantly summarizes (gpt-oss:120b)

        Both GPU models stay busy simultaneously!
        """
        total = len(ticket_ids)
        start_time = time.time()

        print(f"\n🚀 MAXIMUM GPU UTILIZATION PIPELINE")
        print(f"=" * 60)
        print(f"   🔄 Fetch Thread: Constantly fetching from MCP")
        print(f"   🖼️  Vision Thread: DEDICATED {self.vision_model} (always busy)")
        print(f"   📋 Summary Thread: DEDICATED {self.summarizer.model} (always busy)")
        print(f"   Buffers: Fetch={self.fetch_queue.maxsize} | Vision={self.vision_queue.maxsize}")
        print(f"\n   ⚡ BOTH GPU MODELS RUN SIMULTANEOUSLY!")
        print(f"=" * 60)

        # PRE-WARM MODELS - Load both models into GPU memory
        print(f"\n🔥 PRE-WARMING MODELS...")
        print(f"   Loading {self.vision_model} into GPU...")
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.vision_model,
                        "prompt": "Hello",
                        "stream": False,
                        "options": {"num_predict": 1},
                    },
                )
                if response.status_code == 200:
                    print(f"   ✅ {self.vision_model} loaded!")
                else:
                    print(f"   ⚠️ Vision model response: {response.status_code}")
        except Exception as e:
            print(f"   ⚠️ Failed to load vision model: {e}")

        print(f"   Loading {self.summarizer.model} into GPU...")
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.summarizer.model,
                        "prompt": "Hello",
                        "stream": False,
                        "options": {"num_predict": 1},
                    },
                )
                if response.status_code == 200:
                    print(f"   ✅ {self.summarizer.model} loaded!")
                else:
                    print(f"   ⚠️ Summary model response: {response.status_code}")
        except Exception as e:
            print(f"   ⚠️ Failed to load summary model: {e}")

        print(f"🔥 Models pre-warmed!\n")

        # Start vision thread (GPU worker 1)
        vision_thread = threading.Thread(
            target=self.vision_thread_worker,
            args=(show_live,),
            name="VisionThread",
            daemon=True,
        )
        vision_thread.start()
        print(f"   ✅ Vision thread started")

        # Start summary thread (GPU worker 2)
        summary_thread = threading.Thread(
            target=self.summary_thread_worker,
            args=(db_collection, dry_run, show_live, total),
            name="SummaryThread",
            daemon=True,
        )
        summary_thread.start()
        print(f"   ✅ Summary thread started")

        # Async fetch loop - feeds the pipeline
        print(f"   ✅ Starting async fetch loop...")

        async def fetch_all():
            print(f"    [FETCH] Starting to fetch {len(ticket_ids)} tickets...")
            for i, ticket_id in enumerate(ticket_ids):
                fetch_start = time.time()
                ticket = await fetch_ticket_full_context(mcp_client, ticket_id)
                fetch_time = time.time() - fetch_start

                if ticket:
                    self.fetch_queue.put((ticket_id, ticket))
                    print(f"    [FETCH] Ticket {ticket_id} ({i+1}/{len(ticket_ids)}): {fetch_time:.1f}s, queue={self.fetch_queue.qsize()}")
                else:
                    self.fetch_queue.put((ticket_id, None))
                    print(f"    [FETCH] Ticket {ticket_id} FAILED after {fetch_time:.1f}s")
                with self._lock:
                    self.fetch_count += 1
            # Signal end
            print(f"    [FETCH] All {len(ticket_ids)} tickets fetched, sending STOP_SIGNAL...")
            self.fetch_queue.put(STOP_SIGNAL)

        async def get_result_async():
            """Get result from queue without blocking the event loop."""
            while True:
                try:
                    return self.result_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.1)  # Yield to event loop

        # Run fetch in background
        print(f"    [MAIN] Creating fetch task...")
        fetch_task = asyncio.create_task(fetch_all())
        print(f"    [MAIN] Fetch task created, starting result monitoring loop...")

        # Monitor progress and collect results (async-friendly)
        results = []
        loop_count = 0
        last_progress = time.time()

        while True:
            loop_count += 1

            # Non-blocking check for results, yield to event loop
            try:
                result = self.result_queue.get_nowait()
                print(f"    [MAIN] Got result from result_queue: {type(result)}")
            except queue.Empty:
                # Yield control to event loop so fetch_all can run
                await asyncio.sleep(0.1)

                # Print progress every 5 seconds
                if time.time() - last_progress > 5.0:
                    with self._lock:
                        elapsed = time.time() - start_time
                        rate = self.summary_count / elapsed * 3600 if elapsed > 0 else 0
                        print(f"\n   📊 Fetch: {self.fetch_count}/{total} | "
                              f"Vision: {self.vision_count} | "
                              f"Summary: {self.summary_count} | "
                              f"Rate: {rate:.0f}/hr | "
                              f"FQ: {self.fetch_queue.qsize()} | "
                              f"VQ: {self.vision_queue.qsize()}")
                        print(f"    [MAIN] Vision thread alive: {vision_thread.is_alive()}, Summary thread alive: {summary_thread.is_alive()}")
                    last_progress = time.time()
                continue

            if result is STOP_SIGNAL:
                print(f"    [MAIN] Got STOP_SIGNAL, breaking...")
                break

            results.append(result)

            # Progress every 10
            with self._lock:
                done = self.summary_count + self.failed_count
            if done % 10 == 0:
                elapsed = time.time() - start_time
                rate = self.summary_count / elapsed * 3600 if elapsed > 0 else 0
                eta = (total - done) / (done / elapsed) if done > 0 else 0
                print(f"\n📊 Progress: {done}/{total} | "
                      f"Rate: {rate:.0f}/hr | "
                      f"ETA: {eta/60:.1f}min | "
                      f"Images: {self.images_processed}")

        # Wait for threads
        await fetch_task
        vision_thread.join(timeout=5.0)
        summary_thread.join(timeout=5.0)

        with self._lock:
            processed = self.summary_count
            failed = self.failed_count

        return processed, failed


async def main():
    parser = argparse.ArgumentParser(description="Maximum GPU utilization ticket summarization")
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL, help="MCP server URL")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama URL")
    parser.add_argument("--model", default=DEFAULT_SUMMARY_MODEL, help="Summarization model (gpt-oss:120b)")
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL, help="Vision model for images (ministral-3)")
    parser.add_argument("--no-images", action="store_true", help="Disable image processing")
    parser.add_argument("--arango-host", default="localhost", help="ArangoDB host")
    parser.add_argument("--arango-port", type=int, default=8529, help="ArangoDB port")
    parser.add_argument("--fetch-buffer", type=int, default=10, help="Fetch queue buffer size")
    parser.add_argument("--vision-buffer", type=int, default=10, help="Vision queue buffer size")
    parser.add_argument("--start-from", type=int, default=0, help="Start from ticket index")
    parser.add_argument("--max-tickets", type=int, default=0, help="Max tickets to process (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Don't save summaries, just test")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 MAXIMUM GPU UTILIZATION PIPELINE")
    print("=" * 60)
    print(f"MCP URL: {args.mcp_url}")
    print(f"Ollama URL: {args.ollama_url}")
    print(f"Summary Model: {args.model}")
    print(f"Vision Model: {args.vision_model}")
    print(f"Process Images: {not args.no_images}")
    print(f"Buffers: Fetch={args.fetch_buffer} | Vision={args.vision_buffer}")
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

    # Create threaded GPU pipeline
    pipeline = ThreadedGPUPipeline(
        summarizer=summarizer,
        ollama_url=args.ollama_url,
        vision_model=args.vision_model,
        fetch_buffer_size=args.fetch_buffer,
        vision_buffer_size=args.vision_buffer,
        process_images=not args.no_images,
    )

    start_time = time.time()

    try:
        processed, failed = await pipeline.process_tickets(
            mcp_client=mcp_client,
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

