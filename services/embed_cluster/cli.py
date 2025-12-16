#!/usr/bin/env python3
"""
ZTI Embed CLI
Command-line tool for generating and storing ticket embeddings
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import structlog

from services.embed_cluster.embedder import TicketEmbedder
from services.embed_cluster.vector_store import VectorStore

logger = structlog.get_logger()


def embed_from_file(
    input_file: str,
    ollama_url: str,
    qdrant_host: str,
    qdrant_port: int,
    use_local: bool = False,
) -> int:
    """Embed normalized tickets from JSON file and store in Qdrant"""
    print(f"Loading tickets from {input_file}...")
    
    with open(input_file, "r") as f:
        tickets = json.load(f)
    
    if not tickets:
        print("No tickets to process")
        return 0
    
    print(f"Loaded {len(tickets)} tickets")
    
    # Initialize embedder
    print(f"Initializing embedder (ollama={not use_local})...")
    embedder = TicketEmbedder(
        ollama_url=ollama_url,
        use_local=use_local,
    )
    
    # Test embedding to get dimension
    print("Testing embedding model...")
    test_dim = embedder.dimension
    print(f"   Embedding dimension: {test_dim}")
    
    # Initialize vector store
    print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
    vector_store = VectorStore(
        host=qdrant_host,
        port=qdrant_port,
        dimension=test_dim,
    )
    
    # Process tickets
    print("Generating embeddings...")
    ticket_ids = []
    vectors = []
    payloads = []
    
    for i, ticket in enumerate(tickets):
        ticket_id = ticket.get("ticket_id", str(i))
        fulltext = ticket.get("pii_redacted_text") or ticket.get("ticket_fulltext", "")
        
        if not fulltext:
            continue
        
        # Generate embedding
        vector = embedder.embed(fulltext)
        
        ticket_ids.append(ticket_id)
        vectors.append(vector)
        payloads.append({
            "ticket_id": ticket_id,
            "subject": ticket.get("subject", "")[:200],
            "status": ticket.get("status", ""),
            "priority": ticket.get("priority", ""),
            "created_at": ticket.get("created_at", ""),
            "tags": ticket.get("tags", [])[:10],
        })
        
        if (i + 1) % 10 == 0:
            print(f"   Embedded {i + 1}/{len(tickets)} tickets...")
    
    # Store in Qdrant
    print(f"Storing {len(vectors)} embeddings in Qdrant...")
    vector_store.upsert_batch(ticket_ids, vectors, payloads)
    
    print(f"✅ Embedded and stored {len(vectors)} tickets")
    return len(vectors)


def search_similar(
    query: str,
    ollama_url: str,
    qdrant_host: str,
    qdrant_port: int,
    limit: int = 5,
) -> list[dict]:
    """Search for similar tickets"""
    print(f"Searching for: {query[:50]}...")
    
    embedder = TicketEmbedder(ollama_url=ollama_url)
    vector_store = VectorStore(host=qdrant_host, port=qdrant_port)
    
    query_vector = embedder.embed(query)
    results = vector_store.search(query_vector, limit=limit)
    
    print(f"\nFound {len(results)} similar tickets:\n")
    for r in results:
        print(f"  [{r['score']:.3f}] Ticket {r['id']}: {r['payload'].get('subject', 'N/A')}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="ZTI Embed CLI")
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost"))
    parser.add_argument("--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Embed tickets from file")
    embed_parser.add_argument("input", help="Normalized tickets JSON file")
    embed_parser.add_argument("--local", action="store_true", help="Use local model")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search similar tickets")
    search_parser.add_argument("query", help="Search query text")
    search_parser.add_argument("--limit", type=int, default=5, help="Max results")
    
    args = parser.parse_args()
    
    if args.command == "embed":
        embed_from_file(
            args.input,
            args.ollama_url,
            args.qdrant_host,
            args.qdrant_port,
            args.local,
        )
    elif args.command == "search":
        search_similar(
            args.query,
            args.ollama_url,
            args.qdrant_host,
            args.qdrant_port,
            args.limit,
        )


if __name__ == "__main__":
    main()

