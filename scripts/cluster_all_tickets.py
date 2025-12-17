#!/usr/bin/env python3
"""
Cluster all ticket embeddings and generate cluster labels.

This script:
1. Fetches all embeddings from Qdrant
2. Runs UMAP + HDBSCAN clustering
3. Generates cluster labels using LLM
4. Stores clusters in ArangoDB

Usage:
    python scripts/cluster_all_tickets.py --ollama-url http://localhost:11434 --qdrant-host localhost
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.embed_cluster.clusterer import TicketClusterer
from services.embed_cluster.labeler import ClusterLabeler
from services.embed_cluster.vector_store import VectorStore

# Defaults
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
DEFAULT_QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
DEFAULT_LABEL_MODEL = os.getenv("LABEL_MODEL", "gpt-oss:120b")


def main():
    parser = argparse.ArgumentParser(description="Cluster all ticket embeddings")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama URL")
    parser.add_argument("--qdrant-host", default=DEFAULT_QDRANT_HOST, help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=DEFAULT_QDRANT_PORT, help="Qdrant port")
    parser.add_argument("--label-model", default=DEFAULT_LABEL_MODEL, help="LLM model for labeling")
    parser.add_argument("--arango-host", default="localhost", help="ArangoDB host")
    parser.add_argument("--arango-port", type=int, default=8529, help="ArangoDB port")
    parser.add_argument("--min-cluster-size", type=int, default=5, help="Min cluster size")
    parser.add_argument("--min-samples", type=int, default=3, help="Min samples for HDBSCAN")
    args = parser.parse_args()

    print("=" * 60)
    print("🔬 TICKET CLUSTERING PIPELINE")
    print("=" * 60)
    print(f"Ollama URL: {args.ollama_url}")
    print(f"Label Model: {args.label_model}")
    print(f"Qdrant: {args.qdrant_host}:{args.qdrant_port}")
    print(f"ArangoDB: {args.arango_host}:{args.arango_port}")
    print(f"Min cluster size: {args.min_cluster_size}")
    print()

    # Connect to ArangoDB
    from arango import ArangoClient
    arango = ArangoClient(hosts=f"http://{args.arango_host}:{args.arango_port}")
    db = arango.db("zti", username="root", password="zti_dev_password")
    tickets_col = db.collection("tickets")

    # Create clusters collection if it doesn't exist
    if not db.has_collection("clusters"):
        db.create_collection("clusters")
    clusters_col = db.collection("clusters")

    # Initialize vector store
    print("🗄️  Connecting to Qdrant...")
    vector_store = VectorStore(
        host=args.qdrant_host,
        port=args.qdrant_port,
    )

    if not vector_store.check_health():
        print("❌ Qdrant is not healthy")
        return 1

    print("   ✅ Qdrant connected")
    print()

    # Fetch all embeddings
    print("📥 Fetching all embeddings from Qdrant...")
    start_time = time.time()

    # Get all points from Qdrant
    scroll_result = vector_store._client.scroll(
        collection_name=vector_store.collection,
        limit=10000,  # Fetch all
        with_vectors=True,  # IMPORTANT: fetch the vectors!
    )

    points = scroll_result[0]
    print(f"   Found {len(points)} embeddings")

    # Extract embeddings and ticket IDs
    embeddings = []
    ticket_ids = []
    for p in points:
        if p.vector is not None and len(p.vector) > 0:
            embeddings.append(p.vector)
            ticket_ids.append(str(p.id))

    embeddings = np.array(embeddings)
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Sample embedding: {embeddings[0][:5] if len(embeddings) > 0 else 'none'}...")

    # Fetch ticket summaries for labeling
    print("📋 Fetching ticket summaries...")
    ticket_map = {}
    for tid in ticket_ids:
        ticket = tickets_col.get(tid)
        if ticket:
            ticket_map[tid] = ticket.get("summary", ticket.get("subject", ""))

    ticket_texts = [ticket_map.get(tid, "") for tid in ticket_ids]

    fetch_time = time.time() - start_time
    print(f"   ✅ Fetched in {fetch_time:.1f}s")
    print()

    # Run clustering
    print("🔬 Running UMAP + HDBSCAN clustering...")
    clusterer = TicketClusterer(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    cluster_start = time.time()
    clusters = clusterer.cluster(embeddings, ticket_ids, ticket_texts)
    cluster_time = time.time() - cluster_start

    print(f"   ✅ Found {len(clusters)} clusters in {cluster_time:.1f}s")
    print()

    if not clusters:
        print("⚠️  No clusters found")
        return 0

    # Generate labels
    print("🏷️  Generating cluster labels with LLM...")
    labeler = ClusterLabeler(
        ollama_url=args.ollama_url,
        model=args.label_model,
    )

    label_start = time.time()
    labeled_clusters = []

    for i, cluster in enumerate(clusters):
        print(f"   [{i+1}/{len(clusters)}] Labeling cluster {cluster.cluster_id} ({cluster.size} tickets)...")

        # Get representative ticket texts
        rep_texts = [ticket_map.get(tid, "") for tid in cluster.representative_ids[:5]]

        # Generate label
        summary = labeler.summarize_cluster(
            cluster_id=cluster.cluster_id,
            representative_texts=rep_texts,
            keywords=cluster.keywords,
        )

        labeled_clusters.append({
            "cluster": cluster,
            "summary": summary,
        })

    label_time = time.time() - label_start
    print(f"   ✅ Labeled {len(labeled_clusters)} clusters in {label_time:.1f}s")
    print()

    # Store in ArangoDB
    print("💾 Storing clusters in ArangoDB...")
    store_start = time.time()

    for item in labeled_clusters:
        cluster = item["cluster"]
        summary = item["summary"]

        cluster_doc = {
            "_key": str(cluster.cluster_id),
            "label": summary.label,
            "size": cluster.size,
            "keywords": cluster.keywords,
            "issue_description": summary.issue_description,
            "environment": summary.environment,
            "common_symptoms": summary.common_symptoms,
            "recommended_response": summary.recommended_response,
            "deflection_path": summary.deflection_path,
            "representative_tickets": cluster.representative_ids,
            "confidence": summary.confidence,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        # Upsert cluster
        clusters_col.insert(cluster_doc, overwrite=True)

        # Update tickets with cluster_id
        for tid in cluster.ticket_ids:
            try:
                tickets_col.update({"_key": tid, "cluster_id": str(cluster.cluster_id)})
            except:
                pass

    store_time = time.time() - store_start
    print(f"   ✅ Stored in {store_time:.1f}s")
    print()

    # Summary
    total_time = time.time() - start_time
    print("=" * 60)
    print("✅ Clustering complete!")
    print(f"   Clusters: {len(labeled_clusters)}")
    print(f"   Total tickets: {len(ticket_ids)}")
    print(f"   Clustered: {sum(c['cluster'].size for c in labeled_clusters)}")
    print(f"   Noise: {len(ticket_ids) - sum(c['cluster'].size for c in labeled_clusters)}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

