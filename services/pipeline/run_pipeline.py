#!/usr/bin/env python3
"""
ZTI Pipeline Runner - Runs full pipeline: normalize → summarize → embed → cluster → label
All steps run inside the pipeline container on the GPU machine.
"""

import json
import os
import sys
import click
import structlog

log = structlog.get_logger()

# Add app root to path for imports
sys.path.insert(0, "/app")

from services.normalize.normalizer import TicketNormalizer
from services.normalize.redactor import PIIRedactor
from services.embed_cluster.embedder import TicketEmbedder
from services.embed_cluster.clusterer import TicketClusterer
from services.embed_cluster.labeler import ClusterLabeler
from services.embed_cluster.summarizer import TicketSummarizer


@click.command()
@click.option("--input", "-i", "input_file", required=True, help="Input JSON file with raw tickets")
@click.option("--output-dir", "-o", default="/data/zti", help="Output directory for results")
@click.option("--skip-normalize", is_flag=True, help="Skip normalization step")
@click.option("--skip-summarize", is_flag=True, help="Skip summarization step (embed full text)")
@click.option("--skip-embed", is_flag=True, help="Skip embedding step")
@click.option("--skip-cluster", is_flag=True, help="Skip clustering step")
@click.option("--skip-label", is_flag=True, help="Skip labeling step")
@click.option("--enable-redaction", is_flag=True, help="Enable PII redaction (disabled by default)")
@click.option("--ollama-url", default=None, help="Ollama URL (default: from OLLAMA_URL env or http://ollama:11434)")
@click.option("--summarize-model", default="gpt-oss:120b", help="LLM model for summarization")
def main(input_file: str, output_dir: str, skip_normalize: bool, skip_summarize: bool,
         skip_embed: bool, skip_cluster: bool, skip_label: bool, enable_redaction: bool,
         ollama_url: str, summarize_model: str):
    """Run the ZTI pipeline on ticket data."""

    os.makedirs(output_dir, exist_ok=True)

    # Resolve Ollama URL
    ollama_url_resolved = ollama_url or os.getenv("OLLAMA_URL", "http://ollama:11434")
    log.info("Pipeline config", ollama_url=ollama_url_resolved, redaction=enable_redaction)

    # Load input
    log.info("Loading input tickets", file=input_file)
    with open(input_file) as f:
        tickets = json.load(f)
    log.info("Loaded tickets", count=len(tickets))

    # Step 1: Normalize
    if not skip_normalize:
        log.info("Step 1: Normalizing tickets...", redaction=enable_redaction)
        normalizer = TicketNormalizer(enable_redaction=enable_redaction)

        normalized = []
        for ticket in tickets:
            try:
                norm = normalizer.normalize(ticket)
                # Convert Pydantic model to dict for JSON serialization
                normalized.append(norm.model_dump() if hasattr(norm, 'model_dump') else norm)
            except Exception as e:
                log.warning("Failed to normalize ticket", ticket_id=ticket.get("id"), error=str(e))

        norm_file = os.path.join(output_dir, "tickets_normalized.json")
        with open(norm_file, "w") as f:
            json.dump(normalized, f, indent=2, default=str)
        log.info("Normalization complete", output=norm_file, count=len(normalized))
        tickets = normalized
    else:
        log.info("Skipping normalization")

    # Step 2: Summarize (with streaming, metrics, and resume support)
    summaries = None
    summary_file = os.path.join(output_dir, "summaries.json")

    if not skip_summarize:
        log.info("Step 2: Summarizing tickets with streaming...", model=summarize_model)

        # Load existing summaries for resume capability
        existing_summaries = {}
        if os.path.exists(summary_file):
            try:
                with open(summary_file, "r") as f:
                    existing_data = json.load(f)
                    existing_summaries = {
                        str(item["ticket_id"]): item["summary"]
                        for item in existing_data if item.get("summary")
                    }
                log.info("Loaded existing summaries for resume", count=len(existing_summaries))
            except Exception as e:
                log.warning("Could not load existing summaries", error=str(e))

        summarizer = TicketSummarizer(ollama_url=ollama_url_resolved, model=summarize_model)
        summaries, batch_metrics = summarizer.summarize_batch(
            tickets,
            show_live=True,
            existing_summaries=existing_summaries,
        )

        # Save summaries (overwrite with complete set)
        with open(summary_file, "w") as f:
            summary_data = [{"ticket_id": t.get("ticket_id") or t.get("id"), "summary": s}
                           for t, s in zip(tickets, summaries)]
            json.dump(summary_data, f, indent=2)
        log.info("Summarization complete", output=summary_file, count=len(summaries))
    else:
        log.info("Skipping summarization - will embed full text")

    # Step 3: Embed
    embeddings = None
    if not skip_embed:
        log.info("Step 3: Generating embeddings...", ollama_url=ollama_url_resolved)
        embedder = TicketEmbedder(ollama_url=ollama_url_resolved)

        # Use summaries if available, otherwise use full text
        if summaries:
            texts = summaries
            log.info("Embedding summaries")
        else:
            texts = [t.get("ticket_fulltext", "") or f"{t.get('subject', '')} {t.get('description', '')}"
                     for t in tickets]
            log.info("Embedding full text")
        embeddings = embedder.embed_batch(texts)

        embed_file = os.path.join(output_dir, "embeddings.json")
        with open(embed_file, "w") as f:
            # Handle both numpy arrays and plain lists
            emb_list = [e.tolist() if hasattr(e, 'tolist') else e for e in embeddings]
            json.dump({"embeddings": emb_list, "ticket_ids": [t.get("ticket_id") or t.get("id") for t in tickets]}, f)
        log.info("Embedding complete", output=embed_file, vectors=len(embeddings))
    else:
        log.info("Skipping embedding")

    # Step 4: Cluster
    clusters = None
    if not skip_cluster and embeddings is not None:
        log.info("Step 4: Clustering tickets...")
        clusterer = TicketClusterer()

        cluster_results = clusterer.cluster(embeddings, tickets)
        
        # Convert ClusterResult objects to dicts for serialization
        clusters = [
            {
                "cluster_id": c.cluster_id,
                "label": c.label,
                "keywords": c.keywords,
                "ticket_ids": c.ticket_ids,
                "representative_ids": c.representative_ids,
                "size": c.size,
                "centroid": c.centroid,
            }
            for c in cluster_results
        ]

        cluster_file = os.path.join(output_dir, "clusters.json")
        with open(cluster_file, "w") as f:
            json.dump({"clusters": clusters, "total": len(clusters)}, f, indent=2, default=str)
        log.info("Clustering complete", output=cluster_file, num_clusters=len(clusters))
    else:
        log.info("Skipping clustering")

    # Step 5: Label
    if not skip_label and clusters is not None and len(clusters) > 0:
        log.info("Step 5: Labeling clusters...")
        labeler = ClusterLabeler()

        # Build ticket text mapping for labeler
        ticket_texts = {str(t.get("id", t.get("ticket_id", i))): t.get("summary", t.get("subject", "")) 
                        for i, t in enumerate(tickets)}
        
        # Convert cluster dicts back to objects for labeler
        from services.embed_cluster.clusterer import ClusterResult
        cluster_objs = [
            ClusterResult(
                cluster_id=c["cluster_id"],
                label=c["label"],
                keywords=c.get("keywords", []),
                ticket_ids=c["ticket_ids"],
                representative_ids=c["representative_ids"],
                size=c["size"],
                centroid=c.get("centroid"),
            )
            for c in clusters
        ]
        
        labeled = labeler.summarize_clusters_batch(cluster_objs, ticket_texts)

        label_file = os.path.join(output_dir, "clusters_labeled.json")
        with open(label_file, "w") as f:
            from services.embed_cluster.labeler import summary_to_dict
            json.dump([summary_to_dict(s) for s in labeled], f, indent=2)
        log.info("Labeling complete", output=label_file)
    elif clusters is not None and len(clusters) == 0:
        log.info("Skipping labeling - no clusters found (all tickets marked as noise)")
    else:
        log.info("Skipping labeling")

    log.info("Pipeline complete!", output_dir=output_dir)
    print(f"\n✅ Pipeline complete! Results in {output_dir}")


if __name__ == "__main__":
    main()

