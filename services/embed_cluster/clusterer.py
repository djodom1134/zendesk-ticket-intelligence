"""
Ticket Clustering Service
Groups tickets into clusters using UMAP dimensionality reduction and HDBSCAN
"""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class ClusterResult:
    """Result of clustering operation"""
    cluster_id: int
    label: str
    keywords: list[str]
    ticket_ids: list[str]
    representative_ids: list[str]
    size: int
    centroid: Optional[list[float]] = None


class TicketClusterer:
    """Clusters ticket embeddings using UMAP + HDBSCAN."""

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int = 3,
        umap_n_components: int = 10,
        umap_n_neighbors: int = 15,
        metric: str = "cosine",
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.metric = metric
        self._umap = None
        self._hdbscan = None

    def _init_models(self):
        """Lazy initialization of UMAP and HDBSCAN"""
        if self._umap is None:
            import umap
            self._umap = umap.UMAP(
                n_components=self.umap_n_components,
                n_neighbors=self.umap_n_neighbors,
                metric=self.metric,
                random_state=42,
            )

        if self._hdbscan is None:
            import hdbscan
            self._hdbscan = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
            )

    def cluster(
        self,
        embeddings: np.ndarray,
        ticket_ids: list[str],
        ticket_texts: list[str] = None,
    ) -> list[ClusterResult]:
        """Cluster embeddings and return cluster results."""
        n_tickets = len(embeddings)
        logger.info("Starting clustering", n_tickets=n_tickets)

        if n_tickets < self.min_cluster_size:
            logger.warning("Not enough tickets for clustering", n=n_tickets)
            return []

        # For very small datasets, skip UMAP and cluster directly on embeddings
        # UMAP requires n_samples > n_components and n_neighbors <= n_samples
        MIN_FOR_UMAP = 15  # Minimum samples for reliable UMAP
        
        if n_tickets < MIN_FOR_UMAP:
            logger.info(
                "Dataset too small for UMAP, using direct clustering",
                n_tickets=n_tickets,
                min_for_umap=MIN_FOR_UMAP,
            )
            # Use PCA for simple dimensionality reduction on small datasets
            from sklearn.decomposition import PCA
            n_components = min(5, n_tickets - 1)
            pca = PCA(n_components=n_components, random_state=42)
            reduced = pca.fit_transform(embeddings)
            logger.info("PCA reduction complete", dims=n_components)
        else:
            # Adjust UMAP parameters for medium datasets
            effective_n_neighbors = min(self.umap_n_neighbors, n_tickets - 1)
            effective_n_components = min(self.umap_n_components, n_tickets - 2)
            
            if effective_n_neighbors != self.umap_n_neighbors:
                logger.info(
                    "Adjusting UMAP for dataset size",
                    original_neighbors=self.umap_n_neighbors,
                    effective_neighbors=effective_n_neighbors,
                    effective_components=effective_n_components,
                )
                import umap
                self._umap = umap.UMAP(
                    n_components=effective_n_components,
                    n_neighbors=effective_n_neighbors,
                    metric=self.metric,
                    random_state=42,
                )
            else:
                self._init_models()

            # UMAP reduction
            logger.info("Running UMAP reduction", dims=effective_n_components)
            reduced = self._umap.fit_transform(embeddings)
        
        # Initialize HDBSCAN if not done
        if self._hdbscan is None:
            import hdbscan
            # Adjust min_cluster_size for small datasets
            effective_min_cluster = min(self.min_cluster_size, max(2, n_tickets // 3))
            effective_min_samples = min(self.min_samples, effective_min_cluster)
            self._hdbscan = hdbscan.HDBSCAN(
                min_cluster_size=effective_min_cluster,
                min_samples=effective_min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
            )

        # HDBSCAN clustering
        logger.info("Running HDBSCAN clustering")
        labels = self._hdbscan.fit_predict(reduced)

        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise
        logger.info("Found clusters", n_clusters=len(unique_labels), noise=sum(labels == -1))

        clusters = []
        for cluster_id in sorted(unique_labels):
            mask = labels == cluster_id
            cluster_ticket_ids = [tid for tid, m in zip(ticket_ids, mask) if m]
            cluster_embeddings = embeddings[mask]
            centroid = cluster_embeddings.mean(axis=0)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            rep_indices = np.argsort(distances)[:3]
            representative_ids = [cluster_ticket_ids[i] for i in rep_indices]
            keywords = []
            if ticket_texts:
                cluster_texts = [ticket_texts[i] for i, m in enumerate(mask) if m]
                keywords = self._extract_keywords(cluster_texts)
            clusters.append(ClusterResult(
                cluster_id=int(cluster_id),
                label=f"Cluster-{cluster_id}",
                keywords=keywords,
                ticket_ids=cluster_ticket_ids,
                representative_ids=representative_ids,
                size=len(cluster_ticket_ids),
                centroid=centroid.tolist(),
            ))
        return clusters

    def _extract_keywords(self, texts: list[str], top_k: int = 10) -> list[str]:
        """Extract top keywords from cluster texts"""
        if not texts:
            return []
        all_words = []
        for text in texts:
            words = text.lower().split()
            words = [w for w in words if len(w) > 3 and w.isalpha()]
            all_words.extend(words)
        counter = Counter(all_words)
        stop_words = {"this", "that", "with", "from", "have", "been", "were",
                      "will", "would", "could", "should", "which", "their",
                      "there", "about", "when", "what", "your", "please"}
        for word in stop_words:
            counter.pop(word, None)
        return [word for word, _ in counter.most_common(top_k)]