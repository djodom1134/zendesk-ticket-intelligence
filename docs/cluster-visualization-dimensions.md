# Cluster Visualization Dimensions Explained

## Overview

The cluster graph visualization uses **UMAP (Uniform Manifold Approximation and Projection)** to reduce high-dimensional ticket embeddings to 2D coordinates for visualization.

## How x,y Coordinates are Calculated

### Step 1: Get Cluster Centroids (4096 dimensions)

Each ticket is embedded using the `qwen3-embedding:8b` model, producing a **4096-dimensional vector**.

For each cluster:
1. Fetch representative ticket embeddings from Qdrant
2. Compute the **centroid** (mean) of these vectors
3. Result: One 4096-dim vector per cluster

### Step 2: UMAP Dimensionality Reduction

UMAP reduces the 4096-dimensional centroids to **10 dimensions**:

```python
umap_model = umap.UMAP(
    n_components=10,      # Reduce to 10 dimensions
    n_neighbors=15,       # Local neighborhood size
    min_dist=0.1,         # Minimum distance between points
    metric='cosine',      # Similarity metric
    random_state=42       # Reproducibility
)
positions_10d = umap_model.fit_transform(centroids_4096d)
```

**Output**: 10 UMAP components per cluster

### Step 3: Select Dimensions for x and y

Users can select which 2 of the 10 UMAP components to display:

- **x_dim**: Component for x-axis (0-9, default 0)
- **y_dim**: Component for y-axis (0-9, default 1)

```python
positions_2d = positions_10d[:, [x_dim, y_dim]]
```

### Step 4: Normalize to 0-100 Range

For consistent visualization scaling:

```python
# Normalize to 0-1
positions_2d = (positions_2d - min) / (max - min)
# Scale to 0-100
positions_2d = positions_2d * 100
```

**The 0-100 range is arbitrary** - it's just for visualization convenience.

---

## What Do the Axes Represent?

### Important: x and y are NOT Original Dimensions!

**x and y are NOT specific dimensions from the 4096-dim embedding.**

Instead:
- **x-axis (Component 0)**: A learned combination of ALL 4096 dimensions
- **y-axis (Component 1)**: Another learned combination of ALL 4096 dimensions

### What Each UMAP Component Captures

- **Component 0**: Primary semantic dimension (captures most variance)
- **Component 1**: Secondary semantic dimension (orthogonal to Component 0)
- **Component 2**: Tertiary semantic dimension
- **Components 3-9**: Additional semantic dimensions

Each component:
- Preserves **local structure** (similar clusters stay close)
- Preserves **global structure** (distinct clusters stay far apart)
- Captures different aspects of cluster relationships

### Example Interpretations

While UMAP components are not directly interpretable, they often capture:

- **Component 0**: Might separate "technical issues" from "account issues"
- **Component 1**: Might separate "urgent" from "routine" issues
- **Component 2**: Might separate "hardware" from "software" issues

**Note**: These are examples - actual meanings depend on the data.

---

## Dimension Selection Examples

### Default View (x=0, y=1)
```
GET /api/clusters?include_positions=true&x_dim=0&y_dim=1
```
- Most common view
- Shows primary and secondary semantic dimensions
- Best overall cluster separation

### Alternative Perspective (x=0, y=2)
```
GET /api/clusters?include_positions=true&x_dim=0&y_dim=2
```
- Shows primary vs tertiary dimension
- May reveal different cluster relationships

### Explore Tertiary Relationships (x=2, y=3)
```
GET /api/clusters?include_positions=true&x_dim=2&y_dim=3
```
- Shows less dominant semantic dimensions
- May reveal subtle cluster patterns

---

## Why 0-100 Range?

The 0-100 range is **purely for visualization convenience**:

1. **Consistent Scaling**: All graphs use the same coordinate system
2. **Easy to Work With**: No negative numbers, reasonable magnitude
3. **Graph Library Compatibility**: Works well with D3, Three.js, etc.

The actual UMAP output could be any range (e.g., -5.2 to 8.7). We normalize it to 0-100 for consistency.

---

## Can We Use Other Dimensions?

Yes! Here are alternative approaches:

### Option 1: Metadata Dimensions (Future Enhancement)
Plot clusters by actual metadata:
- **x-axis**: Cluster size (ticket count)
- **y-axis**: Average confidence score
- **x-axis**: Creation date
- **y-axis**: Priority (high=3, medium=2, low=1)

### Option 2: PCA Instead of UMAP (Future Enhancement)
PCA components are ordered by variance explained:
- **PC1**: Explains most variance
- **PC2**: Explains second-most variance
- More interpretable than UMAP

### Option 3: t-SNE (Future Enhancement)
Alternative dimensionality reduction:
- Better for local structure
- Slower than UMAP
- Non-deterministic

---

## Technical Details

### UMAP Parameters

- **n_components=10**: Number of dimensions to reduce to
- **n_neighbors=15**: Size of local neighborhood (affects local vs global structure)
- **min_dist=0.1**: Minimum distance between points (affects clustering tightness)
- **metric='cosine'**: Similarity metric (cosine similarity for text embeddings)
- **random_state=42**: Seed for reproducibility

### Performance

- **Computation Time**: ~2-5 seconds for 63 clusters
- **Caching**: Not currently cached (computed on-demand)
- **Future Optimization**: Cache UMAP model and 10D positions, only recompute 2D projection

---

## API Usage

### Get Clusters with Default Dimensions
```bash
curl "http://localhost:8000/api/clusters?include_positions=true"
```

### Get Clusters with Custom Dimensions
```bash
curl "http://localhost:8000/api/clusters?include_positions=true&x_dim=2&y_dim=3"
```

### Response Format
```json
{
  "id": "35",
  "label": "Storage detection failure",
  "size": 106,
  "x": 35.33,
  "y": 31.70
}
```

---

## Summary

- **x,y are UMAP components**, not original embedding dimensions
- **0-100 range is arbitrary**, just for visualization
- **10 components available**, users can select which 2 to display
- **Each component captures different semantic relationships**
- **Default (0,1) usually provides best overall view**

