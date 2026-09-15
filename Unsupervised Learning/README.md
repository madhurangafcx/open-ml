# Unsupervised Learning

In **Unsupervised Learning**, the model is provided with unlabeled data ($X$) without explicit ground-truth targets. The objective is to uncover latent patterns, intrinsic groupings, clusters, or lower-dimensional representations hidden within the data.

---

## Core Problem Types & Key Algorithms

### 1. Clustering
Algorithms that group similar data points together based on distance metrics or density.

* **K-Means Clustering**: Partitions data into a predefined number of distinct, non-overlapping clusters ($k$) by iteratively updating cluster centroids. Widely used for customer segmentation and pattern discovery.
* **Hierarchical Clustering**: Builds an agglomerative (bottom-up) or divisive (top-down) hierarchy of clusters visualized as a dendrogram, eliminating the need to specify $k$ in advance.
* **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Discovers clusters of arbitrary geometric shapes based on point density thresholds ($\epsilon$ and `minPts`), naturally detecting and isolating noise/outliers.
* **Gaussian Mixture Models (GMM)**: Soft, probabilistic clustering method assuming data is generated from a mixture of underlying Gaussian distributions (optimized via Expectation-Maximization).

### 2. Dimensionality Reduction
Algorithms that reduce the number of input features while preserving as much meaningful variance or structural geometry as possible.

* **Principal Component Analysis (PCA)**: Linear transformation that projects data along orthogonal axes of maximum variance, speeding up downstream algorithms and mitigating the curse of dimensionality.
* **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Non-linear probabilistic technique tailored for visual exploration of high-dimensional datasets in 2D or 3D manifolds by preserving local neighborhoods.
* **UMAP (Uniform Manifold Approximation and Projection)**: Modern non-linear reduction method based on Riemannian geometry that is faster than t-SNE and better preserves global topology alongside local structure.
* **Autoencoders**: Neural networks trained to compress input data into a bottleneck latent code (encoder) and reconstruct the original input (decoder).

### 3. Association Rule Learning
Rule-based methods that discover interesting correlations, frequent patterns, and co-occurrences between variables in large databases.

* **Apriori**: Identifies frequent item sets through candidate generation and assesses association rules using support, confidence, and lift metrics (frequently used in Market Basket Analysis: *"If a customer buys bread, they are likely to buy butter"*).
* **FP-Growth (Frequent Pattern Growth)**: Compresses datasets into an FP-tree structure to mine frequent itemsets without costly candidate generation.

---

## Roadmap & Planned Notebooks

| Topic | Planned Directory | Description | Status |
| :--- | :--- | :--- | :--- |
| **K-Means Clustering** | `K_Means/` | Centroid-based clustering & elbow method | :black_square_button: Planned |
| **DBSCAN Clustering** | `DBSCAN/` | Density clustering & outlier detection | :black_square_button: Planned |
| **PCA & Dimensionality Reduction** | `PCA/` | Variance projection, scree plots & visualization | :black_square_button: Planned |
| **t-SNE & UMAP** | `Manifold_Learning/` | High-dimensional data visualization | :black_square_button: Planned |
| **Association Rules (Apriori)** | `Association_Rules/` | Market basket analysis & transaction mining | :black_square_button: Planned |
| **Autoencoders** | `Autoencoders/` | Latent space representations & anomaly detection | :black_square_button: Planned |
