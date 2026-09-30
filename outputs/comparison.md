# Baseline vs custom variants (k=3, dim=64)

| Metric | baseline | top2 |
|---|---:|---:|
| Silhouette (cosine), all | 0.2013* | 0.1543 |
| Davies-Bouldin, all | 2.6104* | 3.0300 |
| Calinski-Harabasz, all | 61.3428* | 55.9682 |
| ARI vs doc_id, all | 0.2157 | 0.3580* |
| NMI vs doc_id, all | 0.3150 | 0.4615* |
| Silhouette, test docs only | 0.2943* | 0.2518 |
| ARI, test docs (global clusters) | 0.5415 | 0.5655* |
| NMI, test docs (global clusters) | 0.4376 | 0.4821* |
| ARI, test docs (re-clustered) | 0.5664 | 0.6715* |
| NMI, test docs (re-clustered) | 0.4589 | 0.5698* |

Rows won: baseline=4, top2=6

Notes: * = best in row. 'all' rows include documents the custom models trained on (optimistic for them); 'test docs' rows use held-out documents only. Davies-Bouldin: lower is better.

Variant settings:

- top2: {'top_layers': 2, 'p_far': 0.25, 'window': 3, 'mask_same_doc': True, 'epochs': 3, 'lr_encoder': 2e-05}