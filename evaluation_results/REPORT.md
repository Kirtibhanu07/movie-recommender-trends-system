# Personalized movie recommendation benchmark

The validation-selected recommendation is **BPR_MF_hybrid_0.3**. On 5720 eligible
test users it achieved **NDCG@20 0.0979** and **Recall@20 11.14%**.
Its NDCG@20 is 18.7% higher than the popularity baseline on this split.
This is a retrospective benchmark, not online engagement or commercial impact.

## Dataset and split

Official MovieLens 1M has 1,000,209 ratings, 6,040 users and
3,883 catalogue items. A rating of 4 or 5 is treated as a positive.
Only movie metadata and ratings are used. Demographic data is not loaded.

Distinct timestamp groups are ordered within each user. The earliest 70% of groups
are training, the next 15% validation, and the remaining groups test. Tied timestamp
groups never cross boundaries. Users with fewer than three groups remain training-only.
Eligible test users need five training positives and at least one positive in both
validation and test. This selects a particular population and is not representative
of every user. A seeded subset of 1024 eligible users is used
for checkpoint and blending selection. Final testing uses all eligible users.

This is **per-user chronological evaluation**, not a globally chronological simulation.
Another user's training event may occur later than a query user's held-out event.
The catalogue is assumed static. Ratings do not establish movie watch order, exposure,
or satisfaction. Unobserved movies are not known dislikes.

## Models and selection

- Popularity from training positive counts.
- Content profiles from training likes and IDF-weighted movie genres.
- Bayesian Personalized Ranking matrix factorization, 32-dimensional embeddings.
- Two-layer LightGCN with normalized user-item graph propagation and BPR loss.
- A validation-selected genre blend for the better learned checkpoint.

Both learned models use the same fixed training budget and seed. The model graph,
negative rejection and content profiles use training ratings only. Epochs and genre
weights are selected using validation NDCG@20. All choices are saved before scoring
the final test. No refit or test-based retuning is performed. Equal training budgets
do not establish a globally optimized comparison or state-of-the-art result.

Test ranking considers the full 3,883-item catalogue, removing
items already rated in training or validation. There is no sampled-negative ranking.
Metrics are macro averages over users. Recall@20 measures the fraction of the user's
held-out positive movies retrieved, not classification accuracy.

## Final test

| Model | NDCG@20 | Recall@20 | Precision@20 | Catalogue coverage |
|---|---:|---:|---:|---:|
| BPR_MF_hybrid_0.3 | 0.0979 | 11.14% | 6.20% | 35.77% |
| BPR_MF | 0.0976 | 10.87% | 6.22% | 34.90% |
| LightGCN | 0.0961 | 10.60% | 6.28% | 17.56% |
| popularity | 0.0824 | 9.13% | 5.51% | 5.61% |
| genre_content | 0.0121 | 1.73% | 0.70% | 43.34% |

## Uncertainty and serving measurements

The selected-minus-popularity NDCG@20 difference is
0.0154, with a 95% paired user-bootstrap interval
of [0.0123, 0.0186]. This measures user sampling uncertainty on one split
and one training seed. It does not include training variability or online effects.

Warm full-catalogue ranking median is 0.19 ms and p95 is
0.22 ms across 200 recorded queries. This includes
score calculation, filtering and sorting but not app rendering, network or model load.
It is a local single-process benchmark, not a production throughput test.

## Reproducibility and distribution

See the frozen experiment plan, training history, validation leaderboard,
`selection.json`, test leaderboard, per-user derived metrics and environment/data
hashes. Run `download_data.py`, then `train_evaluate.py` to reproduce. The raw dataset
and data-derived serving artifacts are not included in shared source archives.

MovieLens 1M is research-only and disallows raw data redistribution without separate
permission. This demonstration is not licensed for commercial deployment. Cite:
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and
Context. ACM TiiS 5(4), Article 19. https://doi.org/10.1145/2827872

LightGCN follows Xiangnan He et al., SIGIR 2020: https://arxiv.org/abs/2002.02126

## Limits

The data is from around 2000 and does not measure recommendations for today's
catalogue or current audiences. There is one seed, a limited hyperparameter budget,
and no online A/B test. Cold-start favorite-based recommendations and diversity
reranking are separate app features, not covered by the warm-user metrics above.
The trained model is new to this project, not a novel research architecture.
