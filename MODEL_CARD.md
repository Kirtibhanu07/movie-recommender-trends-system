# Model card

## Intended use
Educational, research and portfolio demonstration of personalized movie ranking. The catalogue and ratings are historical. Recommendations are not predictions of current trends or availability.

## Training and selection
Rating >= 4 is positive implicit feedback. Per-user distinct timestamp groups are split approximately 70/15/15 into training, validation and test, keeping tied sessions together. Users with insufficient distinct sessions remain training-only. Evaluation requires at least five training likes and one positive in both validation and test.

BPR matrix factorization and two-layer LightGCN use 32-dimensional embeddings, a fixed seed of 42, 30 epochs and the parameters recorded in experiment_plan.json. Checkpoints and genre blending weights are selected using validation NDCG@20 on 1,024 users. Test results never select the model. There is no final refit. A genre-only baseline and popularity baseline are included.

## Evaluation scope
The selected model is BPR_MF_hybrid_0.3. On 5,720 test users, NDCG@20 = 0.097853, Recall@20 = 0.111445 and catalogue coverage@20 = 35.77%. Popularity NDCG@20 = 0.082406. The paired user bootstrap interval for the absolute NDCG improvement is [0.01234, 0.01856]. This interval describes users within a single seeded run, not variation across training seeds or deployments.

All 3,883 catalogue items are candidates, excluding every training and validation rated item, including low ratings. Negatives during training exclude training positives only. Unobserved feedback is not proof of dislike.

Per-user chronology does not enforce a global calendar cutoff. Other users' training interactions may be later than a target user's held-out interaction. Catalogue and genre metadata are static. This is not a prospective production simulation. Single dataset, single seed and limited hyperparameter search constrain generalization.

Favorite-based cold start, filtering and diversity reranking are interactive heuristics. Their quality is not measured by the reported warm-user benchmark. Reasons describe genre overlap, not causal explanations. Ranking scores are not calibrated probabilities. Low-support items and users can receive weaker recommendations.

## Operations
Serving uses precomputed vectors and shared UI/API logic. Reported latency measures only local warm-user ranking, excluding model loading, network and UI. No authentication, monitoring, online learning, A/B test, fairness audit or commercial deployment has been performed. Raw ratings and serving artifacts stay local under MovieLens terms. No demographic fields are used.
