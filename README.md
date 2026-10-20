# FrameLight · Personalized Movie Discovery

A reproducible movie recommendation project with a Streamlit interface, FastAPI service, BPR matrix factorization, LightGCN, genre hybridization, and an audited offline evaluation.

## Verified result

On MovieLens 1M, the validation-selected BPR + genre hybrid achieved **0.09785 NDCG@20**, versus **0.08241** for popularity, an **18.7% relative improvement** across 5,720 eligible test users. Recall@20 was 0.11145. LightGCN was trained and evaluated but did not outperform the selected hybrid. These are ranking metrics, not prediction accuracy.

Read [the complete evaluation](evaluation_results/REPORT.md) and [model card](MODEL_CARD.md) for the protocol and limitations.

## Run locally

Use Python 3.12 in a virtual environment.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-personalized.txt
python download_data.py
python train_evaluate.py
python -m pytest -q
streamlit run app.py --server.port 8503
```

Training produces local serving artifacts. The download and training steps are required in a fresh checkout. The current workspace already contains trained artifacts. Exact installed versions are recorded in requirements.lock.txt.

Run the API separately:

```sh
uvicorn api:app --host 127.0.0.1 --port 8001
```

Interactive API documentation is at `/docs`. Endpoints: `GET /health`, `POST /recommend/favorites` with `{"movie_ids":[1,2571],"k":10,"diversity":0.2}`, and `GET /recommend/user/1?k=10`. User IDs refer only to anonymous dataset profiles. The API is a local demonstration without authentication or production rate limits.

## Features

- Favorite-based discovery without an account, plus anonymous warm-user demonstrations
- Genre and release-year filters, optional genre diversity reranking, readable recommendation reasons
- Shared service for UI and API, input validation, safe NumPy artifacts loaded without pickle
- Full-catalogue ranking, exclusion of previously rated items, timestamp-tie-safe splits
- Fixed experiment plan, validation selection, paired bootstrap confidence interval, saved metrics and chart
- Tests covering chronology, graph normalization, negative sampling, optimization, metrics, filtering, API, and UI

## Data and reproducibility

MovieLens 1M contains 1,000,209 ratings from 6,040 users and a 3,883-title catalogue. It is an established historical dataset, newly integrated into this project. Movies generally predate 2001. No live trending or streaming availability is claimed.

The official dataset has research-use and redistribution restrictions. Read `data/README` after downloading. Raw data and derived serving artifacts are deliberately excluded from shared archives and Git. No demographic file is downloaded. Model outputs require the same applicable dataset terms. Do not treat this demonstration as commercially licensed.

Dataset: https://grouplens.org/datasets/movielens/1m/

Citation: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM TIIS 5(4), Article 19. https://doi.org/10.1145/2827872

LightGCN: Xiangnan He et al. 2020. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. https://arxiv.org/abs/2002.02126

The architectures are established research methods, not a claim of a novel model or state-of-the-art performance. No external movie API key is required.
