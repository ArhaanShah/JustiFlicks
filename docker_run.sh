#!/usr/bin/env bash
docker build -t movie-recommender:dev .
docker run --rm -p 8501:8501 \
  -v $(pwd)/data:/home/app/app/data \
  -v $(pwd)/experiments/runs:/home/app/app/experiments/runs \
  movie-recommender:dev \
  bash -lc "streamlit run demo/streamlit_app.py --server.port 8501 --server.headless true"
