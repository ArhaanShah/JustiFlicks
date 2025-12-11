# Sprint Plan

**Goal:** High abstraction timeline with clear objectives, key tasks, success criteria.

---

## Sprint 0: Prep

**Objectives:** repo environment, sample dataset snapshot, repo setup.

**Key tasks:**

- Repo layout, README, and architecture diagram; legality checks for data (licensing).
- Ensure reproducibility (Docker).
- Prepare Colab notebook; set up experiment tracking (W&B/CSV).
- Download small MovieLens data sample and save snapshot in case data changes.

**Success criteria:** initial repo+notebook created; sample dataset available.

---

## Sprint 1: EDA

**Objectives:** gather data, assess coverage and cold-start.

**Key tasks:**

- Fetch MovieLens and TMDB metadata, posters, user reviews (cache results).
- Run EDA on sparsity, rating distribution, poster/synopsis/cast data coverage, cold-start fraction.
- Slice based EDA
- Produce a short data inventory and issue log.

**Success criteria:** data inventory completed with % coverage metrics.

---

## Sprint 2: Set baseline

**Objectives:** implement Collaborative Filtering baseline and evaluation harness.

**Key tasks:**

- Implement matrix factorization or LightFM (classic baseline).
- Implement NeuMF (neural baseline) and compare metrics.
- Create train/val/test splits and compute Recall\@K, NDCG\@K, MAP.
- Build preprocessing pipelines & logging.
- Save baseline configs and results.

**Success criteria:** baseline metrics stable and outputs are saved.

---

## Sprint 3: Multimodal embeddings & retrieval

**Objectives:** compute text/image embeddings, build FAISS index, integrate into hybrid ranking.

**Key tasks:**

- Generate CLIP (ViT‑B/32) image embeddings and SBERT text embeddings.
- Build FAISS CPU index and tune ANN settings.
- Implement hybrid scoring (combine CF + content embeddings) and compare to baseline.
- Report statistical significance for metric improvements (confidence intervals, paired tests across users/slices)

**Success criteria:** hybrid ranking improves performance (Recall\@K/NDCG\@K) vs baseline.

---

## Sprint 4: Explanation + PEFT (LoRA)

**Objectives:** design explanation template, fine-tune small LLM with PEFT, implement fidelity checks.

**Key tasks:**

- Define a strict explanation template and prompts.
- Create synthetic (signals→explanation) training pairs and LoRA fine-tune a small model on Colab/Kaggle.
- Implement automated fidelity verifier that matches LLM claims to top signals.

**Success criteria:** LLM outputs grounded explanations; automated fidelity reaches target threshold.

---

## Sprint 5: Hyperparameter tuning

**Objectives:** systematically tune key hyperparameters across CF, retrieval, hybrid scoring, and the LLM adapter to maximize ranking and fidelity while respecting resource constraints.

**Key tasks:**

- Define search spaces for major knobs (n\_factors, candidate\_pool\_size, FAISS nprobe/efSearch, hybrid weights, LoRA rank, learning rate, neuMF params).
- Run cheap random search/Optuna trials on a small sampled dataset (10k–50k) using early stopping; promote top configs to full runs on complete data.
- Use progressive resizing / multi-fidelity (short epochs, fewer candidates) to save compute.
- Log all trials (CSV or W&B), save best checkpoints and exact configs.

**Success criteria:**

- Candidate recall for K reaches target (e.g., ≥0.95 at candidate pool size).
- Validation ranking metric (e.g., NDCG\@K) improves over default config.
- Best configs and experiment logs are saved.

---

## Sprint 6: Evaluation & Robustness

**Objectives:** verify model quality, grounding, and reliability.

**Key tasks:**

- Run automated fidelity and ranking evaluations across slices; collect 50+ human evaluations for clarity/usefulness; evaluation rigor checks.
- Perform basic robustness checks (hallucinations, missing/ noisy metadata).
- Succinct acceptance criteria list (documented failure cases if not successful).
- Minimal CI/unit tests/smoke tests.

**Success criteria:** concise evaluation report; core targets met or issues clearly logged.

---

## Sprint 7: Demo & Packaging

**Objectives:** polishing for final demo.

**Key tasks:**

- Build a lightweight Streamlit demo with precomputed embeddings + LoRA.
- Prepare README, model card, ethics note.
- Profile latency, memory, error rate.

**Success criteria:** working demo and polished documentation.

---