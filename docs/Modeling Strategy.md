# Modeling Strategy

---

## Objectives

- Balance **relevance** (head performance) and **coverage/novelty** (tail performance).
- Provide strong **cold-start** behavior for new and low-support movies.
- Control popularity amplification while preserving accuracy for active users.

---

## Models

- **Basic Collaborative Filtering (CF)**: Rely purely on collaborative signals. Used as a head-specialist model.
- **Hybrid matrix factorization (NeuMF)**: combine collaborative signals with side features for non-cold items and partial cold-start.
- **LightFM**: fast, interpretable for side-feature experiments (genres, directors, cast embeddings).
- **Content-only (SBERT/CLIP) models**: used for hard cold-start (overview/keywords/poster embeddings).
- **Ensemble**: Combines all model scores into a final score.

---

## Features & representations

- **Intrinsic metadata**: title, genres, runtime, production country, production company, original\_language, release\_year, adult, revenue, budget.
- **Text embeddings**: SBERT on overview+keywords+tags.
- **Visual embeddings**: CLIP embeddings for posters/backdrops for high-support items where available.
- **People features**: sparse embeddings for cast/directors/writers.
- **External signals**: IMDb/TMDB ratings and vote counts as features, but **confidence-weighted** by num\_votes.

Implementation notes: precompute embeddings, normalize numeric features, keep missingness flags as features.

---

## Cold-start strategy

- **Primary approach**: hybrid scoring = α·CF\_score + (1−α)·content\_score, with α a monotone function of item support.
- **IMDb/TMDB priors**: use Bayesian shrinkage to turn sparse vote averages into robust priors; weight by num\_votes.
- **Cold-only model**: train a content-only ranker optimized on leave-movie-out probe; use it as a fallback.
- **Feature defaults**: use explicit `has_*` flags so the model learns to fallback when assets are missing.

---

## Popularity & heavy-user mitigation (training-time)

- **Loss reweighting**: inverse item-popularity or per-item caps to reduce head dominance.
- **Per-user caps**: limit number of interactions per user per epoch or down-weight extreme users.
- **Negative sampling**: popularity-aware negatives (sample more from mid-tail) to avoid trivial positives-only learning.
- **Regularization**: stronger item/user regularization for low-support entities; use embedding shrinkage.

---

## Labeling & objective

- For ranking models convert explicit ratings to **binary relevance** (e.g., rating≥4) or graded relevance where appropriate.
- Use pairwise/pointwise ranking losses (BPR, hinge, or logistic) and validate on ranking metrics defined in Evaluation Protocol.
- Maintain RMSE/MAE as diagnostic for models that also predict scores.

---

## Training, validation & tuning

- Use the Evaluation Protocol's chronological folds for hyperparameter tuning; treat rolling validation as the default.
- Optimize ranking objective on mixed slices (include soft-cold items in validation for hybrid models).
- Log experiments (W&B) and save model + preprocessing artifacts (embeddings, scalers, split IDs).

---

## Explainability & debugging

- Maintain per-item and per-user diagnostic tables: support, avg\_predicted\_score, top-feature-contributors.
- Use partial-dependence / SHAP-style checks for important numeric features (popularity, runtime, vote\_count).
- Monitor shifts: distribution of top-K popularity, language balance, cold-start rate in recommendations.

---

## Production notes

- Precompute heavy embeddings (SBERT/CLIP) and store in a vector store; fetch at inference time.
- Keep a small, fast content-only fallback for hard cold-starts.
- Budget: prefer compact embeddings (128–512 dims) and quantize for memory-sensitive deployments.

---

## Short checklist before first model release

- Implement inverse-popularity weighting and per-user caps in training.
- Add confidence-weighting for external ratings.
- Ensure content embeddings and missingness flags are available at inference.
- Produce slice-level validation tables (support, language, era) as part of CI.

