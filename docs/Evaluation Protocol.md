# Evaluation Protocol

Derived directly from EDA findings (extreme catalog cold-start, vote-based support signals, language exposure differences).&#x20;

---

## Data splits

- **Primary split: chronological, user-based holdout**

  - For each user, sort interactions by timestamp and hold out the last *N* interactions for **test** (eg. N=5), the prior *M* interactions for **validation** (eg. M=5), and earlier interactions for **training**.
  - **Why**: Chronological splits avoid temporal leakage. Aligns training results with real-world/production results.

- **Rolling validation (Optional)**

  - Use rolling windows that advance the training window forward in time and validate on the next period to estimate production stability.

- **Cold Start Handling**

  - **Leave-movie-out** split used only as a dedicated cold-start probe, do not mix with chronological user split for main metrics. Ensure held-out movies include a mix of genres, languages and release eras.

---

## Primary metrics

- **Ranking metrics**

  - NDCG\@K (K=10,20)
  - Recall\@K (K=10,20)
  - MAP\@K (Optional: if measuring relevance precision)

- **Why**: EDA shows extreme popularity concentration; ranking metrics measure recommendation relevance and head/tail performance better than pointwise losses.

---

## Secondary metrics

- **Rating prediction**: RMSE / MAE (report only, not used)
- **Coverage**: proportion of catalog recommended, mean popularity of recommendations
- **Diversity**: by genre/ production country/ language
- **Calibration & bias**: exposure concentration, popularity uplift

Report these metrics separated by the slices below.

---

## Slices

For all primary and secondary metrics, report results for the following slices:

- Item support bins: `0`, `1–4`, `5–19`, `≥20`
- IMDb vote-support bins
- Release era bins (1900–1970, 1970–1980, 1980–1990, 1990–2000, 2000–2010, 2010–2020, 2020–2030)

---

## Model-specific metric guidance

- **Collaborative Filtering (CF) models**

  - Optimize: ranking metric (NDCG\@K / Recall\@K) on head + mid-support slices
  - Report: RMSE for completeness; cold-start slices will be poor so report separately
  - Note: apply item-level regularization and popularity-aware negative sampling during training

- **Hybrid / Content Augmented models**

  - Optimize: ranking metric on mixed slices including soft-cold and weakly-supported items
  - Report: NDCG\@K broken down by support bins and language to show gains in tail coverage; show how many cold items appear in top-K
  - Note: treat external ratings (IMDb rating) with confidence weighting (IMDb num\_votes)

- **Content Only models**

  - Optimize: ranking on hard-cold and soft-cold slices
  - Report: overall ranking + novelty/diversity
  - Note: treat external ratings (IMDb rating) with confidence weighting (IMDb num\_votes)

---

## Practical evaluation details

- **K values**: report K=5,10,20 to cover short and medium recommendation lists.
- **Statistical testing**: include confidence intervals / paired tests for primary metric differences.
- **Minimum support for bin estimates**: for stability, only compute slice metrics when the slice contains ≥N items/users (suggested N=200); still report small-slice counts.
- **Reproducibility**: fix random seeds, log dataset version and preprocessing steps, and save train/val/test splits as artifacts.

---

## Quick checklist for running an experiment

1. Generate chronological user-based train/val/test splits and save them.
2. Compute per-item support and label cold-start strata.
3. Train model on training set; select hyperparameters on validation via NDCG\@10.
4. Evaluate on test set and produce slice tables for required bins.
5. Run leave-movie-out cold-start probe and report metrics for cold-start models.
6. Save metrics, seeds, and artifacts for reproducibility.



