# Issue Log

---

### 1) Extreme interaction sparsity

- **What:** \~80% of movies have zero MovieLens ratings; interaction matrix is highly skewed.
- **Impact:** CF coverage limited; offline metrics dominated by head movies; high cold-start risk.
- **Remediation:** Gate CF models by support, report slice metrics, use content/hybrid models, add long-tail sampling and loss reweighting.&#x20;
- **Priority:** High

---

### 2) Heavy-user / calibration skew

- **What:** Small set of users contribute disproportionate ratings; some rate almost everything ≥4.
- **Impact:** Training gradients dominated by extreme users; popularity amplification; biased negatives.
- **Remediation:** Per-user caps or down-weighting at training time; popularity-aware negative sampling; keep users in evaluation but control influence in training.&#x20;
- **Priority:** High

---

### 3) Missingness concentrated in tail metadata

- **What:** `revenue` 96% missing, `budget` 93% missing, `tagline` \~79% missing, `keywords` \~60% missing, `backdrop_url` \~57% missing, `overview` \~9% missing, `runtime` \~13% missing.
- **Impact:** Content signals weaker for obscure / older titles; model must learn to fallback.
- **Remediation:** Keep explicit `has_*` flags (overview, runtime, tagline, poster, backdrop, cast, directors); use missingness as a feature.&#x20;
- **Priority:** High

---

### 4) Unreliable external-rating fields and vote confidence

- **What:** `average_rating_tmdb` missing for \~37% of movies; `num_votes_*` vary widely.
- **Impact:** External ratings can be noisy when vote counts are low; using averages directly is risky.
- **Remediation:** Use Bayesian shrinkage / confidence-weighting by `num_votes_*`; treat vote counts as separate features.&#x20;
- **Priority:** High

---

### 7) Asset coverage imbalance (poster/backdrop)

- **What:** `poster_url` \~16.6% missing; `backdrop_url` \~57% missing.
- **Impact:** Visual models (CLIP) will be usable only for head items; fallback logic needed where assets absent.
- **Remediation:** Add `has_poster`/`has_backdrop` flags, add fallback logic, and measure asset-coverage by support/era slices.&#x20;
- **Resolved**

---

### 8) Temporal issues

- **What:** Current model has future user preference knowledge, but production model won't.
- **Impact:** Random splits leak future intent; temporal leakage could inflate offline metrics.
- **Remediation:** Use chronological, per-user holdouts for evaluation; keep rolling validation as optional stability check.&#x20;
- **Priority:** High

---

### 10) Language grouping ambiguity

- **What:** `original_language` has many rare values; grouping decisions currently deferred.
- **Impact:** Model complexity and slice reporting depend on grouping; small-language slices unstable.
- **Remediation:** Use min value count and an `other` bucket.&#x20;
- **Resolved**

---

### 11) Potential duplicates / inconsistent IDs

- **What:** Multiple source merges (TMDB, IMDb, MovieLens) can create duplicates or inconsistent metadata versions.
- **Impact:** Incorrect item deduplication leads to fragmented support and noisy features.
- **Remediation:** Run duplicate detection on titles + year + ids; standardize using `imdbId` field.
- **Resolved**

---

### 12) Missing or inconsistent timestamps

- **What:** `release_date` missing for \~3.7% of movies; some release dates may be incorrect (far future/past).
- **Impact:** Temporal slicing and era-based features may be incorrect for a subset.
- **Remediation:** Validate date ranges, create `has_release_date` field.
- **Resolved**

---

### 13) Storage & pipeline hygiene

- **What:** Parquet artifacts and checksums exist, but versioning must be enforced.
- **Impact:** Reproducibility risk if upstream data changes or transforms are not recorded.
- **Remediation:** Ensure dataset versioning, store checksums, save as artifacts (W&B).&#x20;
- **Resolved**

---

*Last updated: 27.12.2025*

