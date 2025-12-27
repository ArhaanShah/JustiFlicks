# Dataset Inventory

## Summary
| Dataset | Rows | Cols | Mem (MB) |
|---|---:|---:|---:|
| movieTags | 87,585 | 4 | 3.7 |
| movieLinks | 87,585 | 3 | 1.7 |
| movieRatings | 32,000,204 | 5 | 237.4 |
| movieData | 424,552 | 39 | 149.4 |
| Users | 200,948 | 4 | 4.7 |

---

## movieTags
**Purpose:** Movie titles, and movie tags used as auxiliary semantic signals.

**Primary key:** `movieId`

**Key columns:**
- `movieId`
- `title`
- `tag_list`
- `tag_counts`

**Notes:** Tags are user-generated and noisy; primarily used for text/embedding features rather than canonical metadata.

---

## movieLinks
**Purpose:** ID mapping between MovieLens movies and external identifiers.

**Primary key:** `movieId`

**Key columns:**
- `movieId`
- `imdbId`
- `tmdbId`

**Notes:** `tmdbId` has a small fraction of missing values (~0.14%).

---

## movieRatings
**Purpose:** Core interaction log used for training, evaluation, and temporal analysis.

**Grain:** One row per `(userId, movieId, datetime)` rating event.

**Key columns:**
- `userId`
- `movieId`
- `rating`
- `datetime`
- `has_metadata`

**Notes:** Very large table (~32M rows). Used for collaborative signals, chronological splits, and engagement statistics.

---

## movieData
**Purpose:** Canonical movie metadata table used for side features and content-based modeling.

**Primary key:** `imdbId`

**Selected columns:**
- Identifiers: `imdbId`, `tmdbId`
- Titles & language: `title`, `original_title`, `original_language`
- Temporal: `release_date`, `release_year`, `has_release_date`
- Text: `overview`, `tagline`, `keywords_list`
- People: `cast_list`, `directors`, `writers`
- Categories: `genres`, `production_countries`, `spoken_languages`
- Assets: `poster_url`, `backdrop_url`, `has_poster`, `has_backdrop`
- External signals: `average_rating_tmdb`, `num_votes_tmdb`, `average_rating_imdb`, `num_votes_imdb`, `popularity`
- Numeric attributes: `runtime`, `budget`, `revenue`
- Missingness flags: `has_keywords`, `has_cast`, `has_directors`, `has_writers`, `has_budget`, `has_revenue`, `has_production_company`, `has_production_country`, `has_spoken_language`

**Notes:** Central source for all non-interaction features. Includes explicit `has_*` flags to model missingness and enable robust cold-start behavior.

---

## Users
**Purpose:** User-level summary table for sampling, diagnostics, and split construction.

**Primary key:** `userId`

**Key columns:**
- `userId`
- `rating_count`
- `first_rating`
- `last_rating`

**Notes:** Used to analyze engagement skew, and to construct chronological per-user train/validation/test splits.

---

## Join Graph
- `movieRatings.movieId` → `movieLinks.movieId`
- `movieLinks.imdbId` → `movieData.imdbId`
- `movieTags.movieId` → `movieLinks.movieId`

`imdbId` is treated as the canonical movie identifier when combining metadata sources.

