from pathlib import Path
from collections import Counter
import html
import re
import unicodedata

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
MOVIELENS_ROOT = RAW_ROOT / "MovieLens" / "ml-32m"
TMDB_IMDB_PATH = RAW_ROOT / "TMDB  IMDB Movies Dataset.csv"

MIN_ACTOR_APPEARANCES = 5
MIN_DIRECTOR_APPEARANCES = 2
MIN_WRITER_APPEARANCES = 3
MIN_COMPANY_APPEARANCES = 50
MIN_COUNTRY_APPEARANCES = 50
MIN_LANGUAGE_APPEARANCES = 100

PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)


def top_n_from_list(lst, n=20):
    counter = Counter(lst)
    most = counter.most_common(n)
    return [tag for tag, _ in most], [count for _, count in most]


def normalize_cast(cast_value):
    if not isinstance(cast_value, str):
        return []
    names = [name.strip().lower() for name in cast_value.split(",")]
    names = [re.sub(r"\s+", " ", name) for name in names if name]
    return list(dict.fromkeys(names))


def clean_text(text):
    if not isinstance(text, str):
        return text
    text = html.unescape(text)
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_unify_name(name):
    if not isinstance(name, str):
        return name
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    return name.title()


def parse_list_field(value, sep=",", normalize_fn=lambda x: x):
    if not isinstance(value, str) or not value.strip():
        return []
    parts = [part.strip() for part in value.split(sep) if part.strip()]
    return [normalize_fn(part) for part in parts]


def normalize_name(name):
    if not isinstance(name, str):
        return name
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    name = unicodedata.normalize("NFKD", name)
    return name.encode("ascii", "ignore").decode("ascii")


def parse_list(value):
    if not isinstance(value, str) or not value.strip():
        return []
    return [normalize_name(item) for item in value.split(",") if item.strip()]


def process_movie_tags():
    movies = pd.read_csv(MOVIELENS_ROOT / "movies.csv")
    movies = movies.drop_duplicates()
    movies["title"] = movies["title"].astype(str).str.replace(
        r"\s*\(\d{4}\)\s*$", "", regex=True
    ).str.strip()

    tags = pd.read_csv(MOVIELENS_ROOT / "tags.csv")
    tags_df = tags.copy()
    tags_df["tag_norm"] = tags_df["tag"].astype(str).str.strip().str.lower()

    grouped = tags_df.groupby("movieId")["tag_norm"].apply(list).reset_index(name="tag_norms")
    grouped[["tag_list", "tag_counts"]] = grouped["tag_norms"].apply(
        lambda lst: pd.Series(top_n_from_list(lst, 20))
    )

    movies = movies.merge(
        grouped[["movieId", "tag_list", "tag_counts"]],
        on="movieId",
        how="left",
    )

    movies["tag_list"] = movies["tag_list"].apply(lambda x: x if isinstance(x, list) else [])
    movies["tag_counts"] = movies["tag_counts"].apply(lambda x: x if isinstance(x, list) else [])

    movie_tags = movies[["movieId", "title", "tag_list", "tag_counts"]].copy()

    movie_tags.to_parquet(PROCESSED_ROOT / "movieTags.parquet", index=False)


def process_movie_links():
    links = pd.read_csv(MOVIELENS_ROOT / "links.csv")
    links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce").astype("Int64")
    links["tmdbId"] = pd.to_numeric(links.get("tmdbId", pd.Series()), errors="coerce").astype("Int64")
    links["imdbId"] = pd.to_numeric(links.get("imdbId", pd.Series()), errors="coerce").astype("Int64")

    links = links.dropna(subset=["movieId"])
    links = links.drop_duplicates()

    if links["movieId"].duplicated().sum() > 0:
        links = links.drop_duplicates(subset=["movieId"], keep="first")

    links.to_parquet(PROCESSED_ROOT / "movieLinks.parquet", index=False)


def process_movie_ratings():
    ratings = pd.read_csv(MOVIELENS_ROOT / "ratings.csv")
    ratings["userId"] = pd.to_numeric(ratings["userId"], errors="coerce").astype("Int64")
    ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce").astype("Int64")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce").astype("float64")
    ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce").astype("Int64")

    ratings = ratings.dropna(subset=["userId", "movieId", "rating", "timestamp"])
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings = ratings.sort_values(by=["userId", "movieId", "datetime"])
    ratings = ratings.drop_duplicates(subset=["userId", "movieId"], keep="last")
    ratings = ratings.drop(columns=["timestamp"])

    links_path = PROCESSED_ROOT / "movieLinks.parquet"
    data_path = PROCESSED_ROOT / "movieData.parquet"
    if links_path.exists() and data_path.exists():
        links = pd.read_parquet(links_path)
        movie_meta = pd.read_parquet(data_path)
        movie_to_imdb = links.set_index("movieId")["imdbId"]
        valid_imdb_ids = set(int(x) for x in movie_meta["imdbId"].dropna().astype(int).tolist())
        ratings["has_metadata"] = ratings["movieId"].map(movie_to_imdb).isin(valid_imdb_ids)
    else:
        ratings["has_metadata"] = False

    ratings.to_parquet(PROCESSED_ROOT / "movieRatings.parquet", index=False)


def process_users():
    ratings = pd.read_csv(MOVIELENS_ROOT / "ratings.csv")
    ratings["userId"] = pd.to_numeric(ratings["userId"], errors="coerce").astype("Int64")
    ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce").astype("Int64")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce").astype("float64")
    ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce").astype("Int64")

    ratings = ratings.dropna(subset=["userId", "movieId", "rating", "timestamp"])
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings = ratings.sort_values(by=["userId", "movieId", "datetime"])
    ratings = ratings.drop_duplicates(subset=["userId", "movieId"], keep="last")

    users = ratings.groupby("userId").agg(
        rating_count=("rating", "count"),
        first_rating=("datetime", "min"),
        last_rating=("datetime", "max"),
    ).reset_index()

    users["userId"] = users["userId"].astype("Int64")
    users["rating_count"] = users["rating_count"].astype("int64")

    users.to_parquet(PROCESSED_ROOT / "Users.parquet", index=False)


def process_movie_metadata():
    if not Path(TMDB_IMDB_PATH).exists():
        return

    df = pd.read_csv(TMDB_IMDB_PATH)

    if "status" in df.columns and "tconst" in df.columns:
        df_released = df[df["status"] == "Released"].copy()
        df_released["imdbId"] = (
            df_released["tconst"].astype(str).str.replace("tt", "", regex=False)
        )
        df_released = df_released.drop(columns=[c for c in ["status", "tconst"] if c in df_released.columns])
    else:
        df_released = df.copy()
        if "imdbId" not in df_released.columns and "tconst" in df_released.columns:
            df_released["imdbId"] = df_released["tconst"].astype(str).str.replace("tt", "", regex=False)

    if "imdbId" in df_released.columns:
        df_released["imdbId"] = pd.to_numeric(df_released["imdbId"], errors="coerce")
    df_released = df_released.dropna(subset=["imdbId"]) if "imdbId" in df_released.columns else df_released

    if "cast" in df_released.columns:
        df_released["cast_norm"] = df_released["cast"].apply(normalize_cast)
        df_released["cast_count"] = df_released["cast_norm"].apply(len)
    else:
        df_released["cast_norm"] = [[] for _ in range(len(df_released))]
        df_released["cast_count"] = 0

    df_valid = df_released.copy()
    if "imdbId" in df_valid.columns:
        df_valid = df_valid.sort_values(by=["imdbId", "cast_count"], ascending=[True, False])
        df_best = df_valid.drop_duplicates(subset=["imdbId"], keep="first")
    else:
        df_best = df_valid

    df_clean = df_best.copy()

    df_clean["title"] = df_clean.get("title", df_clean.columns[0]).astype(str)

    df_clean["release_date"] = pd.to_datetime(df_clean.get("release_date", pd.Series()), errors="coerce")
    today = pd.Timestamp.today()
    invalid_date_mask = (
        df_clean["release_date"].notna()
        & ((df_clean["release_date"] > today) | (df_clean["release_date"].dt.year < 1800))
    )
    if "release_date" in df_clean.columns:
        df_clean = df_clean[~invalid_date_mask]

    df_clean["has_release_date"] = df_clean.get("release_date").notna()
    df_clean["release_year"] = df_clean.get("release_date").dt.year.astype("Int64")

    if "runtime" in df_clean.columns:
        df_clean["runtime"] = pd.to_numeric(df_clean["runtime"], errors="coerce").astype("Int64")
        df_clean.loc[(df_clean["runtime"] < 1) | (df_clean["runtime"] > 500), "runtime"] = pd.NA
    else:
        df_clean["runtime"] = pd.NA

    if "budget" in df_clean.columns:
        df_clean["budget"] = pd.to_numeric(df_clean["budget"], errors="coerce")
        df_clean.loc[df_clean["budget"] <= 0, "budget"] = np.nan
        df_clean["budget"] = df_clean["budget"].astype("Int64")
    else:
        df_clean["budget"] = pd.NA

    if "revenue" in df_clean.columns:
        df_clean["revenue"] = pd.to_numeric(df_clean["revenue"], errors="coerce")
        df_clean.loc[df_clean["revenue"] <= 0, "revenue"] = np.nan
        df_clean["revenue"] = df_clean["revenue"].astype("Int64")
    else:
        df_clean["revenue"] = pd.NA

    rename_map = {}
    if "vote_average" in df_clean.columns:
        rename_map["vote_average"] = "average_rating_tmdb"
    if "vote_count" in df_clean.columns:
        rename_map["vote_count"] = "num_votes_tmdb"
    if "averageRating" in df_clean.columns:
        rename_map["averageRating"] = "average_rating_imdb"
    if "numVotes" in df_clean.columns:
        rename_map["numVotes"] = "num_votes_imdb"
    if "id" in df_clean.columns:
        rename_map["id"] = "tmdbId"

    if rename_map:
        df_clean = df_clean.rename(columns=rename_map)

    df_clean["average_rating_tmdb"] = pd.to_numeric(df_clean.get("average_rating_tmdb", pd.Series()), errors="coerce")
    df_clean["num_votes_tmdb"] = pd.to_numeric(df_clean.get("num_votes_tmdb", pd.Series()), errors="coerce").astype("Int64")
    df_clean["average_rating_imdb"] = pd.to_numeric(df_clean.get("average_rating_imdb", pd.Series()), errors="coerce")
    df_clean["num_votes_imdb"] = pd.to_numeric(df_clean.get("num_votes_imdb", pd.Series()), errors="coerce").astype("Int64")

    if "num_votes_tmdb" in df_clean.columns:
        df_clean.loc[df_clean["num_votes_tmdb"] == 0, "average_rating_tmdb"] = np.nan
    if "num_votes_imdb" in df_clean.columns:
        df_clean.loc[df_clean["num_votes_imdb"] == 0, "average_rating_imdb"] = np.nan

    if "original_language" in df_clean.columns:
        lang_map = {"cn": "zh", "mo": "ro", "sh": "sr", "xx": None}
        df_clean["original_language"] = df_clean["original_language"].astype(str).str.lower().replace(lang_map)
        df_clean.loc[df_clean["original_language"] == "none", "original_language"] = pd.NA
        lang_counts = df_clean["original_language"].value_counts(dropna=True)
        rare_langs = lang_counts[lang_counts < MIN_LANGUAGE_APPEARANCES].index
        df_clean.loc[df_clean["original_language"].isin(rare_langs), "original_language"] = "other"
    else:
        df_clean["original_language"] = "other"

    df_clean["has_budget"] = df_clean["budget"].notna()
    df_clean["has_revenue"] = df_clean["revenue"].notna()
    df_clean["has_poster"] = df_clean.get("poster_path").notna()
    df_clean["has_backdrop"] = df_clean.get("backdrop_path").notna()

    tmdb_img_base = "https://image.tmdb.org/t/p/w500"
    df_clean["poster_url"] = np.where(df_clean.get("poster_path").notna(), tmdb_img_base + df_clean["poster_path"], pd.NA)
    df_clean["backdrop_url"] = np.where(df_clean.get("backdrop_path").notna(), tmdb_img_base + df_clean["backdrop_path"], pd.NA)

    df_clean["overview"] = df_clean.get("overview").apply(clean_text) if "overview" in df_clean.columns else pd.NA
    df_clean["tagline"] = df_clean.get("tagline").apply(clean_text) if "tagline" in df_clean.columns else pd.NA

    df_clean["cast_list"] = df_clean.get("cast", "").fillna("").apply(lambda x: [actor.strip() for actor in str(x).split(",") if actor.strip()])
    actor_counts = Counter(actor for cast_list in df_clean["cast_list"] for actor in cast_list)
    frequent_actors = {actor for actor, count in actor_counts.items() if count >= MIN_ACTOR_APPEARANCES}
    df_clean["kept_actor_count"] = df_clean["cast_list"].apply(lambda lst: sum(actor in frequent_actors for actor in lst))

    df_processed = df_clean.copy()

    df_processed["genres"] = df_processed.get("genres", "").apply(lambda x: parse_list_field(x, sep=",", normalize_fn=lambda genre: genre.strip().lower()))
    df_processed["directors"] = df_processed.get("directors", "").apply(lambda x: parse_list_field(x, sep=",", normalize_fn=strip_unify_name))
    df_processed["writers"] = df_processed.get("writers", "").apply(lambda x: parse_list_field(x, sep=",", normalize_fn=strip_unify_name))
    df_processed["cast_list"] = df_processed["cast_list"].apply(lambda x: [strip_unify_name(a) for a in x])

    actor_counts = Counter(actor for cast_list in df_processed["cast_list"] for actor in cast_list)
    frequent_actors = {actor for actor, count in actor_counts.items() if count >= MIN_ACTOR_APPEARANCES}

    def group_actor_name(name):
        return name if name in frequent_actors else "rare_actor"

    df_processed["cast_list"] = df_processed["cast_list"].apply(lambda lst: [group_actor_name(actor) for actor in lst])

    actor_to_weight = {}
    for actor, count in actor_counts.items():
        if count < MIN_ACTOR_APPEARANCES:
            actor_to_weight[actor] = 0.1
        else:
            actor_to_weight[actor] = float(np.log1p(count))

    df_processed["cast_weight"] = df_processed["cast_list"].apply(
        lambda lst: [actor_to_weight.get(actor, 0.1) if actor != "rare_actor" else 0.1 for actor in lst]
    )

    director_counts = Counter(director for directors in df_processed["directors"] for director in directors)
    writer_counts = Counter(writer for writers in df_processed["writers"] for writer in writers)

    def group_directors(director_list):
        return [director if director_counts.get(director, 0) >= MIN_DIRECTOR_APPEARANCES else "rare_director" for director in director_list]

    def group_writers(writer_list):
        return [writer if writer_counts.get(writer, 0) >= MIN_WRITER_APPEARANCES else "rare_writer" for writer in writer_list]

    df_processed["directors"] = df_processed["directors"].apply(group_directors)
    df_processed["writers"] = df_processed["writers"].apply(group_writers)

    for c in ["kept_actor_count", "backdrop_path", "poster_path", "cast", "homepage"]:
        if c in df_processed.columns:
            df_processed = df_processed.drop(columns=[c])

    df_processed["production_companies"] = df_processed.get("production_companies", "").apply(parse_list)
    df_processed["production_countries"] = df_processed.get("production_countries", "").apply(parse_list)
    df_processed["spoken_languages"] = df_processed.get("spoken_languages", "").apply(parse_list)

    company_counts = Counter(company for companies in df_processed["production_companies"] for company in companies)
    country_counts = Counter(country for countries in df_processed["production_countries"] for country in countries)
    language_counts = Counter(language for languages in df_processed["spoken_languages"] for language in languages)

    frequent_companies = {company for company, count in company_counts.items() if count >= MIN_COMPANY_APPEARANCES}
    frequent_countries = {country for country, count in country_counts.items() if count >= MIN_COUNTRY_APPEARANCES}
    frequent_languages = {language for language, count in language_counts.items() if count >= MIN_LANGUAGE_APPEARANCES}

    def group_companies(company_list):
        return [company if company in frequent_companies else "other_company" for company in company_list]

    def group_languages(language_list):
        return [language if language in frequent_languages else "other_language" for language in language_list]

    def group_countries(country_list):
        return [country if country in frequent_countries else "other_country" for country in country_list]

    df_processed["production_companies"] = df_processed["production_companies"].apply(group_companies)
    df_processed["spoken_languages"] = df_processed["spoken_languages"].apply(group_languages)
    df_processed["production_countries"] = df_processed["production_countries"].apply(group_countries)

    df_processed["has_production_company"] = df_processed["production_companies"].apply(bool)
    df_processed["has_production_country"] = df_processed["production_countries"].apply(bool)
    df_processed["has_spoken_language"] = df_processed["spoken_languages"].apply(bool)

    df_processed["keywords_list"] = df_processed.get("keywords", "").apply(parse_list)
    df_processed["has_keywords"] = df_processed.get("keywords", "").apply(bool)

    if "tmdbId" not in df_processed.columns and "id" in df_clean.columns:
        df_processed = df_processed.rename(columns={"id": "tmdbId"})

    df_processed["has_cast"] = df_processed["cast_list"].apply(lambda lst: len(lst) > 0)
    df_processed["has_directors"] = df_processed["directors"].apply(lambda lst: len(lst) > 0)
    df_processed["has_writers"] = df_processed["writers"].apply(lambda lst: len(lst) > 0)

    df_processed["has_overview"] = df_processed["overview"].notna() & (df_processed["overview"].str.strip() != "")
    df_processed["has_tagline"] = df_processed["tagline"].notna() & (df_processed["tagline"].str.strip() != "")
    df_processed["has_runtime"] = df_processed["runtime"].notna()

    new_column_order = [
        "imdbId",
        "tmdbId",
        "title",
        "original_title",
        "original_language",
        "release_date",
        "release_year",
        "has_release_date",
        "adult",
        "overview",
        "tagline",
        "genres",
        "keywords",
        "keywords_list",
        "has_overview",
        "has_tagline",
        "has_keywords",
        "cast_list",
        "cast_weight",
        "directors",
        "writers",
        "has_cast",
        "has_directors",
        "has_writers",
        "production_companies",
        "production_countries",
        "spoken_languages",
        "has_production_company",
        "has_production_country",
        "has_spoken_language",
        "average_rating_tmdb",
        "num_votes_tmdb",
        "average_rating_imdb",
        "num_votes_imdb",
        "popularity",
        "runtime",
        "has_runtime",
        "budget",
        "revenue",
        "has_budget",
        "has_revenue",
        "poster_url",
        "backdrop_url",
        "has_poster",
        "has_backdrop",
    ]

    existing_cols = [c for c in new_column_order if c in df_processed.columns]
    df_processed = df_processed[existing_cols]

    df_processed.to_parquet(PROCESSED_ROOT / "movieData.parquet", index=False)


def main():
    process_movie_tags()
    process_movie_links()
    process_movie_metadata()
    process_movie_ratings()
    process_users()


if __name__ == "__main__":
    main()