import zipfile, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def load_data(dataset_mode="Auto-download MovieLens Small"):
    DATA.mkdir(exist_ok=True)
    movies_path, ratings_path = DATA / "movies.csv", DATA / "ratings.csv"
    if movies_path.exists() and ratings_path.exists():
        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)
        movies, ratings = clean_data(movies, ratings)
        return movies, ratings, "Local data/movies.csv + data/ratings.csv"
    if dataset_mode == "Use local CSV files":
        raise FileNotFoundError("Place movies.csv and ratings.csv in the project's data/ folder.")
    zip_path = DATA / "ml-latest-small.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(SMALL_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA / "unzipped")
    folder = DATA / "unzipped" / "ml-latest-small"
    movies = pd.read_csv(folder / "movies.csv")
    ratings = pd.read_csv(folder / "ratings.csv")
    movies, ratings = clean_data(movies, ratings)
    return movies, ratings, "MovieLens Latest Small (downloaded from GroupLens)"

def clean_data(movies, ratings):
    movies = movies.copy()
    ratings = ratings.copy()
    movies = movies.drop_duplicates("movieId")
    ratings = ratings.dropna(subset=["userId","movieId","rating"]).drop_duplicates(["userId","movieId"])
    movies["title"] = movies["title"].fillna("Unknown title")
    movies["genres"] = movies["genres"].fillna("")
    return movies, ratings

def _popularity(movies, ratings):
    stats = ratings.groupby("movieId").rating.agg(["mean","count"]).reset_index()
    global_mean = ratings.rating.mean()
    m = max(5, float(stats["count"].quantile(.60)))
    stats["score"] = (stats["count"]/(stats["count"]+m))*stats["mean"] + (m/(stats["count"]+m))*global_mean
    return stats.merge(movies, on="movieId", how="inner")

def recommend_for_user(user_id, model, movies, ratings, n=10):
    watched = set(ratings.loc[ratings.userId == user_id, "movieId"].tolist())
    candidates = movies[~movies.movieId.isin(watched)].copy()
    if model == "Popularity Based":
        ranked = _popularity(movies, ratings)
        out = candidates.merge(ranked[["movieId","score","mean","count"]], on="movieId", how="left")
        out["score"] = out["score"].fillna(ratings.rating.mean())
        return out.sort_values("score", ascending=False).head(n)[["title","genres","score","mean","count"]].rename(
            columns={"mean":"average_rating","count":"rating_count"})
    if model == "Content Based (Genres)":
        user_ratings = ratings[ratings.userId == user_id]
        liked_ids = user_ratings[user_ratings.rating >= 4].movieId
        liked = movies[movies.movieId.isin(liked_ids)]
        if liked.empty:
            return recommend_for_user(user_id, "Popularity Based", movies, ratings, n)
        all_movies = pd.concat([liked, candidates]).drop_duplicates("movieId")
        genre_sets = all_movies.set_index("movieId").genres.fillna("").str.get_dummies(sep="|")
        weights = user_ratings.set_index("movieId").rating
        liked_weights = weights.reindex(genre_sets.index).fillna(0).clip(lower=0)
        profile = (genre_sets.mul(liked_weights, axis=0)).sum(axis=0)
        scores = genre_sets.mul(profile, axis=1).sum(axis=1) / max(float(np.linalg.norm(profile)), 1)
        out = candidates.merge(scores.rename("score"), left_on="movieId", right_index=True, how="left")
        return out.sort_values("score", ascending=False).head(n)[["title","genres","score"]]
    # User-user collaborative filtering: weighted average from similar users
    matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    if user_id not in matrix.index:
        return recommend_for_user(user_id, "Popularity Based", movies, ratings, n)
    filled = matrix.fillna(0)
    sim = cosine_similarity(filled)
    user_idx = list(filled.index).index(user_id)
    sims = pd.Series(sim[user_idx], index=filled.index).drop(user_id).sort_values(ascending=False).head(40)
    numer, denom = {}, {}
    for other, similarity in sims.items():
        if similarity <= 0: continue
        row = matrix.loc[other].dropna()
        for movie_id, rating in row.items():
            if movie_id not in watched:
                numer[movie_id] = numer.get(movie_id, 0) + similarity * rating
                denom[movie_id] = denom.get(movie_id, 0) + similarity
    scores = pd.DataFrame({"movieId": list(numer), "score": [numer[k]/denom[k] for k in numer]})
    return candidates.merge(scores, on="movieId", how="inner").sort_values("score", ascending=False).head(n)[["title","genres","score"]]

def recommend_similar_movies(title, movies, ratings, n=10):
    match = movies[movies.title == title]
    if match.empty: return pd.DataFrame()
    mid = match.iloc[0].movieId
    genres = set(str(match.iloc[0].genres).split("|"))
    candidate = movies[movies.movieId != mid].copy()
    candidate["genre_overlap"] = candidate.genres.fillna("").apply(lambda x: len(genres & set(x.split("|"))))
    means = ratings.groupby("movieId").rating.mean().rename("average_rating")
    candidate = candidate.join(means, on="movieId")
    candidate["score"] = candidate["genre_overlap"] + candidate["average_rating"].fillna(0)/10
    return candidate.sort_values("score", ascending=False).head(n)[["title","genres","score","average_rating"]]

def evaluate_models(movies, ratings, k=10):
    """Backward-compatible wrapper: summary table from the full per-model evaluation (see evaluation.py)."""
    from evaluation import evaluate_all
    return evaluate_all(movies, ratings, k=k)["summary"]
