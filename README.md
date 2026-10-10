# CineMatch — Movie Recommendation & Analytics System (DWM)

A Streamlit mini-project with selectable recommendation methods, a MovieLens dataset loader, a model comparison dashboard, and performance charts.

## Features
- Recommendation model selector:
  - Popularity Based (Bayesian-smoothed rating)
  - Content Based (movie genres)
  - Collaborative Filtering (user-user similarity)
- Pick a user and get recommendations.
- Find genre-similar movies from a selected title.
- Dataset exploration: genres, rating distribution, highest-rated movies.
- **Full model comparison** (dashboard tab *Model comparison* or `python run_evaluation.py`): 28 metrics and 16 performance graphs, each model scored on its own predictions and its own top-K lists (see below).
- Export metrics (CSV) and all graphs (PNG / ZIP).

## Dataset
The app downloads the **MovieLens Latest Small** dataset from GroupLens on first run (about 1 MB). It contains ratings and movie metadata from MovieLens, not Netflix. The project can use Netflix data only if you have a properly licensed/available Netflix dataset in the expected CSV format. The official Netflix Prize data is a separate historical dataset and not directly plug-compatible with this project.

Official dataset page: https://grouplens.org/datasets/movielens/
Dataset files expected for local mode:
- `data/movies.csv`: `movieId,title,genres`
- `data/ratings.csv`: `userId,movieId,rating,timestamp`

## Run on Windows
1. Install Python 3.10 or newer.
2. Extract the ZIP.
3. Open Command Prompt in the extracted folder.
4. (Recommended) Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate
   ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Start the dashboard:
   ```bash
   streamlit run app.py
   ```
7. Open the local URL Streamlit prints, usually `http://localhost:8501`.

## DWM workflow
1. **Extract:** read ratings and movie metadata CSVs.
2. **Transform:** remove duplicate ratings, handle missing titles/genres, aggregate rating data.
3. **Load:** clean data into in-memory analytical tables.
4. **Mine:** generate recommendations using popularity, content similarity, and collaborative filtering.
5. **Evaluate:** test on a held-out split and compare metrics.
6. **Visualize:** inspect charts and export model metrics.

## Model comparison: metrics
All models are evaluated on the **same random 80/20 split of the ratings** (seed 42). A rating >= 4 in the test set means the user *liked* the film. A **Random baseline** is included as a reference line; "best model" rankings consider only the three project models.

| Family | Metrics | Better |
|---|---|---|
| Rating prediction | RMSE, MAE, Median AE, R², Bias (mean error), Within ±0.5, Within ±1.0 | RMSE/MAE/Median AE lower; others higher (bias closer to 0) |
| Like / dislike classification (predicted rating >= 3.5 = "like") | Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, PR-AUC, confusion matrix | higher |
| Top-K ranking (K = 10 by default, curves for K = 1..20) | Precision@K, Recall@K, F1@K, NDCG@K, MAP@K, HitRate@K, MRR@K | higher |
| Beyond accuracy | Catalogue coverage, Novelty, List diversity (genre Jaccard), Long-tail share | higher = broader / more varied (not a quality score alone) |
| Efficiency | Fit time, Scoring time (all users x all films) | lower |
| Segments | RMSE by user activity and by item popularity (cold start) | lower |

### Graphs (saved to `outputs/graphs/` by `run_evaluation.py`, shown in the app)
01 RMSE and MAE · 02 R² and accuracy bands · 03 error distribution and box plot · 04 calibration and bias by rating · 05 confusion matrices · 06 classification metrics · 07 ROC and precision-recall curves · 08 ranking metrics at K · 09 Precision/Recall/F1/NDCG/MAP/HitRate/MRR vs K and precision-recall trade-off · 10 beyond-accuracy metrics · 11 fit / scoring time · 12 RMSE by user and item segment · 13 radar of relative strengths · 14 all-metrics heatmap · 15 overall ranking · 16 error vs ranking quality.

Tables written to `outputs/`: `model_metrics_full.csv`, `best_model_per_metric.csv`, `overall_ranking.csv`, `ranking_curves_by_K.csv`, `rmse_by_segment.csv`, `calibration_by_rating.csv`.

### How each model is evaluated (`evaluation.py`)
- **Popularity:** Bayesian-smoothed item mean (same formula as the recommender); same list for every user.
- **Content (genres):** each user's average rating per genre (shrunk towards the user's mean); an item's score is the mean of the user's averages for its genres. Ties between films with identical genres are broken by a tiny popularity term.
- **Collaborative (user similarity):** cosine similarity of ratings, 40 nearest neighbours (as in the app), mean-centred weighted average for rating prediction and a support-shrunk score for ranking.
- **Random baseline:** global-mean rating, random ranking.
- Top-K lists exclude films the user rated in the training set. Ranking metrics are averaged over users who liked at least one test film.

### Notes on interpretation
- Ranking metrics are small in absolute terms (typically a few percent) because each user has thousands of unseen films and only a handful of held-out likes; compare models relative to each other and to the baseline.
- The evaluation uses vectorised equivalents of the three recommenders so every user can be scored in seconds; the recommender tables in the app use the original implementations in `recommender.py`.
- Metrics vary with the split and the data. Use `python run_evaluation.py --k 5` (or the slider in the app) to change K.

## Suggested report screenshots
1. Main dashboard with dataset counts.
2. Model selector and recommendation table.
3. Model comparison metrics table.
4. RMSE/MAE and error-distribution charts.
5. ROC / precision-recall / Precision@K / NDCG@K charts, radar and heatmap.
6. Genre and rating distribution charts.
7. Terminal showing `streamlit run app.py`.

## Citation
Harper, F. M. & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), 19:1–19:19.
