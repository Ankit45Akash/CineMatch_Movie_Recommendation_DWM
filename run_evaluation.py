"""Command-line model evaluation: writes every metric table and every performance graph to outputs/.

    python run_evaluation.py                       # MovieLens Small (downloaded automatically)
    python run_evaluation.py --movies m.csv --ratings r.csv --k 10
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from recommender import load_data, clean_data
from evaluation import evaluate_all, best_per_metric
from plots import save_figures

OUT = Path(__file__).resolve().parent / "outputs"


def save_tables(res, out=OUT):
    out.mkdir(exist_ok=True)
    res["summary"].to_csv(out / "model_metrics_full.csv", index=False)
    res["curves"].to_csv(out / "ranking_curves_by_K.csv", index=False)
    res["segments"].to_csv(out / "rmse_by_segment.csv", index=False)
    res["rating_levels"].to_csv(out / "calibration_by_rating.csv", index=False)
    res["ranking"].to_csv(out / "overall_ranking.csv", index=False)
    best_per_metric(res["summary"], res["k"]).to_csv(out / "best_model_per_metric.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies"); ap.add_argument("--ratings")
    ap.add_argument("--k", type=int, default=10, help="list length for Precision@K, Recall@K, NDCG@K ... (max 20)")
    a = ap.parse_args()
    if a.movies and a.ratings:
        movies, ratings = clean_data(pd.read_csv(a.movies), pd.read_csv(a.ratings))
    else:
        movies, ratings, _ = load_data()
    res = evaluate_all(movies, ratings, k=a.k)
    save_tables(res)
    files = save_figures(res, OUT / "graphs")
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print(res["summary"].set_index("Model").T.round(4))
    print("\nOverall ranking\n", res["ranking"].round(2).to_string(index=False))
    print(f"\nSaved {len(files)} graphs to {OUT / 'graphs'} and metric tables to {OUT}")


if __name__ == "__main__":
    main()
