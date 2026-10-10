import streamlit as st
import pandas as pd
import io, zipfile
import matplotlib.pyplot as plt
import plotly.express as px
from recommender import load_data, recommend_for_user, recommend_similar_movies
from evaluation import evaluate_all, best_per_metric, MODELS
from plots import make_figures
from run_evaluation import save_tables, OUT

st.set_page_config(page_title="CineMatch | DWM Project", page_icon="🎬", layout="wide")
st.markdown("""
<style>
.main {background: linear-gradient(135deg, #101322 0%, #171d33 55%, #101322 100%);}
h1,h2,h3 {color:#f3f5ff}
[data-testid="stMetric"] {background:rgba(255,255,255,.06); padding:14px; border-radius:14px;}
</style>
""", unsafe_allow_html=True)

st.title("🎬 CineMatch")
st.caption("Movie Recommendation & Analytics System | Data Warehousing and Mining")
with st.sidebar:
    st.header("⚙️ Model controls")
    model_choice = st.selectbox("Recommendation model", [
        "Popularity Based", "Content Based (Genres)", "Collaborative Filtering (User Similarity)"
    ])
    st.divider()
    st.subheader("Dataset")
    dataset_mode = st.radio("Data source", ["Auto-download MovieLens Small", "Use local CSV files"], index=0)
    st.caption("MovieLens is a public research dataset, not Netflix viewing data.")
    st.divider()
    st.caption("Tip: run evaluation once; results are cached during this session.")

try:
    movies, ratings, source_note = load_data(dataset_mode)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.info("Check internet connection for first-run download, or place movies.csv and ratings.csv in the data/ folder.")
    st.stop()

users = sorted(ratings["userId"].dropna().unique().astype(int).tolist())
st.success(f"Dataset ready: {source_note}")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Movies", f"{movies.movieId.nunique():,}")
c2.metric("Ratings", f"{len(ratings):,}")
c3.metric("Users", f"{ratings.userId.nunique():,}")
c4.metric("Average rating", f"{ratings.rating.mean():.2f} / 5")

tab1, tab2, tab3, tab4 = st.tabs(["✨ Recommendations", "📊 Model comparison", "🔎 Dataset explorer", "📘 Project guide"])

with tab1:
    left, right = st.columns([1,2])
    with left:
        st.subheader("Personalized picks")
        user_id = st.selectbox("Select a sample user", users, index=0)
        n = st.slider("Number of recommendations", 5, 20, 10)
        liked_title = st.selectbox("Or find movies similar to…", ["(Choose a movie)"] + movies.title.sort_values().tolist())
    with right:
        if liked_title != "(Choose a movie)":
            recs = recommend_similar_movies(liked_title, movies, ratings, n)
            st.markdown(f"### Similar to **{liked_title}**")
        else:
            recs = recommend_for_user(user_id, model_choice, movies, ratings, n)
            st.markdown(f"### Recommended for user **{user_id}** · {model_choice}")
        if recs.empty:
            st.warning("Not enough data for this selection.")
        else:
            show = recs.copy()
            if "genres" in show:
                show["genres"] = show["genres"].fillna("").str.replace("|", " · ", regex=False)
            st.dataframe(show, use_container_width=True, hide_index=True)
            if "score" in recs:
                chart = recs.head(10).sort_values("score")
                st.plotly_chart(px.bar(chart, x="score", y="title", orientation="h",
                    title="Recommendation score", labels={"score":"Score","title":"Movie"}),
                    use_container_width=True)

with tab2:
    st.subheader("Compare recommendation approaches")
    st.write("All three models are scored on the **same held-out split**, and each model produces its **own** predictions and its "
             "own top-K lists, so every number below is a genuine per-model result. A random recommender is included as a reference baseline.")
    kcol, bcol = st.columns([1, 3])
    k_eval = kcol.slider("List length K", 5, 20, 10, help="Used for Precision@K, Recall@K, NDCG@K, MAP@K, HitRate@K, MRR@K")
    if bcol.button("▶ Run full model evaluation", type="primary"):
        with st.spinner("Evaluating all models and drawing graphs…"):
            res = evaluate_all(movies, ratings, k=k_eval)
            figs = make_figures(res)
            try:
                OUT.mkdir(exist_ok=True); save_tables(res)
            except OSError:
                pass
            st.session_state["eval"] = (res, figs)
    if "eval" in st.session_state:
        res, figs = st.session_state["eval"]
        S, k, meta = res["summary"], res["k"], res["meta"]
        st.caption(f"{meta['train_ratings']:,} training ratings · {meta['test_ratings']:,} test ratings · {meta['users']:,} users · "
                   f"{meta['items']:,} movies · {meta['eligible_users']:,} users with ≥1 liked test movie · liked = rating ≥ {meta['like_threshold']:g}")
        rk = res["ranking"]
        w = st.columns(len(rk))
        for col, (_, r) in zip(w, rk.iterrows()):
            col.metric(f"#{rk.index[rk.Model == r.Model][0] + 1} {r.Model}", f"avg rank {r['Average rank']:.1f}", f"{int(r['Metrics won'])} of 10 core metrics won", delta_color="off")
        t_tab, g1, g2, g3, g4, g5 = st.tabs(["📋 Tables", "📉 Rating error", "👍 Like / dislike", "🏆 Top-K ranking", "🌈 Beyond accuracy & speed", "🧭 Overall"])
        with t_tab:
            st.markdown("#### Every metric, every model")
            st.dataframe(S.set_index("Model").T.round(4), use_container_width=True)
            st.markdown("#### Best of the three models, per metric")
            st.dataframe(best_per_metric(S, k).round(4), use_container_width=True, hide_index=True)
            st.markdown("#### RMSE by user activity / item popularity")
            st.dataframe(res["segments"].round(4), use_container_width=True, hide_index=True)
        groups = {g1: ["01_", "02_", "03_", "04_", "12_"], g2: ["05_", "06_", "07_"], g3: ["08_", "09_"],
                  g4: ["10_", "11_"], g5: ["13_", "14_", "15_", "16_"]}
        for tab, prefixes in groups.items():
            with tab:
                for name, fig in figs.items():
                    if name.startswith(tuple(prefixes)):
                        st.pyplot(fig, use_container_width=True)
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download full metrics CSV", S.to_csv(index=False), "model_metrics_full.csv", "text/csv")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for name, fig in figs.items():
                img = io.BytesIO(); fig.savefig(img, format="png", bbox_inches="tight"); z.writestr(f"{name}.png", img.getvalue())
        d2.download_button("⬇ Download all graphs (ZIP of PNGs)", buf.getvalue(), "performance_graphs.zip", "application/zip")
        st.caption("Tables are also saved to the outputs/ folder; run `python run_evaluation.py` to save the graphs there as PNG files.")
    else:
        st.info("Click **Run full model evaluation** to generate the comparison tables and all performance graphs.")

with tab3:
    st.subheader("Dataset overview")
    genre_counts = (movies.assign(genre=movies.genres.fillna("").str.split("|")).explode("genre")
                    .query("genre != ''").groupby("genre").size().reset_index(name="movies")
                    .sort_values("movies", ascending=False))
    col1,col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.bar(genre_counts.head(15), x="movies", y="genre", orientation="h",
            title="Most common genres"), use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(ratings, x="rating", nbins=10, title="Rating distribution"), use_container_width=True)
    st.markdown("#### Highest-rated movies (minimum 20 ratings)")
    agg = ratings.groupby("movieId").rating.agg(["mean","count"]).reset_index()
    agg = agg[agg["count"] >= 20].merge(movies, on="movieId").sort_values("mean", ascending=False)
    st.dataframe(agg[["title","genres","mean","count"]].head(20).rename(columns={"mean":"average_rating","count":"rating_count"}),
                 use_container_width=True, hide_index=True)
    st.markdown("#### Raw data preview")
    st.dataframe(ratings.head(20), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Project guide")
    st.markdown("""
**Problem statement:** Help users discover movies using ratings and genre metadata.

**DWM pipeline:** Source CSV → cleaning and transformation → analytical tables → recommendation models → evaluation metrics → dashboard.

**Models available**
- **Popularity Based:** ranks movies by Bayesian-smoothed average rating.
- **Content Based:** recommends movies with genres similar to movies a user rated highly.
- **Collaborative Filtering:** uses user-user rating similarity to estimate preference.

**Metrics**
- **RMSE / MAE / Median AE:** rating prediction error (lower is better). **R²:** variance explained (higher is better).
- **Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, PR-AUC:** can the model tell liked (≥4) from not-liked movies?
- **Precision@K / Recall@K / F1@K / NDCG@K / MAP@K / HitRate@K / MRR@K:** quality of the top-K list (higher is better).
- **Coverage, novelty, diversity, long-tail share:** how broad and varied the recommendations are.
- **Fit time / scoring time:** speed in seconds.

**Important:** metrics are computed on a held-out sample and may vary with data/sampling. This educational prototype is not a production Netflix system.
""")
    st.download_button("⬇ Download project README", open("README.md", encoding="utf-8").read(), "README.md", "text/markdown")
