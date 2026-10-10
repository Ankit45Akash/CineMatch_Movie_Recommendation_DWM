"""Full performance evaluation of the CineMatch recommenders.

Every model is evaluated on the SAME held-out split and each model produces its OWN
predictions and its OWN top-N lists, so every metric below is a genuine per-model number.

Metric families
  1. Rating prediction : RMSE, MAE, R2, median abs error, bias, within +-0.5 / +-1.0 accuracy
  2. Like / dislike    : accuracy, precision, recall, F1, specificity, ROC-AUC, PR-AUC, confusion matrix
  3. Top-K ranking     : Precision@K, Recall@K, F1@K, NDCG@K, MAP@K, HitRate@K, MRR (+ curves for K=1..20)
  4. Beyond accuracy   : catalogue coverage, novelty, intra-list diversity, long-tail share
  5. Efficiency        : fit time, scoring time
  6. Segment analysis  : RMSE by user activity and by item popularity (cold-start behaviour)
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score, confusion_matrix)

POPULARITY = "Popularity Based"
CONTENT = "Content Based (Genres)"
COLLAB = "Collaborative Filtering (User Similarity)"
RANDOM = "Random Baseline"
MODELS = [POPULARITY, CONTENT, COLLAB]          # the three project models
ALL_MODELS = MODELS + [RANDOM]                   # + a reference baseline

LIKE_THRESHOLD = 4.0      # a held-out rating >= 4 means "the user liked it" (relevant item)
PRED_THRESHOLD = 3.5      # a predicted rating >= 3.5 is classified as "like"
K_MAX = 20
N_NEIGHBOURS = 40         # same neighbourhood size as the app's recommender


# --------------------------------------------------------------------------- models
def _bayes_popularity(R, M):
    cnt = M.sum(0).astype(float)
    mean = np.divide(R.sum(0), cnt, out=np.zeros_like(cnt), where=cnt > 0)
    gm = R.sum() / max(M.sum(), 1)
    m = max(5.0, float(np.quantile(cnt[cnt > 0], .60))) if (cnt > 0).any() else 5.0
    bayes = (cnt / (cnt + m)) * mean + (m / (cnt + m)) * gm      # identical formula to recommender.py
    return bayes, cnt, gm


def _fit_popularity(R, M, G, rng):
    bayes, _, gm = _bayes_popularity(R, M)
    return {"bayes": bayes, "gm": gm}


def _score_popularity(st, R, M, G, rng):
    U = R.shape[0]
    pred = np.tile(st["bayes"], (U, 1))
    return np.clip(pred, .5, 5), pred


def _fit_content(R, M, G, rng):
    bayes, _, gm = _bayes_popularity(R, M)
    n_u = M.sum(1)
    user_mean = np.where(n_u > 0, R.sum(1) / np.maximum(n_u, 1), gm)
    k = 5.0                                                        # shrinkage towards the user's own mean
    g_sum, g_cnt = R @ G, M.astype(float) @ G
    genre_mean = (g_sum + k * user_mean[:, None]) / (g_cnt + k)    # user x genre average rating
    return {"genre_mean": genre_mean, "G": G, "user_mean": user_mean, "bayes": bayes}


def _score_content(st, R, M, G, rng):
    G = st["G"]
    n_g = G.sum(1)
    pred = (st["genre_mean"] @ G.T) / np.maximum(n_g, 1)
    pred = np.where(n_g[None, :] > 0, pred, st["user_mean"][:, None])
    # tiny popularity tie-break: many films share an identical genre set
    return np.clip(pred, .5, 5), pred + 1e-3 * st["bayes"][None, :]


def _fit_collab(R, M, G, rng):
    bayes, _, gm = _bayes_popularity(R, M)
    n_u = M.sum(1)
    user_mean = np.where(n_u > 0, R.sum(1) / np.maximum(n_u, 1), gm)
    norms = np.linalg.norm(R, axis=1, keepdims=True)
    Rn = R / np.maximum(norms, 1e-9)
    sim = Rn @ Rn.T                          # cosine similarity of zero-filled ratings (as in the app)
    np.fill_diagonal(sim, -1)
    k = min(N_NEIGHBOURS, sim.shape[0] - 1)
    idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    W = np.zeros_like(sim)
    rows = np.arange(sim.shape[0])[:, None]
    W[rows, idx] = np.maximum(sim[rows, idx], 0)
    return {"W": W, "user_mean": user_mean, "bayes": bayes, "gm": gm}


def _score_collab(st, R, M, G, rng):
    W, um = st["W"], st["user_mean"]
    Mf = M.astype(float)
    Rc = (R - um[:, None]) * M               # mean-centred ratings (0 where unrated)
    num, den = W @ Rc, W @ Mf
    fallback = um[:, None] + (st["bayes"] - st["gm"])[None, :]       # no neighbour rated it
    pred = np.where(den > 1e-9, um[:, None] + num / np.maximum(den, 1e-9), fallback)
    rank = num / (den + 0.5) + 1e-3 * st["bayes"][None, :]          # shrunk, support-aware ranking score
    return np.clip(pred, .5, 5), rank


def _fit_random(R, M, G, rng):
    return {"gm": R.sum() / max(M.sum(), 1)}


def _score_random(st, R, M, G, rng):
    return np.full(R.shape, st["gm"]), rng.random(R.shape)


_IMPL = {POPULARITY: (_fit_popularity, _score_popularity),
         CONTENT: (_fit_content, _score_content),
         COLLAB: (_fit_collab, _score_collab),
         RANDOM: (_fit_random, _score_random)}


# --------------------------------------------------------------------------- helpers
def _prepare(movies, ratings, test_size, seed, max_ratings):
    data = ratings[ratings.movieId.isin(movies.movieId)][["userId", "movieId", "rating"]].copy()
    if len(data) > max_ratings:
        data = data.sample(max_ratings, random_state=seed)
    train, test = train_test_split(data, test_size=test_size, random_state=seed)
    users = np.sort(data.userId.unique())
    items = np.sort(data.movieId.unique())
    u_ix = {u: i for i, u in enumerate(users)}
    i_ix = {m: i for i, m in enumerate(items)}
    U, I = len(users), len(items)

    def to_idx(df):
        return df.userId.map(u_ix).to_numpy(), df.movieId.map(i_ix).to_numpy(), df.rating.to_numpy(float)

    tr_u, tr_i, tr_y = to_idx(train)
    te_u, te_i, te_y = to_idx(test)
    R = np.zeros((U, I)); M = np.zeros((U, I), bool)
    R[tr_u, tr_i] = tr_y; M[tr_u, tr_i] = True
    genres = (movies.drop_duplicates("movieId").set_index("movieId").reindex(items).genres.fillna("")
              .replace("(no genres listed)", "").str.get_dummies(sep="|"))
    G = genres.to_numpy(float)
    return R, M, G, (te_u, te_i, te_y), users, items


def _topk(score, M, k):
    s = np.where(M, -np.inf, score)                          # never recommend already-rated films
    part = np.argpartition(-s, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, part, 1), axis=1)
    return np.take_along_axis(part, order, 1)


def _ranking_curves(top, T, eligible):
    """Per-user hit matrix -> metric curves for K = 1..K_MAX (averaged over eligible users)."""
    hits = T[eligible[:, None], top[eligible]].astype(float)           # (E, K_MAX)
    n_rel = T[eligible].sum(1).astype(float)
    ks = np.arange(1, K_MAX + 1)
    ch = np.cumsum(hits, 1)
    prec = ch / ks
    rec = ch / n_rel[:, None]
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    hit = (ch > 0).astype(float)
    disc = 1 / np.log2(ks + 1)
    dcg = np.cumsum(hits * disc, 1)
    ideal_cum = np.cumsum(disc)
    idcg = ideal_cum[np.minimum(n_rel[:, None], ks[None, :]).astype(int) - 1]
    ndcg = dcg / idcg
    ap = np.cumsum(prec * hits, 1) / np.minimum(n_rel[:, None], ks[None, :])
    first = np.where(hits.any(1), hits.argmax(1) + 1, np.inf)
    mrr = np.where(first[:, None] <= ks[None, :], 1 / first[:, None], 0.0)
    return pd.DataFrame({"K": ks, "Precision": prec.mean(0), "Recall": rec.mean(0), "F1": f1.mean(0),
                         "NDCG": ndcg.mean(0), "MAP": ap.mean(0), "HitRate": hit.mean(0), "MRR": mrr.mean(0)})


def _rating_metrics(y, p):
    e = p - y
    sst = ((y - y.mean()) ** 2).sum()
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "R2": float(1 - (e ** 2).sum() / sst) if sst else 0.0,
            "Median AE": float(np.median(np.abs(e))), "Bias (mean error)": float(e.mean()),
            "Within ±0.5": float(np.mean(np.abs(e) <= .5)), "Within ±1.0": float(np.mean(np.abs(e) <= 1.0))}


def _class_metrics(y, p):
    yb, pb = y >= LIKE_THRESHOLD, p >= PRED_THRESHOLD
    tn, fp, fn, tp = confusion_matrix(yb, pb, labels=[False, True]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return ({"Accuracy": (tp + tn) / len(y), "Precision (like)": prec, "Recall (like)": rec,
             "F1 (like)": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
             "Specificity": tn / (tn + fp) if tn + fp else 0.0,
             "ROC-AUC": float(roc_auc_score(yb, p)), "PR-AUC": float(average_precision_score(yb, p))},
            np.array([[tn, fp], [fn, tp]]))


# --------------------------------------------------------------------------- main entry
def evaluate_all(movies, ratings, k=10, test_size=0.2, seed=42, max_ratings=200_000):
    """Run the full evaluation. Returns a dict of DataFrames / arrays used by plots.py and the app."""
    rng = np.random.default_rng(seed)
    R, M, G, (te_u, te_i, te_y), users, items = _prepare(movies, ratings, test_size, seed, max_ratings)
    U, I = R.shape
    k = int(min(max(k, 1), K_MAX))
    item_cnt = M.sum(0)
    head = np.argsort(-item_cnt)[: max(1, int(.2 * I))]                  # top-20% most rated films
    is_head = np.zeros(I, bool); is_head[head] = True
    novelty_w = -np.log2((item_cnt + 1) / (U + 1))                        # self-information of an item
    G_ = (G > 0).astype(float); g_sz = G_.sum(1)
    T = np.zeros((U, I), bool)
    rel = te_y >= LIKE_THRESHOLD
    T[te_u[rel], te_i[rel]] = True
    eligible = np.where(T.sum(1) > 0)[0]
    user_cnt = M.sum(1)
    div_users = rng.choice(U, size=min(U, 1000), replace=False)

    rows, curves, roc, pr, conf, resid, seg_rows, level_rows = [], [], {}, {}, {}, {}, [], []
    for name in ALL_MODELS:
        fit, score = _IMPL[name]
        t0 = time.perf_counter(); st = fit(R, M, G, rng); fit_t = time.perf_counter() - t0
        t0 = time.perf_counter(); pred_full, rank_full = score(st, R, M, G, rng); score_t = time.perf_counter() - t0
        p = pred_full[te_u, te_i]

        row = {"Model": name, **_rating_metrics(te_y, p)}
        cm, conf_mat = _class_metrics(te_y, p)
        row.update(cm); conf[name] = conf_mat
        yb = te_y >= LIKE_THRESHOLD
        roc[name] = roc_curve(yb, p)[:2]
        pr_p, pr_r, _ = precision_recall_curve(yb, p)
        pr[name] = (pr_r, pr_p)

        top = _topk(rank_full, M, K_MAX)
        cv = _ranking_curves(top, T, eligible); cv.insert(0, "Model", name); curves.append(cv)
        at_k = cv[cv.K == k].iloc[0]
        row.update({f"Precision@{k}": at_k.Precision, f"Recall@{k}": at_k.Recall, f"F1@{k}": at_k.F1,
                    f"NDCG@{k}": at_k.NDCG, f"MAP@{k}": at_k.MAP, f"HitRate@{k}": at_k.HitRate,
                    f"MRR@{k}": at_k.MRR})
        tk = top[:, :k]
        divs = []
        for u in div_users:
            g = G_[tk[u]]; inter = g @ g.T
            union = g_sz[tk[u]][:, None] + g_sz[tk[u]][None, :] - inter
            jac = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
            divs.append(1 - (jac.sum() - k) / (k * (k - 1))) if k > 1 else divs.append(0.0)
        row.update({"Catalogue coverage": len(np.unique(tk)) / I, "Novelty": float(novelty_w[tk].mean()),
                    "Diversity": float(np.mean(divs)), "Long-tail share": float(1 - is_head[tk].mean()),
                    "Fit time (s)": fit_t, "Scoring time (s)": score_t})
        rows.append(row)

        resid[name] = (te_y, p)
        a_bins = pd.cut(user_cnt[te_u], [-1, 20, 50, 100, np.inf], labels=["≤20", "21–50", "51–100", ">100"])
        p_bins = pd.cut(item_cnt[te_i], [-1, 0, 5, 20, 100, np.inf], labels=["0 (cold)", "1–5", "6–20", "21–100", ">100"])
        for kind, bins in (("User activity (train ratings)", a_bins), ("Item popularity (train ratings)", p_bins)):
            for b in bins.categories:
                sel = np.asarray(bins == b)
                if sel.sum():
                    seg_rows.append({"Model": name, "Segment": kind, "Bucket": b, "n": int(sel.sum()),
                                     "RMSE": float(np.sqrt(np.mean((p[sel] - te_y[sel]) ** 2)))})
        for lvl in np.unique(te_y):
            sel = te_y == lvl
            level_rows.append({"Model": name, "True rating": lvl, "Mean prediction": float(p[sel].mean()),
                               "Mean error": float((p[sel] - lvl).mean()), "n": int(sel.sum())})

    summary = pd.DataFrame(rows)
    return {"summary": summary, "k": k, "curves": pd.concat(curves, ignore_index=True), "roc": roc, "pr": pr,
            "confusion": conf, "residuals": resid, "segments": pd.DataFrame(seg_rows),
            "rating_levels": pd.DataFrame(level_rows), "ranking": overall_ranking(summary, k),
            "meta": {"users": U, "items": I, "train_ratings": int(M.sum()), "test_ratings": int(len(te_y)),
                     "eligible_users": int(len(eligible)), "k": k, "like_threshold": LIKE_THRESHOLD,
                     "pred_threshold": PRED_THRESHOLD, "seed": seed}}


def metric_directions(k=10):
    """metric -> True when higher is better."""
    higher = ["R2", "Within ±0.5", "Within ±1.0", "Accuracy", "Precision (like)", "Recall (like)", "F1 (like)",
              "Specificity", "ROC-AUC", "PR-AUC", f"Precision@{k}", f"Recall@{k}", f"F1@{k}", f"NDCG@{k}",
              f"MAP@{k}", f"HitRate@{k}", f"MRR@{k}", "Catalogue coverage", "Novelty", "Diversity",
              "Long-tail share"]
    lower = ["RMSE", "MAE", "Median AE", "Fit time (s)", "Scoring time (s)"]
    return {**{m: True for m in higher}, **{m: False for m in lower}}


ACCURACY_METRICS = lambda k: ["RMSE", "MAE", "R2", "F1 (like)", "ROC-AUC", f"Precision@{k}", f"Recall@{k}",
                              f"NDCG@{k}", f"MAP@{k}", f"HitRate@{k}"]


def overall_ranking(summary, k=10):
    """Average rank (1 = best) of the three project models over the core accuracy metrics."""
    s = summary[summary.Model.isin(MODELS)].set_index("Model")
    d = metric_directions(k)
    ranks = pd.DataFrame({m: s[m].rank(ascending=not d[m], method="min") for m in ACCURACY_METRICS(k)})
    out = pd.DataFrame({"Model": ranks.index, "Average rank": ranks.mean(1).to_numpy(),
                        "Metrics won": (ranks == 1).sum(1).to_numpy()})
    return out.sort_values("Average rank").reset_index(drop=True)


def best_per_metric(summary, k=10):
    """Winner among the three project models for every metric."""
    s = summary[summary.Model.isin(MODELS)].set_index("Model")
    d = metric_directions(k)
    return pd.DataFrame([{"Metric": m, "Better when": "higher" if d[m] else "lower",
                          "Best model": (s[m].idxmax() if d[m] else s[m].idxmin()),
                          "Best value": (s[m].max() if d[m] else s[m].min())} for m in d if m in s.columns])
