# clustering.py — Refactored for stability, speed, and KMeans-only enforcement
# --------------------------------------------------------------

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import List, Iterable, Dict, Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

DEFAULT_YEAR_COL = "year"
DEFAULT_EMBED_COL = "embedding"
DEFAULT_TEXT_COLS: Tuple[str, ...] = ("title",)
DEFAULT_K = 100

TITLE_COLS_CAND = ("title", "paper_title", "doc_title", "name", "subject", "headline")
_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")

STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "have", "has", "had", "but", "not", "you",
    "your", "our", "their", "its", "it's", "they", "them", "his", "her", "she", "him", "who", "what", "when", "where", "how",
    "why", "can", "could", "would", "should", "a", "an", "of", "to", "in", "on", "at", "by", "as", "is", "be", "or", "if", "we",
    "et", "al", "via", "using", "use", "based", "into", "over", "under", "per", "also", "may", "might", "than", "then", "out",
    "up", "down", "new", "more", "most", "less", "least", "such", "these", "those", "each", "other"
}


# ------------------------------------------------------------
# 공통 유틸
# ------------------------------------------------------------
def _simple_tokenize(t: str) -> List[str]:
    toks = _TOKEN_RE.findall((t or "").lower())
    return [x for x in toks if len(x) >= 2 and x not in STOP_WORDS]


def _pick_text_cols(df: pd.DataFrame, user_cols: Iterable[str] | None = None) -> List[str]:
    # 사용자가 지정한 컬럼 중 존재하는 것
    if user_cols:
        cols = [c for c in user_cols if c in df.columns]
        if cols:
            return cols
    # 프리셋 후보
    for c in TITLE_COLS_CAND:
        if c in df.columns:
            return [c]
    # 문자열 컬럼 폴백
    obj = [c for c in df.columns if df[c].dtype == "object" and c.lower() != "embedding"]
    return obj[:2]


def _join(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    return df[list(cols)].fillna("").astype(str).agg(" ".join, axis=1)


def _as_vec(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.asarray(v, dtype=np.float32)
    if isinstance(v, str):
        s = v.strip()
        try:
            return np.asarray(json.loads(s), dtype=np.float32)
        except Exception:
            try:
                return np.asarray(ast.literal_eval(s), dtype=np.float32)
            except Exception:
                return None
    return None


def _embed_matrix(df: pd.DataFrame, col: str) -> np.ndarray:
    arr = []
    for v in df[col].to_numpy():
        vec = _as_vec(v)
        if vec is None:
            raise ValueError("Invalid embedding format in column 'embedding'")
        arr.append(vec)
    X = np.vstack(arr)
    return X


# ------------------------------------------------------------
# GLOBAL TF-IDF (한 번만 fit)
# ------------------------------------------------------------
def _build_global_tfidf(df: pd.DataFrame, text_cols: Iterable[str]) -> TfidfVectorizer:
    all_text = _join(df, text_cols)
    vec = TfidfVectorizer(
        tokenizer=_simple_tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=120_000,
    )
    vec.fit(all_text)
    return vec


# ------------------------------------------------------------
# Per-year clustering
# ------------------------------------------------------------
def _cluster_one_year(
    df_year: pd.DataFrame,
    year: int,
    k: int,
    embed_col: str,
    label_col: str,
    vec_global: TfidfVectorizer,
    text_cols: Iterable[str] | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_year = df_year.reset_index(drop=True)

    # 1) 임베딩 정규화 + 차원축소
    X = _embed_matrix(df_year, embed_col).astype(np.float32)
    Xn = normalize(X, axis=1)

    n_docs = len(df_year)
    ncomp = max(2, min(50, n_docs - 1))
    if ncomp < 2:
        Xr = Xn
    else:
        svd = TruncatedSVD(n_components=ncomp, random_state=42)
        Xr = svd.fit_transform(Xn)

    # 2) MiniBatchKMeans
    k_eff = max(1, min(k, n_docs))
    km = MiniBatchKMeans(
        n_clusters=k_eff,
        batch_size=min(1024, max(64, n_docs)),
        reassignment_ratio=0.01,
        init="k-means++",
        n_init=5,
        random_state=42,
    )
    labels = km.fit_predict(Xr)
    df_year[label_col] = labels

    # 3) counts
    counts = df_year[label_col].value_counts().sort_index()
    counts_df = counts.rename_axis("cluster_id").reset_index(name="count")
    counts_df["ratio"] = counts_df["count"] / counts_df["count"].sum()

    # 4) TF-IDF (전역 vocab 사용, text_cols 인자 사용)
    cols = _pick_text_cols(df_year, text_cols)
    if not cols:
        tfidf_df = pd.DataFrame(
            [{"cluster_id": None, "term": "", "tfidf": 0.0, "docs": 0}]
        )
        return df_year, counts_df, tfidf_df

    text = _join(df_year, cols)
    Xtf = vec_global.transform(text)  # sparse
    vocab = np.array(vec_global.get_feature_names_out())

    rows: List[Dict[str, Any]] = []
    for cid in counts_df["cluster_id"]:
        mask = (df_year[label_col] == cid)
        if not mask.any():
            continue
        vec_mean = np.asarray(Xtf[mask].mean(axis=0)).ravel()
        if vec_mean.size == 0:
            continue
        top = vec_mean.argsort()[::-1][:30]
        for t, sc in zip(vocab[top], vec_mean[top]):
            rows.append(
                {"cluster_id": int(cid), "term": t, "tfidf": float(sc), "docs": int(mask.sum())}
            )

    if rows:
        tfidf_df = pd.DataFrame(rows)
    else:
        tfidf_df = pd.DataFrame(
            [{"cluster_id": None, "term": "", "tfidf": 0.0, "docs": 0}]
        )

    return df_year, counts_df, tfidf_df


# ------------------------------------------------------------
# Matching (year-to-year) — TF-IDF 벡터는 전역 vec_global 재사용
# ------------------------------------------------------------
def _build_joint_lsa(
    text1: pd.Series,
    text2: pd.Series,
    vec_global: TfidfVectorizer,
) -> Tuple[np.ndarray, np.ndarray]:
    # 전역 vocab 으로 transform 만 수행
    joint = pd.concat([text1, text2], ignore_index=True)
    X = vec_global.transform(joint)  # sparse

    n_samples, n_feat = X.shape
    ncomp = min(80, max(2, min(n_samples - 1, n_feat - 1)))
    if ncomp < 2:
        Z = normalize(X.toarray(), axis=1)
    else:
        svd = TruncatedSVD(n_components=ncomp, random_state=42)
        Z = normalize(svd.fit_transform(X), axis=1)
    return Z[: len(text1)], Z[len(text1) :]


def _centroids(Z: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.unique(labels)
    C, meta = [], []
    for cid in ids:
        m = labels == cid
        if m.sum() > 0:
            C.append(Z[m].mean(axis=0))
            meta.append(cid)
    return np.vstack(C), np.array(meta)


def _match_years(
    y1: int,
    y2: int,
    k: int,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    text_cols: Iterable[str],
    label_col: str,
    vec_global: TfidfVectorizer,
    thr: float = 0.08,
) -> pd.DataFrame:
    cols1 = _pick_text_cols(df1, text_cols)
    cols2 = _pick_text_cols(df2, text_cols)
    t1 = _join(df1, cols1)
    t2 = _join(df2, cols2)

    Z1, Z2 = _build_joint_lsa(t1, t2, vec_global)
    C1, ids1 = _centroids(Z1, df1[label_col].to_numpy())
    C2, ids2 = _centroids(Z2, df2[label_col].to_numpy())

    S = cosine_similarity(C1, C2)
    j = S.argmax(axis=1)
    sims = S[np.arange(len(ids1)), j]
    matched = sims >= thr

    return pd.DataFrame(
        {
            "year_from": y1,
            "year_to": y2,
            "prev_id": ids1,
            "next_id": np.where(matched, ids2[j], None),
            "similarity": np.where(matched, sims, np.nan),
            "matched": matched,
        }
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run_clustering(
    df_embed: pd.DataFrame,
    n_clusters: int = DEFAULT_K,
    year_col: str = DEFAULT_YEAR_COL,
    embed_col: str = DEFAULT_EMBED_COL,
    text_cols: Iterable[str] = DEFAULT_TEXT_COLS,
    output_root: Path | str = Path("cluster_out"),
    match_threshold: float = 0.08,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    output_root = Path(output_root).resolve()
    years = sorted(
        pd.to_numeric(df_embed[year_col], errors="coerce").dropna().astype(int).unique()
    )

    # ---- Global TF-IDF vocab (한 번만 fit) ----
    global_text_cols = _pick_text_cols(df_embed, text_cols)
    vec_global = _build_global_tfidf(df_embed, global_text_cols)

    label_col = f"cluster_k{n_clusters}"

    clustered_parts: List[pd.DataFrame] = []
    counts_by: Dict[int, pd.DataFrame] = {}
    tfidf_by: Dict[int, pd.DataFrame] = {}

    total = len(years)
    done = 0
    if progress_cb:
        progress_cb(0, total)

    # ---- 1) 연도별 클러스터링 ----
    for y in years:
        df_y = df_embed[df_embed[year_col] == y].copy()
        if df_y.empty:
            done += 1
            if progress_cb:
                progress_cb(done, total)
            continue

        df_l, cdf, tdf = _cluster_one_year(
            df_y,
            y,
            n_clusters,
            embed_col,
            label_col,
            vec_global,
            text_cols,
        )
        clustered_parts.append(df_l)
        counts_by[int(y)] = cdf
        tfidf_by[int(y)] = tdf

        done += 1
        if progress_cb:
            progress_cb(done, total)

    if not clustered_parts:
        return df_embed.copy(), {
            "keywords": [],
            "titles": [],
            "paths": {},
            "artifacts": {},
        }

    df_all = pd.concat(clustered_parts, ignore_index=True)

    # ---- 2) 전이 매칭 ----
    edges: List[pd.DataFrame] = []
    for i in range(len(years) - 1):
        y1, y2 = int(years[i]), int(years[i + 1])
        e = _match_years(
            y1,
            y2,
            n_clusters,
            df_all[df_all[year_col] == y1],
            df_all[df_all[year_col] == y2],
            text_cols,
            label_col,
            vec_global,
            match_threshold,
        )
        edges.append(e)

    # ---- 3) Flow Year Metrics ----
    flow: List[Dict[str, Any]] = []
    for i in range(len(years) - 1):
        y1, y2 = int(years[i]), int(years[i + 1])
        links = edges[i]
        c1 = counts_by[y1].rename(columns={"cluster_id": "cid"})
        c2 = counts_by[y2].rename(columns={"cluster_id": "cid"})
        for _, r in links[links["matched"]].iterrows():
            d1 = int(c1.loc[c1["cid"] == r["prev_id"], "count"].values[0])
            d2 = int(c2.loc[c2["cid"] == r["next_id"], "count"].values[0])
            yoy = (d2 - d1) / d1 if d1 > 0 else np.nan
            flow.append(
                {
                    "year_from": y1,
                    "year_to": y2,
                    "prev_id": int(r["prev_id"]),
                    "next_id": int(r["next_id"]),
                    "similarity": float(r["similarity"]),
                    "docs_from": d1,
                    "docs_to": d2,
                    "yoy_docs": yoy,
                }
            )
    flow_df = pd.DataFrame(flow)

    # ---- 4) 키워드 / 대표 타이틀 ----
    kw: Dict[str, float] = {}
    for tdf in tfidf_by.values():
        for _, r in tdf.iterrows():
            term = str(r["term"])
            if not term:
                continue
            val = float(r.get("tfidf", 0.0))
            kw[term] = kw.get(term, 0.0) + val
    keywords = [
        t for t, _ in sorted(kw.items(), key=lambda x: x[1], reverse=True)[:100]
    ]

    titles: List[str] = []
    for (y, cid), g in df_all.groupby([year_col, label_col]):
        cols = _pick_text_cols(g, text_cols)
        if cols:
            sample = g[cols].head(2).astype(str).values.ravel().tolist()
            titles.extend(sample)
            if len(titles) >= 120:
                break

    summary = {
        "keywords": keywords,
        "titles": titles[:120],
        "paths": {
            "yearly_dir": str(output_root / "yearly_v4"),
            "compare_dir": str(output_root / f"compare_methods_k{n_clusters}"),
            "label_col": label_col,
        },
        "artifacts": {
            "years": years,
            "label_col": label_col,
            "clustered_by_year": {
                int(y): df_all[df_all[year_col] == y].copy() for y in years
            },
            "counts_by_year": counts_by,
            "tfidf_by_year": tfidf_by,
            "edges": edges,
            "flow_edges_df": flow_df.copy(),
        },
    }

    return df_all, summary
