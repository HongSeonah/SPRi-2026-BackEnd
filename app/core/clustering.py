# app/core/clustering.py
from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2norm
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 기본 설정
# -----------------------------
DEFAULT_YEAR_COL = "year"
DEFAULT_EMBED_COL = "embedding"
DEFAULT_TEXT_COLS: Tuple[str, ...] = ("title",)  # 없으면 자동탐색
DEFAULT_K = 100

ROOT = Path("cluster_out").resolve()  # 절대경로 (경로 정보는 summary에 참고용으로만 제공)

# 간단 스톱워드 (NLTK 의존 제거)
_BASIC_STOPWORDS = {
    "the","and","for","with","that","this","from","are","was","were","have","has","had","but","not","you",
    "your","our","their","its","it's","they","them","his","her","she","him","who","what","when","where","how",
    "why","can","could","would","should","a","an","of","to","in","on","at","by","as","is","be","or","if","we",
    "et","al","via","using","use","based","into","over","under","per","also","may","might","than","then","out",
    "up","down","new","more","most","less","least","such","these","those","each","other"
}
_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")

# 텍스트 컬럼 후보 (없으면 자동 선택)
TITLE_COLS_CAND = ("title","paper_title","doc_title","name","subject","headline")

# -----------------------------
# 유틸
# -----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _join_text(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    return df[list(cols)].fillna("").astype(str).agg(" ".join, axis=1)

def _simple_tokenize(text: str) -> List[str]:
    # 한글/영문 2글자 토큰까지 허용하여 정보 손실 최소화
    toks = _TOKEN_RE.findall((text or "").lower())
    return [t for t in toks if len(t) >= 2 and t not in _BASIC_STOPWORDS]

def _pick_text_cols(df: pd.DataFrame, user_cols: Iterable[str] | None) -> list[str]:
    # 사용자가 지정한 컬럼 중 존재하는 것
    if user_cols:
        cols = [c for c in user_cols if c in df.columns]
        if cols:
            return cols
    # 프리셋 후보
    for c in TITLE_COLS_CAND:
        if c in df.columns:
            return [c]
    # 마지막 폴백: 문자열형(object) 컬럼(embedding 제외)에서 1~2개
    obj_cols = [c for c in df.columns if df[c].dtype == "object" and c.lower() != "embedding"]
    return obj_cols[:2]

def _embedding_matrix(df_year: pd.DataFrame, embed_col: str) -> np.ndarray:
    # embedding 컬럼이 list/ndarray/JSON string일 수 있음 → 전부 ndarray로 변환
    def _as_vec(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return np.asarray(v, dtype=np.float32)
        if isinstance(v, str):
            s = v.strip()
            try:
                import json
                return np.asarray(json.loads(s), dtype=np.float32)
            except Exception:
                try:
                    import ast
                    return np.asarray(ast.literal_eval(s), dtype=np.float32)
                except Exception:
                    return None
        return None

    arrs = []
    for v in df_year[embed_col].to_numpy():
        vec = _as_vec(v)
        if vec is None:
            raise ValueError("embedding 컬럼에 변환 불가 항목이 있습니다.")
        arrs.append(vec)
    X = np.vstack(arrs)
    return X

def _empty_tfidf_df(reason: str, meta: Dict[str, Any] | None = None) -> pd.DataFrame:
    return pd.DataFrame([{
        "cluster_id": np.nan, "term": "", "tfidf": 0.0, "docs": 0,
        "error_reason": reason, "meta": meta or {}
    }])

# -----------------------------
# ① 연도별 클러스터링 (메모리 전용)
# -----------------------------
def _cluster_one_year(
    df_year: pd.DataFrame,
    year: int,
    k: int,
    embed_col: str,
    text_cols: Iterable[str] | None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    반환: (라벨 달린 DF, counts_df, tfidf_df)
    - PCA/LSA 소표본 가드
    - k > n_docs 방지 (k_eff 적용)
    - 텍스트 없거나 적을 때 TF-IDF 완화/폴백 + 재시도(백오프)
    """
    df_year = df_year.reset_index(drop=True)

    # 1) 임베딩 정규화 + 차원축소 (소표본 가드)
    X = _embedding_matrix(df_year, embed_col)
    Xn = normalize(X, axis=1)

    n_docs, n_feat = Xn.shape[0], Xn.shape[1]
    ncomp = max(1, min(50, n_feat, n_docs - 1))  # 1..min(n_docs-1, n_feat)
    if ncomp >= 2:
        Xr = PCA(n_components=ncomp, svd_solver="full", random_state=42).fit_transform(Xn)
    else:
        Xr = Xn  # 너무 적으면 PCA 생략

    # 2) KMeans (소표본 가드)
    k_eff = max(1, min(k, n_docs))   # 샘플 수보다 큰 k 방지
    mbk = MiniBatchKMeans(n_clusters=k_eff, random_state=42, n_init="auto")
    labels = mbk.fit_predict(Xr)
    label_col = f"cluster_k{k}"      # 열 이름은 원래 k 유지(유니크 라벨 수는 k_eff)
    df_year[label_col] = labels

    # 3) counts (메모리)
    counts = df_year[label_col].value_counts().sort_index()
    counts_df = counts.rename_axis("cluster_id").reset_index(name="count")
    counts_df["ratio"] = counts_df["count"] / counts_df["count"].sum()

    # 4) tf-idf (텍스트 컬럼 자동 선택 + 소표본 완화)
    safe_text_cols = _pick_text_cols(df_year, text_cols)

    try:
        if not safe_text_cols:
            tfidf_df = _empty_tfidf_df("no_text_columns",
                                       {"picked_cols": [], "available_cols": list(df_year.columns)})
            return df_year, counts_df, tfidf_df

        text = _join_text(df_year, safe_text_cols)
        if text.str.len().fillna(0).sum() == 0:
            tfidf_df = _empty_tfidf_df("empty_text_after_join", {"picked_cols": safe_text_cols})
            return df_year, counts_df, tfidf_df

        # 소표본일수록 완화
        min_df_val = 1 if n_docs < 20 else 3
        # 고중복/소표본 보호: max_df 완화
        max_df_val = 1.0 if n_docs <= 20 else 0.95

        def _fit_tfidf(min_df_v: int | float, max_df_v: float) -> tuple[TfidfVectorizer, Any]:
            tf = TfidfVectorizer(
                tokenizer=_simple_tokenize,
                token_pattern=None,          # tokenizer 사용시 경고 제거
                ngram_range=(1, 2),
                min_df=min_df_v,
                max_df=max_df_v,
                max_features=120_000
            )
            Xtf_local = tf.fit_transform(text)
            return tf, Xtf_local

        # 1차 시도
        tf, Xtf = _fit_tfidf(min_df_val, max_df_val)

        # 피처가 0이면 백오프 재시도: max_df=1.0, min_df=1
        if Xtf.shape[1] == 0:
            tf, Xtf = _fit_tfidf(1, 1.0)

        if Xtf.shape[1] == 0:
            tfidf_df = _empty_tfidf_df(
                "tfidf_no_features_after_backoff",
                {"n_docs": n_docs, "min_df_try": [min_df_val, 1], "max_df_try": [max_df_val, 1.0], "picked_cols": safe_text_cols}
            )
            return df_year, counts_df, tfidf_df

        vocab = np.array(tf.get_feature_names_out())
        rows: List[Dict[str, Any]] = []

        labels_arr = df_year[label_col].to_numpy()  # (n_docs,)
        for cid in np.unique(labels_arr):
            mask = (labels_arr == cid)  # (n_docs,) boolean
            if not mask.any():
                continue
            # 클러스터 문서들의 TF-IDF 평균 벡터
            vec = np.asarray(Xtf[mask].mean(axis=0)).ravel()
            if vec.size == 0:
                continue
            top = vec.argsort()[::-1][:30]
            for t, sc in zip(vocab[top], vec[top]):
                rows.append({"cluster_id": int(cid), "term": t, "tfidf": float(sc), "docs": int(mask.sum())})

        if not rows:
            tfidf_df = _empty_tfidf_df(
                "no_rows_after_scoring",
                {"n_docs": n_docs, "features": int(Xtf.shape[1]), "picked_cols": safe_text_cols}
            )
            return df_year, counts_df, tfidf_df

        tfidf_df = pd.DataFrame(rows)

    except Exception as e:
        tfidf_df = _empty_tfidf_df(
            "exception",
            {"message": str(e), "picked_cols": safe_text_cols, "n_docs": n_docs}
        )

    return df_year, counts_df, tfidf_df

# -----------------------------
# ② 전이 매칭 (연속 연도 간, 메모리 전용)
# -----------------------------
def _build_joint_lsa(text_prev: pd.Series, text_next: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_df=0.95,
                          tokenizer=_simple_tokenize, token_pattern=None)
    joint = pd.concat([text_prev, text_next], ignore_index=True)
    X = vec.fit_transform(joint)

    # TruncatedSVD 제약: 1 < n_components <= min(n_samples, n_features)-1
    n_samples, n_features = X.shape
    ncomp = min(256, max(2, min(n_features - 1, n_samples - 1)))
    if ncomp < 2:
        # 극소표본/피처일 때는 LSA 생략하고 TF-IDF 행벡터를 정규화하여 사용
        Z = l2norm(X.toarray(), axis=1)
        return Z[: len(text_prev)], Z[len(text_prev):]

    Z = TruncatedSVD(n_components=ncomp, random_state=42).fit_transform(X)
    Z = l2norm(Z, axis=1)
    return Z[: len(text_prev)], Z[len(text_prev):]

def _centroids(Z: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.unique(labels)
    C, meta = [], []
    for cid in ids:
        mask = (labels == cid)
        if mask.sum() > 0:
            C.append(Z[mask].mean(axis=0))
            meta.append(cid)
    return np.vstack(C), np.array(meta)

def _match_years(
    y1: int, y2: int, k: int,
    df1: pd.DataFrame, df2: pd.DataFrame,
    text_cols: Iterable[str],
    label_col: str,
    thr: float = 0.08
) -> pd.DataFrame:
    # 텍스트 결합(연도 파일의 실제 컬럼 기준으로 유연하게)
    cols1 = _pick_text_cols(df1, text_cols)
    cols2 = _pick_text_cols(df2, text_cols)
    t1 = _join_text(df1, cols1)
    t2 = _join_text(df2, cols2)

    Z1, Z2 = _build_joint_lsa(t1, t2)
    C1, ids1 = _centroids(Z1, df1[label_col].to_numpy())
    C2, ids2 = _centroids(Z2, df2[label_col].to_numpy())

    S = cosine_similarity(C1, C2)
    j = S.argmax(axis=1)
    sims = S[np.arange(len(ids1)), j]
    matched = sims >= thr
    edges = pd.DataFrame({
        "year_from": y1, "year_to": y2,
        "prev_id": ids1,
        "next_id": np.where(matched, ids2[j], None),
        "similarity": np.where(matched, sims, np.nan),
        "matched": matched
    })
    return edges

def _counts_df_to_cid(counts_df: pd.DataFrame) -> pd.DataFrame:
    return counts_df.rename(columns={"cluster_id": "cid"})

# -----------------------------
# Public API
# -----------------------------
def run_clustering(
    df_embed: pd.DataFrame,
    n_clusters: int = DEFAULT_K,
    year_col: str = DEFAULT_YEAR_COL,
    embed_col: str = DEFAULT_EMBED_COL,
    text_cols: Iterable[str] | None = DEFAULT_TEXT_COLS,
    output_root: Path | str = ROOT,
    match_threshold: float = 0.08,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    반환: (df_clustered_all, summary_dict)
    - df_clustered_all: 각 연도별 라벨이 추가된 전체 DF
    - summary_dict: {"keywords":[...], "titles":[...], "paths": {...}, "artifacts": {...}}
    """
    # 디스크 경로는 참고 정보만 제공(실제 파일 저장 X)
    output_root = Path(output_root).resolve()
    yearly_dir = output_root / "yearly_v4"
    compare_dir = output_root / f"compare_methods_k{n_clusters}"

    label_col = f"cluster_k{n_clusters}"

    # ---- 1) 연도별 클러스터링 (메모리) ----
    years = sorted(pd.to_numeric(df_embed[year_col], errors="coerce").dropna().astype(int).unique().tolist())
    clustered_parts: List[pd.DataFrame] = []
    counts_by_year: Dict[int, pd.DataFrame] = {}
    tfidf_by_year: Dict[int, pd.DataFrame] = {}

    for y in years:
        df_y = df_embed[df_embed[year_col] == y].copy()
        if df_y.empty:
            continue
        df_y_labeled, counts_df, tfidf_df = _cluster_one_year(
            df_y, y, n_clusters, embed_col, text_cols
        )
        clustered_parts.append(df_y_labeled)
        counts_by_year[int(y)] = counts_df
        tfidf_by_year[int(y)] = tfidf_df

    if not clustered_parts:
        return df_embed.copy(), {"keywords": [], "titles": [], "paths": {}, "artifacts": {}}

    df_all_clustered = pd.concat(clustered_parts, ignore_index=True)

    # ---- 2) 전이 매칭 (메모리) ----
    edges_all: List[pd.DataFrame] = []
    for i in range(len(years) - 1):
        edges = _match_years(
            years[i], years[i + 1], n_clusters,
            df_all_clustered[df_all_clustered[year_col] == years[i]],
            df_all_clustered[df_all_clustered[year_col] == years[i + 1]],
            text_cols or (),
            label_col,
            thr=match_threshold
        )
        edges_all.append(edges)

    # ---- 3) Flow Year Metrics (메모리) ----
    rows = []
    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        links = edges_all[i]
        c1 = _counts_df_to_cid(counts_by_year[y1])
        c2 = _counts_df_to_cid(counts_by_year[y2])
        links_m = links[links["matched"] == True]  # noqa: E712
        for _, r in links_m.iterrows():
            d1 = int(c1.loc[c1["cid"] == r["prev_id"], "count"].values[0])
            d2 = int(c2.loc[c2["cid"] == r["next_id"], "count"].values[0])
            yoy = np.nan if d1 == 0 else (d2 - d1) / d1
            rows.append({
                "year_from": y1, "year_to": y2,
                "prev_id": int(r["prev_id"]), "next_id": int(r["next_id"]),
                "similarity": float(r["similarity"]),
                "docs_from": d1, "docs_to": d2, "yoy_docs": yoy
            })
    flow_df = pd.DataFrame(rows)

    # ---- summary 산출 ----
    # 키워드: 모든 연도 TF-IDF 상위 용어 점수 합 상위 100개
    kw_counter: Dict[str, float] = {}
    for tdf in tfidf_by_year.values():
        if not {"term","tfidf"} <= set(tdf.columns):
            continue
        for _, rr in tdf.iterrows():
            term = str(rr["term"])
            score = float(rr.get("tfidf", 0.0))
            if term:  # 빈 문자열 방지
                kw_counter[term] = kw_counter.get(term, 0.0) + score
    keywords = [t for t, _ in sorted(kw_counter.items(), key=lambda x: x[1], reverse=True)[:100]]

    # 대표 타이틀: 연도×클러스터 샘플 1~2개
    titles: List[str] = []
    for (y, cid), g in df_all_clustered.groupby([year_col, label_col]):
        cols = _pick_text_cols(g, text_cols)
        if not cols:
            continue
        sample = g[cols].head(2).astype(str).values.ravel().tolist()
        titles.extend(sample)
        if len(titles) >= 120:
            break

    summary = {
        "keywords": keywords,
        "titles": titles[:120],
        "paths": {"yearly_dir": str(yearly_dir), "compare_dir": str(compare_dir), "label_col": label_col},
        "artifacts": {
            "years": years,
            "label_col": label_col,
            "clustered_by_year": {int(y): df_all_clustered[df_all_clustered[year_col] == y].copy() for y in years},
            "counts_by_year": counts_by_year,
            "tfidf_by_year": tfidf_by_year,
            "edges": edges_all,
            "flow_edges_df": flow_df.copy()
        }
    }

    del clustered_parts
    gc.collect()

    return df_all_clustered, summary
