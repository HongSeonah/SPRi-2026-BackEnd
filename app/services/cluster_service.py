import io, re, os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any
from fastapi import UploadFile

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as norm2
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path("./cluster_out")
YEARLY_DIR = ROOT / "yearly_v4"

def _join_text(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1)

def _simple_tokenize(text: str):
    toks = re.findall(r"[가-힣A-Za-z0-9]+", (text or "").lower())
    return [t for t in toks if len(t) > 2]

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _centroids(Z: np.ndarray, labels: np.ndarray):
    ids = np.unique(labels)
    C, meta = [], []
    for cid in ids:
        m = (labels == cid)
        if m.sum() > 0:
            C.append(Z[m].mean(axis=0))
            meta.append(cid)
    return np.vstack(C), np.array(meta)

async def run_cluster_pipeline(
    df_all_parquet: UploadFile,
    k: int,
    pca_dim: int,
    min_df: int,
    max_df: float,
    lsa_dim: int,
    sim_thr: float,
    label_suffix: str,
    method_tag: str,
) -> Dict[str, Any]:
    # 0) 입력 로드
    buf = await df_all_parquet.read()
    df_all = pd.read_parquet(io.BytesIO(buf))
    if not {"title","cleaned_title","year","embedding"}.issubset(df_all.columns):
        raise ValueError("parquet에는 title, cleaned_title, year, embedding 컬럼이 필요합니다.")

    # 1) 연도별 클러스터링 + 저장
    _ensure_dir(YEARLY_DIR)
    years = sorted([int(y) for y in df_all["year"].dropna().unique()])
    label_col = f"cluster_{label_suffix}"
    for y in years:
        d = df_all[df_all["year"]==y].copy()
        out_dir = YEARLY_DIR / str(y)
        _ensure_dir(out_dir)

        X = np.vstack(d["embedding"].to_numpy())
        Xn = normalize(X, axis=1)
        Xr = PCA(n_components=pca_dim, random_state=42).fit_transform(Xn)

        mbk = MiniBatchKMeans(n_clusters=k, batch_size=4096, max_iter=100, n_init=5, random_state=42)
        labels = mbk.fit_predict(Xr)
        d[label_col] = labels

        # ① clustered
        f_clustered = out_dir / f"{y}_clustered_{label_suffix}.csv"
        d.to_csv(f_clustered, index=False)

        # ② counts
        counts = d[label_col].value_counts().sort_index()
        counts_df = counts.rename_axis("cluster_id").reset_index(name="count")
        counts_df["ratio"] = counts_df["count"] / counts_df["count"].sum()
        f_counts = out_dir / f"cluster_counts_{label_suffix}.csv"
        counts_df.to_csv(f_counts, index=False)

        # ③ tf-idf top terms
        vec = TfidfVectorizer(
            tokenizer=_simple_tokenize,
            ngram_range=(1,2),
            min_df=min_df, max_df=max_df, max_features=120000
        )
        texts = _join_text(d, ["title", "cleaned_title"])
        Xtf = vec.fit_transform(texts)
        vocab = np.array(vec.get_feature_names_out())
        rows=[]
        for cid, g in d.groupby(label_col):
            vecm = np.asarray(Xtf[g.index].mean(axis=0)).ravel()
            top = vecm.argsort()[::-1][:30]
            for t, sc in zip(vocab[top], vecm[top]):
                rows.append({"cluster_id": int(cid), "term": t, "tfidf": float(sc), "docs": len(g)})
        pd.DataFrame(rows).to_csv(out_dir / f"tfidf_top_terms_{label_suffix}.csv", index=False)

    # 2) 연속연도 전이 매칭(LSA 기반) + 저장
    COMPARE_DIR = ROOT / f"compare_methods_{label_suffix}"
    _ensure_dir(COMPARE_DIR)

    def _build_joint_lsa(text_prev: pd.Series, text_next: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=3, max_df=0.95)
        joint = pd.concat([text_prev, text_next])
        X = vec.fit_transform(joint)
        k_dim = max(2, min(lsa_dim, X.shape[1]-1))
        Z = TruncatedSVD(n_components=k_dim, random_state=42).fit_transform(X)
        Z = norm2(Z, axis=1)
        return Z[:len(text_prev)], Z[len(text_prev):]

    for i in range(len(years)-1):
        y1, y2 = years[i], years[i+1]
        f1 = YEARLY_DIR / str(y1) / f"{y1}_clustered_{label_suffix}.csv"
        f2 = YEARLY_DIR / str(y2) / f"{y2}_clustered_{label_suffix}.csv"
        df1, df2 = pd.read_csv(f1), pd.read_csv(f2)
        t1 = _join_text(df1, ["title","cleaned_title"])
        t2 = _join_text(df2, ["title","cleaned_title"])
        Z1, Z2 = _build_joint_lsa(t1, t2)

        C1, ids1 = _centroids(Z1, df1[label_col].to_numpy())
        C2, ids2 = _centroids(Z2, df2[label_col].to_numpy())
        S = cosine_similarity(C1, C2)
        j = S.argmax(axis=1)
        sims = S[np.arange(len(ids1)), j]
        matched = sims >= float(sim_thr)

        out_dir = COMPARE_DIR / f"{y1}_to_{y2}"
        _ensure_dir(out_dir)
        pd.DataFrame({
            "year_from": y1,
            "year_to": y2,
            "prev_id": ids1,
            "next_id": np.where(matched, ids2[j], None),
            "similarity": np.where(matched, sims, np.nan),
            "matched": matched
        }).to_csv(out_dir / f"{method_tag}_links.csv", index=False)

    # 3) 흐름 메트릭(flow_year_metrics) 집계
    rows=[]
    def _load_counts(y: int) -> pd.DataFrame:
        f = YEARLY_DIR / str(y) / f"cluster_counts_{label_suffix}.csv"
        return pd.read_csv(f).rename(columns={"cluster_id":"cid"})

    for i in range(len(years)-1):
        y1, y2 = years[i], years[i+1]
        links = pd.read_csv(COMPARE_DIR / f"{y1}_to_{y2}/{method_tag}_links.csv")
        c1 = _load_counts(y1); c2 = _load_counts(y2)
        for _, r in links[links["matched"]].iterrows():
            d1 = int(c1.loc[c1["cid"]==r["prev_id"], "count"].values[0])
            d2 = int(c2.loc[c2["cid"]==r["next_id"], "count"].values[0])
            yoy = (d2-d1)/d1 if d1>0 else np.nan
            rows.append({
                "year_from": y1, "year_to": y2,
                "prev_id": int(r["prev_id"]), "next_id": int(r["next_id"]),
                "similarity": float(r["similarity"]),
                "docs_from": d1, "docs_to": d2, "yoy_docs": yoy
            })
    flow_df = pd.DataFrame(rows)
    flow_out = COMPARE_DIR / f"flow_year_metrics_{method_tag}_{label_suffix}.csv"
    flow_df.to_csv(flow_out, index=False)

    return {
        "yearly_dir": str(YEARLY_DIR),
        "compare_dir": str(COMPARE_DIR),
        "flow_metrics_csv": str(flow_out),
        "k": int(k),
    }
