import os, io, re, json, math
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np
import pandas as pd

from fastapi import UploadFile
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_extraction.text import TfidfVectorizer

from app.utils.io_utils import tmp_path

# ─────────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────────

def _parse_embedding_cell(x) -> np.ndarray | None:
    """
    한 셀의 임베딩 표현을 robust하게 numpy array로 파싱.
    - "[0.1, 0.2, ...]" (JSON) → OK
    - "0.1, 0.2, ..." / "0.1 0.2 ..." → OK
    """
    if isinstance(x, (list, np.ndarray)): return np.array(x, dtype=float)
    if not isinstance(x, str): return None
    s = x.strip()
    try:
        if s.startswith("[") and s.endswith("]"):
            return np.array(json.loads(s), dtype=float)
    except Exception:
        pass
    s = s.replace(",", " ")
    toks = [t for t in s.split() if t]
    try:
        return np.array([float(t) for t in toks], dtype=float)
    except Exception:
        return None

def _load_dataframe_from_upload(upload: UploadFile) -> pd.DataFrame:
    """CSV 또는 Parquet 자동 감지 로더 (기본 CSV)."""
    name = upload.filename or "data.csv"
    suffix = os.path.splitext(name)[1].lower()
    raw = upload.file.read()
    upload.file.seek(0)
    if suffix in (".parquet", ".pq"):
        with io.BytesIO(raw) as bio:
            return pd.read_parquet(bio)
    # CSV로 처리(인코딩 추정)
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            continue
    # 최후: 판다스 기본
    return pd.read_csv(io.BytesIO(raw))

def _safe_metrics(X: np.ndarray, labels: np.ndarray, sample_size: int = 10000) -> Dict[str, float]:
    """실루엣/DB/CH 계산 (클래스 2개 미만이면 NaN)."""
    if len(np.unique(labels)) < 2:
        return dict(silhouette=np.nan, db=np.nan, ch=np.nan)
    n = X.shape[0]
    if n > sample_size:
        idx = np.random.RandomState(42).choice(n, size=sample_size, replace=False)
        Xs, ys = X[idx], labels[idx]
    else:
        Xs, ys = X, labels
    try: sil = float(silhouette_score(Xs, ys, metric="euclidean"))
    except Exception: sil = np.nan
    try: db  = float(davies_bouldin_score(Xs, ys))
    except Exception: db  = np.nan
    try: ch  = float(calinski_harabasz_score(Xs, ys))
    except Exception: ch  = np.nan
    return dict(silhouette=sil, db=db, ch=ch)

def _tokenize_simple(text: str) -> List[str]:
    """한/영/숫자 토큰, 길이≥4"""
    text = (text or "").lower()
    toks = re.findall(r"[가-힣A-Za-z0-9]+", text)
    return [t for t in toks if len(t) >= 4]

# ─────────────────────────────────────────────────────────────────
# 핵심 파이프라인
# ─────────────────────────────────────────────────────────────────

async def run_clustering_pipeline(
    data_csv: UploadFile,
    embed_col: str,
    year_col: str,
    text_cols: List[str],
    pca_dim: int,
    k_list: List[int],
    final_k: int | None,
    do_tfidf: bool,
    min_docs_per_cluster: int,
    similarity_top1: bool,
) -> Tuple[str, Dict]:
    """
    입력 CSV(임베딩/연도/텍스트 포함) → 연도별 KMeans → 연도간 링크 → 흐름/연도 지표.
    결과물은 out_dir 하위에 CSV들로 저장.
    """

    # 0) 로드 & 임베딩 파싱
    df = _load_dataframe_from_upload(data_csv)
    if embed_col not in df.columns:
        raise ValueError(f"'{embed_col}' 컬럼이 없습니다.")
    if year_col not in df.columns:
        raise ValueError(f"'{year_col}' 컬럼이 없습니다. (흐름 추적에 필수)")
    # 임베딩 파싱
    emb = df[embed_col].apply(_parse_embedding_cell)
    ok_mask = emb.apply(lambda v: isinstance(v, np.ndarray))
    if not ok_mask.any():
        raise ValueError("임베딩 파싱 실패: 형식을 확인하세요. 예) \"[0.1, 0.2, ...]\"")
    df = df[ok_mask].reset_index(drop=True)
    X = np.vstack(df[embed_col].apply(_parse_embedding_cell).to_numpy())

    # 출력 디렉토리
    out_dir = os.path.abspath(os.path.join(os.path.dirname(tmp_path()), "cluster_out_api"))
    os.makedirs(out_dir, exist_ok=True)

    # 1) 정규화 → PCA (전체 데이터 공통 기준)
    Xn = normalize(X, norm="l2", axis=1)
    pca = PCA(n_components=pca_dim, random_state=42)
    Xr = pca.fit_transform(Xn)

    # 2) 연도별 KMeans
    years = sorted(df[year_col].dropna().astype(int).unique())
    df["_rowid_"] = np.arange(len(df))
    best_per_year: Dict[int, Dict] = {}
    clustered_paths: List[str] = []
    counts_paths: List[str] = []
    reports: List[str] = []

    for y in years:
        idx = df.index[df[year_col].astype(int) == y].to_numpy()
        if len(idx) < max(min(k_list), 2):
            continue
        Xy = Xr[idx]

        # (a) K 스윕 (final_k 없을 때만)
        if final_k is None:
            rows = []
            for k in k_list:
                try:
                    mbk = MiniBatchKMeans(
                        n_clusters=int(k), batch_size=4096, max_iter=100, n_init=5, random_state=42
                    )
                    labels = mbk.fit_predict(Xy)
                    mets = _safe_metrics(Xy, labels)
                    rows.append({"year": y, "k": k, **mets})
                except Exception:
                    rows.append({"year": y, "k": k, "silhouette": np.nan, "db": np.nan, "ch": np.nan})
            rep = pd.DataFrame(rows).sort_values(["silhouette", "ch"], ascending=[False, False])
            rep_path = os.path.join(out_dir, f"kmeans_report_{y}.csv")
            rep.to_csv(rep_path, index=False)
            reports.append(rep_path)
            # best-k 선택
            best = rep.iloc[0]
            k_star = int(best["k"]) if not np.isnan(best["silhouette"]) else int(k_list[0])
        else:
            k_star = int(final_k)

        # (b) 최종 라벨링
        mbk = MiniBatchKMeans(
            n_clusters=k_star, batch_size=4096, max_iter=100, n_init=5, random_state=42
        )
        labels = mbk.fit_predict(Xy)
        df.loc[idx, f"cluster_k{k_star}"] = labels
        # centroid 저장
        centroids = mbk.cluster_centers_  # in PCA space
        best_per_year[y] = dict(k=k_star, centroids=centroids)

        # (c) 분포/라벨 저장
        counts = pd.Series(labels).value_counts().sort_index()
        counts_df = counts.rename_axis("cluster_id").reset_index(name="count")
        c_path = os.path.join(out_dir, f"cluster_counts_{y}.csv")
        counts_df.to_csv(c_path, index=False); counts_paths.append(c_path)

        out_y = df.loc[idx, [year_col, f"cluster_k{k_star}"] + [c for c in df.columns if c not in [f"cluster_k{k_star}"]]]
        out_path = os.path.join(out_dir, f"{y}_clustered_k{k_star}.csv")
        out_y.to_csv(out_path, index=False); clustered_paths.append(out_path)

        # (d) TF-IDF 요약(옵션)
        if do_tfidf and text_cols:
            # 텍스트 결합
            def _join_text(row):
                parts = []
                for c in text_cols:
                    if c in row and isinstance(row[c], str):
                        parts.append(row[c])
                    elif c in row and pd.notna(row[c]):
                        parts.append(str(row[c]))
                return " ".join(parts)
            sub = df.loc[idx, text_cols].copy()
            sub = sub.fillna("")
            joined = sub.apply(_join_text, axis=1)

            vec = TfidfVectorizer(
                tokenizer=_tokenize_simple, ngram_range=(1, 2),
                min_df=3, max_df=0.9, max_features=120000
            )
            Xtf = vec.fit_transform(joined.astype(str))
            vocab = np.array(vec.get_feature_names_out())

            # 클러스터별 상위어
            lab = labels
            rows = []
            for cid in sorted(np.unique(lab)):
                g_idx = np.where(lab == cid)[0]
                if len(g_idx) < max(1, min_docs_per_cluster):  # 너무 작은 것은 스킵
                    continue
                v = Xtf[g_idx].mean(axis=0)
                v = np.asarray(v).ravel()
                top = np.argsort(v)[::-1][:30]
                for t, s in zip(vocab[top], v[top]):
                    rows.append({"year": y, "cluster_id": int(cid), "term": t, "tfidf": float(s), "docs": int(len(g_idx))})
            tfdf = pd.DataFrame(rows).sort_values(["year", "cluster_id", "tfidf"], ascending=[True, True, False])
            tf_path = os.path.join(out_dir, f"tfidf_top_terms_{y}.csv")
            tfdf.to_csv(tf_path, index=False)

            # 요약표: 상위 10개 키워드
            summ_rows = []
            for cid, g in tfdf.groupby(["year", "cluster_id"]):
                terms = g.sort_values("tfidf", ascending=False)["term"].head(10).tolist()
                summ_rows.append({"year": cid[0], "cluster_id": cid[1], "top_terms": ", ".join(terms), "docs": int(g["docs"].max())})
            summ = pd.DataFrame(summ_rows).sort_values(["year", "docs"], ascending=[True, False])
            summ_path = os.path.join(out_dir, f"cluster_summary_{y}.csv")
            summ.to_csv(summ_path, index=False)

    if not best_per_year:
        raise ValueError("유효한 연도별 데이터가 없습니다.")

    # 3) 연도간 링크 구성 (centroid 코사인 유사도 Top-1 매칭)
    # 모든 centroid는 PCA 공간(L2 정규화 되진 않았지만 비교는 내적/코사인으로 충분)
    link_rows = []
    y_sorted = sorted(best_per_year.keys())
    for y1, y2 in zip(y_sorted[:-1], y_sorted[1:]):
        C1 = best_per_year[y1]["centroids"]  # [k1, d]
        C2 = best_per_year[y2]["centroids"]  # [k2, d]
        # L2 normalize → cosine = dot
        C1n = normalize(C1, norm="l2", axis=1)
        C2n = normalize(C2, norm="l2", axis=1)
        sims = C1n @ C2n.T                     # [k1, k2]
        # prev→next top1
        next_idx = sims.argmax(axis=1)         # [k1]
        next_sim = sims.max(axis=1)            # [k1]
        used_next = set()
        for cid_prev, (cid_next, s) in enumerate(zip(next_idx, next_sim)):
            if similarity_top1:
                # next 중복 허용(merge) → OK
                pass
            # 기록
            link_rows.append({
                "year_from": y1, "prev_id": int(cid_prev),
                "year_to": y2, "next_id": int(cid_next),
                "similarity": float(s)
            })

    links = pd.DataFrame(link_rows)
    links_path = os.path.join(out_dir, "links.csv")
    links.to_csv(links_path, index=False)

    # 4) 흐름(flow) 구성 및 연도별 지표
    # 문서수 맵
    docs_map = {}
    for y in years:
        p = os.path.join(out_dir, f"cluster_counts_{y}.csv")
        if os.path.exists(p):
            cdf = pd.read_csv(p)
            for _, r in cdf.iterrows():
                docs_map[(int(y), int(r["cluster_id"]))] = int(r["count"])

    # in-degree / inflow
    indeg = (links.groupby(["year_to","next_id"])
                   .agg(in_degree=("prev_id","nunique"))
                   .reset_index())
    inflow = (links.merge(
                pd.DataFrame([{"year_from": y, "prev_id": cid, "prev_docs": docs_map.get((y, cid), 0)}
                              for (y, cid) in docs_map.keys()]),
                on=["year_from","prev_id"], how="left")
                   .groupby(["year_to","next_id"])["prev_docs"].sum()
                   .reset_index(name="inflow_docs"))

    indeg = indeg.merge(inflow, on=["year_to","next_id"], how="left").fillna({"inflow_docs": 0})

    # prev→next 매핑 테이블
    m_next = {(int(r.year_from), int(r.prev_id)): (int(r.year_to), int(r.next_id))
              for _, r in links.iterrows()}
    # incoming set (year_to,next_id)
    incoming = {(int(r.year_to), int(r.next_id)) for _, r in links.iterrows()}
    # 모든 노드(연도/클러스터 id)
    nodes = {(y, int(cid)) for (y, cid) in docs_map.keys()}

    # birth = 해당 해에 incoming이 없는 노드
    incoming_same_year = {(y, cid) for (y, cid) in incoming}
    birth_nodes = {n for n in nodes if n not in incoming_same_year}

    # 흐름 체인 구성
    flows = []
    seen = set()
    for b in sorted(birth_nodes):
        if b in seen: continue
        chain = [b]; cur = b; seen.add(b)
        while cur in m_next:
            nxt = m_next[cur]
            if nxt in seen: break
            chain.append(nxt); seen.add(nxt); cur = nxt
        if chain:
            flows.append(chain)

    def flow_id_of(chain):  # 시작년/클러스터로 flow_id 생성
        y0, c0 = chain[0]
        return f"F{y0}_{c0}"

    flow_map = {flow_id_of(ch): ch for ch in flows}

    # 연도별 지표 생성
    rows = []
    for fid, chain in flow_map.items():
        for i, (y, cid) in enumerate(chain):
            docs = docs_map.get((y, cid), 0)
            # prev→this 유사도
            if i > 0:
                y_prev, cid_prev = chain[i-1]
                s_prev = float(links[(links["year_from"]==y_prev) &
                                     (links["prev_id"]==cid_prev) &
                                     (links["year_to"]==y) &
                                     (links["next_id"]==cid)]["similarity"].max())
                docs_prev = docs_map.get((y_prev, cid_prev), 0)
            else:
                s_prev = np.nan
                docs_prev = np.nan

            # in_degree / inflow_docs (이번해에 이 노드로 들어온 것들)
            row_in = indeg[(indeg["year_to"]==y) & (indeg["next_id"]==cid)]
            in_deg = int(row_in["in_degree"].iloc[0]) if not row_in.empty else 0
            inflow_docs = int(row_in["inflow_docs"].iloc[0]) if not row_in.empty else 0

            yoy = (docs - docs_prev)/docs_prev if (isinstance(docs_prev, (int, float)) and docs_prev > 0) else np.nan

            rows.append({
                "flow_id": fid, "year": int(y), "cluster_id": int(cid),
                "pos_in_flow": i, "is_birth": (i==0), "is_death": (i==len(chain)-1),
                "docs": int(docs), "docs_prev": docs_prev, "yoy_docs": yoy,
                "sim_prev": s_prev, "in_degree": in_deg, "inflow_docs": inflow_docs
            })

    per_year = pd.DataFrame(rows).sort_values(["flow_id","year"]).reset_index(drop=True)

    # 연도별 스코어 (0..100)
    def _minmax(s: pd.Series) -> pd.Series:
        v = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
        mn, mx = v.min(skipna=True), v.max(skipna=True)
        if not pd.notna(mn) or not pd.notna(mx) or mx <= mn:
            return pd.Series(50.0, index=s.index)
        return (v - mn) / (mx - mn) * 100.0

    per_year["growth_score"]    = per_year.groupby("year")["yoy_docs"].transform(lambda s: _minmax(s.fillna(0.0)))
    per_year["scale_score"]     = per_year.groupby("year")["docs"].transform(_minmax)
    per_year["merge_score"]     = per_year.groupby("year")["in_degree"].transform(_minmax)
    per_year["stability_score"] = per_year.groupby("year")["sim_prev"].transform(lambda s: _minmax(s.fillna(0.0)))

    per_year["potential_0_100"] = (
        0.45*per_year["growth_score"] +
        0.35*per_year["scale_score"] +
        0.15*per_year["merge_score"] +
        0.05*per_year["stability_score"]
    ).round(2)

    metrics_path = os.path.join(out_dir, "flow_year_metrics.csv")
    per_year.to_csv(metrics_path, index=False)

    # 요약 리턴
    summary = {
        "years": years,
        "reports": reports,
        "clustered_csv_paths": clustered_paths,
        "counts_csv_paths": counts_paths,
        "links_csv": links_path,
        "flow_metrics_csv": metrics_path,
        "n_flows": int(per_year["flow_id"].nunique()) if not per_year.empty else 0,
        "n_rows_metrics": int(len(per_year)),
    }
    return out_dir, summary
