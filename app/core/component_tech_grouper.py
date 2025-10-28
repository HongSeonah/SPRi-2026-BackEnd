from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ComponentTechConfig:
    n_components: int = 100                 # 최종 구성기술 개수
    year_col: str = "year"
    embed_col: str = "embedding"
    cluster_col: str = "cluster_k100"       # run_clustering에서 사용한 label_col
    random_state: int = 42
    pca_max_components: int = 64            # 전역 차원 축소 상한
    centroid_weight_power: float = 0.75     # (선택) 문서수 가중치의 완만한 스케일링
    min_cluster_docs: int = 1               # 원 클러스터 최소 문서 수(필터용)


def _as_vec(v) -> np.ndarray | None:
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.asarray(v, dtype=np.float32)
    if isinstance(v, str):
        s = v.strip()
        try:
            return np.asarray(json.loads(s), dtype=np.float32)
        except Exception:
            try:
                import ast
                return np.asarray(ast.literal_eval(s), dtype=np.float32)
            except Exception:
                return None
    return None


def _embedding_matrix(series: pd.Series) -> np.ndarray:
    arrs: List[np.ndarray] = []
    for v in series.to_numpy():
        vec = _as_vec(v)
        if vec is None:
            raise ValueError("embedding 컬럼에 변환 불가 항목이 있습니다.")
        arrs.append(vec)
    X = np.vstack(arrs)
    return X


def _compute_centroids(df: pd.DataFrame, cfg: ComponentTechConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    """연도×원클러스터 단위로 임베딩 센트로이드 계산"""
    # 문서 임베딩 행렬
    X = _embedding_matrix(df[cfg.embed_col])
    X = normalize(X, axis=1)

    gids = df.groupby([cfg.year_col, cfg.cluster_col]).indices
    rows, C = [], []

    for (y, cid), idx in gids.items():
        if len(idx) < cfg.min_cluster_docs:
            continue
        centroid = X[idx].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        rows.append({"year": int(y), "orig_cluster_id": int(cid), "docs": int(len(idx))})
        C.append(centroid)

    meta_df = pd.DataFrame(rows)
    if meta_df.empty:
        raise RuntimeError("유효한 (year, cluster) 그룹이 없습니다.")

    C = np.vstack(C).astype(np.float32)
    return meta_df, C


def _reduce_dim(C: np.ndarray, max_components: int, random_state: int) -> np.ndarray:
    n, d = C.shape
    if n <= 1:
        return C
    ncomp = max(1, min(max_components, d, n - 1))
    if ncomp < 2:
        return C
    Z = PCA(n_components=ncomp, svd_solver="full", random_state=random_state).fit_transform(C)
    Z = normalize(Z, axis=1)
    return Z


def _global_grouping(C: np.ndarray, k: int, docs: np.ndarray, random_state: int) -> np.ndarray:
    """
    전역 (year,orig_cluster) 센트로이드를 k개 구성기술로 묶는다.
    문서 수 가중치를 미니배치 KMeans에 반영(샘플 반복)하지 않고,
    초기화 안정화를 위해 n_init='auto' 사용.
    """
    # (선택) 문서수 기반 가중치 -> 벡터 스케일링 (cosine에는 영향 없지만 KMeans 거리엔 영향)
    # 너무 큰 왜곡을 피하려고 docs**power 를 스케일로 사용
    w = np.maximum(1.0, np.asarray(docs, dtype=np.float32)) ** 0.0  # 기본: 영향 끔
    Xw = C * w[:, None]

    k_eff = max(1, min(k, Xw.shape[0]))
    mbk = MiniBatchKMeans(n_clusters=k_eff, random_state=random_state, n_init="auto")
    labels = mbk.fit_predict(Xw)
    return labels


def run_component_grouping(
    df_clustered: pd.DataFrame,
    cfg: ComponentTechConfig = ComponentTechConfig(),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    구성기술로 묶기:
      1) 문서 임베딩 → (year, orig_cluster) 센트로이드
      2) 전역 차원 축소(PCA)
      3) 전역 KMeans → 구성기술 ID(0..k-1)
      4) 문서 레벨 매핑 생성(component_tech_id를 각 문서에 부여)

    Returns
    -------
    (df_result, summary)
      df_result: 원본 df에 'component_tech_id' 열 추가
      summary  : 구성기술 메타 및 군집 품질 간단 통계
    """
    if cfg.year_col not in df_clustered.columns:
        raise KeyError(f"'{cfg.year_col}' 컬럼이 없습니다.")
    if cfg.embed_col not in df_clustered.columns:
        raise KeyError(f"'{cfg.embed_col}' 컬럼이 없습니다.")
    if cfg.cluster_col not in df_clustered.columns:
        raise KeyError(f"'{cfg.cluster_col}' 컬럼이 없습니다.")

    # 1) (year, orig_cluster) 센트로이드
    meta_df, C = _compute_centroids(df_clustered, cfg)

    # 2) 차원 축소
    Z = _reduce_dim(C, cfg.pca_max_components, cfg.random_state)

    # 3) 전역 KMeans → 구성기술
    comp_labels = _global_grouping(Z, cfg.n_components, meta_df["docs"].to_numpy(), cfg.random_state)
    meta_df["component_tech_id"] = comp_labels.astype(int)

    # 4) 문서 레벨로 조인
    out = df_clustered[[cfg.year_col, cfg.cluster_col]].copy()
    out = out.rename(columns={cfg.cluster_col: "orig_cluster_id"})
    out = out.join(df_clustered.drop(columns=[cfg.year_col, cfg.cluster_col]), how="left")

    key_cols = ["year", "orig_cluster_id"]
    out = out.merge(meta_df[key_cols + ["component_tech_id"]], on=key_cols, how="left")

    # 5) 요약(간단 품질 지표)
    #   - 구성기술별 (연도수, 원클러스터수, 문서수) / 센트로이드 간 평균 코사인
    comp_stats = (
        meta_df.groupby("component_tech_id")
        .agg(n_years=("year", "nunique"), n_orig_clusters=("orig_cluster_id", "nunique"), docs=("docs", "sum"))
        .reset_index()
        .sort_values("docs", ascending=False)
    )

    # 내부 밀도(centroid간 평균 유사도) 계산
    sims_rows = []
    for cid, g in meta_df.groupby("component_tech_id"):
        idx = g.index.to_numpy()
        if len(idx) >= 2:
            S = cosine_similarity(Z[idx], Z[idx])
            iu = np.triu_indices_from(S, k=1)
            dens = float(S[iu].mean()) if iu[0].size > 0 else 1.0
        else:
            dens = 1.0
        sims_rows.append({"component_tech_id": int(cid), "mean_intra_cosine": dens})
    density_df = pd.DataFrame(sims_rows)

    summary = {
        "n_components": int(cfg.n_components),
        "comp_stats": comp_stats,
        "density": density_df,
        "mapping_rows": int(len(out)),
        "meta_rows": int(len(meta_df)),
    }
    return out, summary


def save_component_file(
    df_with_component: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """결과 파일 저장(parquet 권장)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in (".parquet", ".pq"):
        df_with_component.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        df_with_component.to_csv(output_path, index=False)
    else:
        # 기본 parquet
        output_path = output_path.with_suffix(".parquet")
        df_with_component.to_parquet(output_path, index=False)
    return output_path
