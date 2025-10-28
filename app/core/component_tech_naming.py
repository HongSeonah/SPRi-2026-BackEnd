# app/core/component_tech_naming.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.clustering import _simple_tokenize as _tok, _pick_text_cols as _pick_cols
from app.core.tech_naming import _run_flowagg

def _join_text(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    return df[list(cols)].fillna("").astype(str).agg(" ".join, axis=1)

def _build_component_tfidf(df: pd.DataFrame, label_col: str, text_cols: Iterable[str] | None):
    cols = _pick_cols(df, text_cols)
    if not cols:
        return pd.DataFrame(columns=["cluster_id","term","tfidf","docs"])
    text = _join_text(df, cols)
    if text.str.len().fillna(0).sum() == 0:
        return pd.DataFrame(columns=["cluster_id","term","tfidf","docs"])
    n_docs = len(df)
    min_df_val = 1 if n_docs < 20 else 3
    max_df_val = 1.0 if n_docs <= 20 else 0.95
    def _fit(min_df_v, max_df_v):
        tf = TfidfVectorizer(tokenizer=_tok, token_pattern=None, ngram_range=(1,2),
                             min_df=min_df_v, max_df=max_df_v, max_features=120_000)
        return tf, tf.fit_transform(text)
    tf, X = _fit(min_df_val, max_df_val)
    if X.shape[1] == 0:
        tf, X = _fit(1, 1.0)
        if X.shape[1] == 0:
            return pd.DataFrame(columns=["cluster_id","term","tfidf","docs"])
    vocab = np.array(tf.get_feature_names_out())
    rows = []
    labels = df[label_col].to_numpy()
    for cid in np.unique(labels):
        m = (labels == cid)
        if not m.any(): continue
        vec = np.asarray(X[m].mean(axis=0)).ravel()
        if vec.size == 0: continue
        top = vec.argsort()[::-1][:30]
        for t, sc in zip(vocab[top], vec[top]):
            rows.append({"cluster_id": int(cid), "term": t, "tfidf": float(sc), "docs": int(m.sum())})
    return pd.DataFrame(rows)

def generate_component_names_csv(
    df_with_component: pd.DataFrame,
    *,
    label_col: str = "component_tech_id",
    text_cols: Iterable[str] | None = ("title",),
    output_csv_path: str | Path | None = None,
) -> str:
    """
    구성기술(= component_tech_id) '전체'에 대해 네이밍을 수행하고 CSV로 저장.
    반환: 출력 CSV 절대경로 (OUTPUT_DIR/component_tech_names.csv 기본)
    """
    # 1) 연도 0으로 통일
    df0 = df_with_component.copy()
    df0["year"] = 0

    # 2) TF-IDF(구성기술별) 구축
    tfidf_df = _build_component_tfidf(df0, label_col=label_col, text_cols=text_cols)

    # 3) panel (flow_id=component_tech_id, year=0)
    comp_counts = df0.groupby(label_col)["year"].size().rename("docs").reset_index()
    panel = comp_counts[[label_col, "docs"]].copy()
    panel["flow_id"] = panel[label_col].astype(int)
    panel["year"] = 0
    panel["cluster_id"] = panel[label_col].astype(int)
    panel = panel[["flow_id", "year", "cluster_id", "docs"]]

    # 4) artifacts (flow-agg 엔진 최소 요구)
    artifacts = {
        "label_col": label_col,
        "clustered_by_year": {0: df0},
        "tfidf_by_year": {0: tfidf_df},
    }

    # 5) 대상 flow 전부 (정렬은 보기 편하게 문서수 내림차순)
    target_flows = comp_counts.sort_values("docs", ascending=False)[label_col].astype(int).tolist()

    # 6) 네이밍 실행 → CSV 정규화 저장
    out_dir = Path(os.getenv("OUTPUT_DIR", "/var/lib/app/outputs")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(output_csv_path) if output_csv_path else (out_dir / "component_tech_names.csv")

    tmp_csv = _run_flowagg(panel, target_flows, artifacts)  # 기존 엔진이 만든 CSV
    df_names = pd.read_csv(tmp_csv)

    # flow_id -> component_tech_id 로 컬럼 리네임
    rename_map = {"flow_id": label_col}
    for k in list(rename_map.keys()):
        if k not in df_names.columns:
            rename_map.pop(k, None)
    df_out = df_names.rename(columns=rename_map)

    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return str(out_csv.resolve())
