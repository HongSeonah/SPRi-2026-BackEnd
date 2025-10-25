import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any, Optional

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# 불용어/토크나이저 (임베딩 전처리와 톤 맞춤)
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def _simple_tokenize(text: str) -> List[str]:
    toks = re.findall(r"[가-힣A-Za-z0-9]+", str(text or "").lower())
    return [t for t in toks if len(t) > 2 and t not in STOPWORDS]

def _top_terms_per_cluster(texts: List[str], labels: np.ndarray, topk: int = 10) -> Dict[int, List[str]]:
    """
    클러스터별 상위 키워드 추출 (TF-IDF)
    """
    if len(texts) == 0:
        return {}
    vec = TfidfVectorizer(tokenizer=_simple_tokenize, min_df=2, max_df=0.8)
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())

    top_terms: Dict[int, List[str]] = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            top_terms[int(c)] = []
            continue
        # 해당 클러스터 문서들 TF-IDF 평균
        mean_tfidf = np.asarray(X[idx].mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[::-1][:topk]
        top_terms[int(c)] = vocab[top_idx].tolist()
    return top_terms

def _rep_titles_per_cluster(titles: List[str], labels: np.ndarray, topn: int = 3) -> Dict[int, List[str]]:
    rep: Dict[int, List[str]] = {}
    s = pd.Series(titles)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            rep[int(c)] = []
            continue
        # 간단히 앞쪽 몇 개를 대표로 (원하면 길이/인기도 등 기준 정교화 가능)
        rep[int(c)] = s.iloc[idx].head(topn).tolist()
    return rep

def _cluster_one_block(
    df_block: pd.DataFrame,
    n_clusters: int,
    pca_components: int = 50,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[int, List[str]], Dict[int, List[str]]]:
    """
    하나의 블록(예: 특정 연도)의 임베딩을 클러스터링하고,
    클러스터별 키워드/대표제목을 리턴
    """
    X = np.vstack(df_block["embedding"].to_numpy())  # (N, D)
    Xn = normalize(X, axis=1)
    # 차원 축소 (D가 작으면 PCA 내부에서 자동 처리)
    Xr = PCA(n_components=min(pca_components, Xn.shape[1]), random_state=random_state).fit_transform(Xn)

    mbk = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
    labels = mbk.fit_predict(Xr)

    # 키워드/대표제목 추출용 텍스트 준비
    # title 없으면 abstract, 둘 다 없으면 빈 문자열
    if "title" in df_block.columns:
        texts = df_block["title"].fillna("").astype(str).tolist()
    elif "abstract" in df_block.columns:
        texts = df_block["abstract"].fillna("").astype(str).tolist()
    else:
        texts = [""] * len(df_block)

    top_terms = _top_terms_per_cluster(texts, labels, topk=10)
    rep_titles = _rep_titles_per_cluster(
        df_block["title"].fillna("").astype(str).tolist() if "title" in df_block.columns else texts,
        labels,
        topn=3,
    )
    return labels, top_terms, rep_titles

def run_clustering(df_embed: pd.DataFrame, n_clusters: int = 100):
    """
    df_embed: embedding(list[float]) 컬럼과 year 컬럼이 존재한다고 가정.
    연도별로 임베딩 정규화 → (필요할 때만) PCA → MiniBatchKMeans.
    소표본/저차원에서도 안전하게 동작하도록 가드 추가.
    """
    import os
    import numpy as np
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans

    out_dir = "./app/data/outputs"
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    years = sorted([y for y in df_embed["year"].dropna().unique().tolist() if y != -1])

    for y in years:
        df_y = df_embed[df_embed["year"] == y].copy()
        if df_y.empty:
            continue

        # --- X 구성 ---
        # embedding 컬럼이 list[float] 형태라고 가정
        X = np.vstack(df_y["embedding"].to_numpy())
        n_samples, n_features = X.shape

        # --- 클러스터 수 가드: 샘플 수보다 클 수 없음 ---
        k = min(n_clusters, max(1, n_samples))  # 최소 1
        if k > n_samples:
            k = n_samples

        # --- 정규화 ---
        Xn = normalize(X, axis=1)

        # --- PCA: 조건부 수행 (표본/특징 수가 여유 있을 때만) ---
        Xr = Xn
        if n_features > 50 and n_samples > 50:
            # PCA 차원 안전 가드
            safe_pca_dim = min(50, n_features - 1, n_samples - 1)
            if safe_pca_dim >= 2:  # 1차원 이하로 줄이는 건 이득이 적으므로 조건
                Xr = PCA(n_components=safe_pca_dim, random_state=42).fit_transform(Xn)

        # --- MiniBatchKMeans ---
        # k가 1이면 군집화 의미가 적지만, 에러 없이 라벨 0으로 배정됩니다.
        mbk = MiniBatchKMeans(n_clusters=k, random_state=42)
        labels = mbk.fit_predict(Xr)

        df_y[f"cluster_k{k}"] = labels

        out_csv = f"{out_dir}/{y}_clustered_k{k}.csv"
        df_y.to_csv(out_csv, index=False)
        results[y] = {"samples": n_samples, "features": n_features, "k": k, "pca_dim": Xr.shape[1]}

    return df_embed, {"years": results}
