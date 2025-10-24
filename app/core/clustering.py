from fastapi import APIRouter
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import re, nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

router = APIRouter()

def simple_tokenize(text):
    toks = re.findall(r"[가-힣A-Za-z0-9]+", (text or "").lower())
    return [t for t in toks if len(t) > 2 and t not in STOPWORDS]

def run_clustering(input_path: str, n_clusters=100):
    df = pd.read_parquet(input_path)
    output_dir = "./app/data/outputs/"
    import os; os.makedirs(output_dir, exist_ok=True)

    years = sorted(df["year"].unique())
    for y in years:
        df_y = df[df["year"] == y].copy()
        X = np.vstack(df_y["embedding"].to_numpy())
        Xn = normalize(X, axis=1)
        Xr = PCA(n_components=50, random_state=42).fit_transform(Xn)
        mbk = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        df_y[f"cluster_k{n_clusters}"] = mbk.fit_predict(Xr)

        out_csv = f"{output_dir}/{y}_clustered_k{n_clusters}.csv"
        df_y.to_csv(out_csv, index=False)

    return {
        "message": f"✅ 클러스터링 완료 ({len(years)}개 연도 처리됨)",
        "output_dir": output_dir
    }
