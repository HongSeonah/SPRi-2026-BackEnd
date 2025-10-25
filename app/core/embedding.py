import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def _clean_text(text: str) -> str:
    text = str(text).lower()
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
    return " ".join([w for w in words if w not in STOPWORDS])

def run_embedding(df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> pd.DataFrame:
    """
    DataFrame을 받아 임베딩 컬럼을 추가한 DataFrame을 반환.
    - 입력: df에 최소한 title 또는 abstract 가 존재하면 됨
    - 출력: df['embedding'] = list[float] 컬럼 추가
    """
    # 제목/초록 중 하나를 임베딩 텍스트로 사용
    if "title" in df.columns:
        texts = df["title"].fillna("").astype(str)
    elif "abstract" in df.columns:
        texts = df["abstract"].fillna("").astype(str)
    else:
        # 둘 다 없으면 빈 문자열로 채워서라도 진행
        texts = pd.Series([""] * len(df))

    # 간단 전처리(불용어 제거) — 원 데이터 값은 안 건듦
    cleaned = texts.apply(_clean_text).tolist()

    model = SentenceTransformer(model_name)
    vectors = model.encode(cleaned, show_progress_bar=False, batch_size=64)

    out = df.copy()
    out["embedding"] = [vec.tolist() for vec in vectors]
    return out
