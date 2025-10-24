from fastapi import APIRouter
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

router = APIRouter()

def clean_text(text: str):
    text = str(text).lower()
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
    return " ".join([w for w in words if w not in STOPWORDS])

def run_embedding(input_path: str, model_name="all-MiniLM-L6-v2"):
    df = pd.read_csv(input_path)
    df["cleaned_title"] = df["title"].fillna("").apply(clean_text)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(df["cleaned_title"].tolist(), show_progress_bar=True, batch_size=64)
    df["embedding"] = [e.tolist() for e in embeddings]

    out_path = f"./app/data/processed/embeddings_{model_name.replace('/', '_')}.parquet"
    df.to_parquet(out_path, index=False)
    return {
        "message": "✅ 임베딩 완료",
        "output_path": out_path,
        "rows": len(df)
    }
