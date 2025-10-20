import io, re, json
import pandas as pd
import numpy as np
from typing import Tuple
from fastapi import UploadFile
from sentence_transformers import SentenceTransformer
from app.utils.io_utils import tmp_path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SK_STOP

def _extract_year(s: str) -> int | None:
    m = re.search(r"(\d{4})", str(s))
    if not m: return None
    try:
        return int(m.group(1))
    except:
        return None

def _clean_title(s: str) -> str:
    # 영문자 3+만 남기고 scikit-learn 불용어 제거
    tokens = re.findall(r"[A-Za-z]{3,}", str(s).lower())
    return " ".join([t for t in tokens if t not in SK_STOP])

async def run_embedding(
    input_csv: UploadFile,
    title_col: str,
    date_col: str,
    model_name: str,
    out_parquet_name: str,
) -> Tuple[str, int, int | None, int | None]:
    # 1) 로드
    content = await input_csv.read()
    df = pd.read_csv(io.BytesIO(content))
    if title_col not in df.columns:
        raise ValueError(f"'{title_col}' 컬럼이 없습니다.")
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' 컬럼이 없습니다.")

    # 2) 연도
    df["year"] = df[date_col].astype(str).map(_extract_year)
    # NaN은 드랍하지 않고 남겨도 됨(후단에서 필터 가능)
    y_min = int(np.nanmin(df["year"])) if df["year"].notna().any() else None
    y_max = int(np.nanmax(df["year"])) if df["year"].notna().any() else None

    # 3) 전처리 + 임베딩
    df["cleaned_title"] = df[title_col].fillna("").map(_clean_title)
    model = SentenceTransformer(model_name)
    emb = model.encode(
        df["cleaned_title"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=64,
    )
    df["embedding"] = [e.tolist() for e in emb]

    # 4) 저장(parquet)
    out_path = tmp_path(".parquet", prefer_name=out_parquet_name)
    df_out = df[[title_col, "cleaned_title", "year", "embedding"]].rename(
        columns={title_col: "title"}
    )
    df_out.to_parquet(out_path, index=False)

    return out_path, len(df_out), y_min, y_max
