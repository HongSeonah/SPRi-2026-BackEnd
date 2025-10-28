# app/core/embedding.py
import re
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---- (선택) 간단 텍스트 전처리 ----
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = set()

_WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")

def _clean_text(text: str) -> str:
    text = str(text).lower()
    words = _WORD_RE.findall(text)
    if not STOPWORDS:
        return " ".join(words)
    return " ".join([w for w in words if w not in STOPWORDS])

def _pick_text_column(df: pd.DataFrame, candidates: Iterable[str]) -> Tuple[str, pd.Series]:
    for c in candidates:
        if c in df.columns:
            return c, df[c].fillna("").astype(str)
    col = "__concat_text__"
    ser = df.astype(str).agg(" | ".join, axis=1)
    return col, ser

def run_embedding(
    df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    *,
    batch_size: int = 512,
    text_candidates: Tuple[str, ...] = ("title", "abstract", "summary", "text", "description"),
    checkpoint_dir: Optional[Union[str, Path]] = None,  # 예: f"/tmp/emb_ckpt/{run_id}"
    resume: bool = True,
    progress_cb: Optional[Callable[[int, int], None]] = None,  # (processed, total)
) -> pd.DataFrame:
    """
    DataFrame -> df['embedding']=list[float]
    - 배치 인코딩 + 체크포인트 + 리줌 + 진행률 콜백 지원
    """
    text_col, texts = _pick_text_column(df, text_candidates)
    cleaned = texts.map(_clean_text)
    total = len(cleaned)
    if total == 0:
        out = df.copy()
        out["embedding"] = [[] for _ in range(len(out))]
        if progress_cb:
            progress_cb(0, 0)
        return out

    model = SentenceTransformer(model_name)

    ckpt_dir_path: Optional[Path] = None
    if checkpoint_dir:
        ckpt_dir_path = Path(checkpoint_dir)
        ckpt_dir_path.mkdir(parents=True, exist_ok=True)
        (ckpt_dir_path / "meta.json").write_text(
            json.dumps({"model_name": model_name, "text_col": text_col, "total": total, "batch_size": batch_size}, ensure_ascii=False)
        )

    start_idx = 0
    part_files = []
    if ckpt_dir_path and resume:
        part_files = sorted(ckpt_dir_path.glob("part_*.npy"))
        if part_files:
            try:
                start_idx = int(part_files[-1].stem.split("_")[-1])
            except Exception:
                start_idx = 0

    processed = start_idx
    if progress_cb:
        progress_cb(processed, total)

    i = start_idx
    while i < total:
        j = min(i + batch_size, total)
        batch_texts = cleaned.iloc[i:j].tolist()
        emb = model.encode(batch_texts, show_progress_bar=False, batch_size=batch_size)
        emb = np.asarray(emb)

        if ckpt_dir_path:
            part_path = ckpt_dir_path / f"part_{i:06d}_{j:06d}.npy"
            np.save(part_path, emb)

        part_files.append(ckpt_dir_path / f"part_{i:06d}_{j:06d}.npy" if ckpt_dir_path else None)
        i = j
        processed = j
        if progress_cb:
            try:
                progress_cb(processed, total)
            except Exception:
                pass  # 콜백 에러는 본 처리에 영향 주지 않음

    if ckpt_dir_path:
        files = sorted([p for p in part_files if p is not None])
        if not files:
            vectors = model.encode(cleaned.tolist(), show_progress_bar=False, batch_size=batch_size)
            vectors = np.asarray(vectors)
        else:
            dims, rows_total = None, 0
            for p in files:
                arr = np.load(p, mmap_mode="r")
                rows_total += arr.shape[0]
                dims = arr.shape[1] if dims is None else dims
            final_path = ckpt_dir_path / "embeddings.final.npy"
            mm = np.memmap(final_path, dtype="float32", mode="w+", shape=(rows_total, dims))
            offset = 0
            for p in files:
                arr = np.load(p, mmap_mode="r")
                r = arr.shape[0]
                mm[offset:offset+r, :] = arr[:]
                offset += r
            del mm
            vectors = np.load(final_path)
    else:
        chunks = []
        for i in range(0, total, batch_size):
            j = min(i + batch_size, total)
            batch_texts = cleaned.iloc[i:j].tolist()
            emb = model.encode(batch_texts, show_progress_bar=False, batch_size=batch_size)
            chunks.append(np.asarray(emb))
        vectors = np.vstack(chunks) if len(chunks) > 1 else chunks[0]

    out = df.copy()
    out["embedding"] = [row.astype(float).tolist() for row in vectors]
    return out
