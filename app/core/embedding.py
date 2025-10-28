# app/core/embedding.py
from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union, Callable, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ============================================
# 간단 텍스트 전처리
# ============================================
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

# ============================================
# 체크포인트 안전 IO (레거시 .npy → npz 마이그레이션 포함)
# ============================================
def _to_f32_array(x: Any) -> np.ndarray:
    """
    다양한 입력(x)을 float32 2D ndarray로 변환:
    - ndarray(object 포함) 또는 list/tuple of vectors 모두 허용
    """
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            # 각 원소가 벡터라고 가정
            x = [np.asarray(v, dtype=np.float32) for v in x]
            return np.vstack(x).astype(np.float32, copy=False)
        # 이미 수치형 ndarray
        arr = np.asarray(x, dtype=np.float32, copy=False)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr
    # 파이썬 리스트/튜플 등
    arr = np.vstack([np.asarray(v, dtype=np.float32) for v in x]).astype(np.float32, copy=False)
    return arr

def _safe_load_part(path: Union[str, Path]) -> np.ndarray:
    """
    파트 파일(.npz/.npy)을 안전하게 로드하여 float32 2D 배열로 반환.
    - .npz: key 'emb' 고정
    - .npy: 우선 allow_pickle=False, 실패/객체면 allow_pickle=True → float32 2D로 정규화
    """
    p = Path(path)
    if p.suffix.lower() == ".npz":
        with np.load(str(p)) as data:
            if "emb" not in data:
                raise RuntimeError(f"invalid npz: {p} (missing 'emb')")
            arr = data["emb"]
            return _to_f32_array(arr)
    # 레거시 .npy
    try:
        arr = np.load(str(p), mmap_mode="r", allow_pickle=False)
    except Exception:
        arr = np.load(str(p), allow_pickle=True)
    return _to_f32_array(arr)

def _save_part_npz(path: Union[str, Path], emb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    emb = np.asarray(emb, dtype=np.float32)
    np.savez_compressed(str(path), emb=emb)

def _detect_last_end_index(ckpt_dir_path: Path) -> int:
    """
    part_{i:06d}_{j:06d}.npz / .npy 파일명에서 가장 큰 end 인덱스를 찾아 resume 지점으로 사용.
    npz 우선, 없으면 npy도 고려(레거시).
    """
    cand = sorted(ckpt_dir_path.glob("part_*.npz")) or sorted(ckpt_dir_path.glob("part_*.npy"))
    if not cand:
        return 0
    last = cand[-1].stem.split("_")  # ["part", "000000", "010000"]
    try:
        return int(last[-1])
    except Exception:
        return 0

def _list_parts(ckpt_dir_path: Path, resume: bool) -> list[Path]:
    """
    현재 체크포인트 디렉토리의 part 파일 목록을 반환.
    - resume=True면 npz 우선, 없으면 npy까지 포함(마이그레이션 대상)
    - resume=False면 빈 목록
    """
    if not resume:
        return []
    npz_parts = sorted(ckpt_dir_path.glob("part_*.npz"))
    if npz_parts:
        return npz_parts
    # npz가 없다면 레거시 npy까지 포함하여 마이그레이션 대상으로 취급
    return sorted(ckpt_dir_path.glob("part_*.npy"))

def _migrate_legacy_npy_to_npz(files: list[Path]) -> list[Path]:
    """
    레거시 .npy 파트 파일을 .npz로 변환하고, 변환 성공 시 원본 .npy 삭제.
    반환: 변환 완료 후 존재하는 .npz 파일 목록
    """
    out_files: list[Path] = []
    for p in files:
        if p.suffix.lower() == ".npz":
            out_files.append(p)
            continue
        # .npy → 로드/정규화 → .npz 저장 → 원본 삭제
        arr = _safe_load_part(p)  # float32 2D
        npz_path = p.with_suffix(".npz")
        _save_part_npz(npz_path, arr)
        try:
            p.unlink()
        except Exception:
            pass
        out_files.append(npz_path)
    # 정렬하여 반환
    return sorted(set(out_files))

# ============================================
# 메인: 임베딩 실행
# ============================================
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
    - 배치 인코딩 + 체크포인트(npz, float32) + 레거시 자동 마이그레이션 + 리줌 + 진행률 콜백
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

    # 모델 로드
    model = SentenceTransformer(model_name)

    # 체크포인트 준비
    ckpt_dir_path: Optional[Path] = None
    if checkpoint_dir:
        ckpt_dir_path = Path(checkpoint_dir)
        ckpt_dir_path.mkdir(parents=True, exist_ok=True)
        (ckpt_dir_path / "meta.json").write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "text_col": text_col,
                    "total": total,
                    "batch_size": batch_size,
                    "format": "npz-f32",
                },
                ensure_ascii=False,
            )
        )

    # resume 지점 계산
    start_idx = 0
    if ckpt_dir_path and resume:
        start_idx = _detect_last_end_index(ckpt_dir_path)

    processed = start_idx
    if progress_cb:
        progress_cb(processed, total)

    # 배치 인코딩 + part 저장(npz, f32)
    part_files: list[Path | None] = []
    i = start_idx
    while i < total:
        j = min(i + batch_size, total)
        batch_texts = cleaned.iloc[i:j].tolist()

        emb = model.encode(batch_texts, show_progress_bar=False, batch_size=batch_size)
        emb = np.asarray(emb, dtype=np.float32)

        if ckpt_dir_path:
            part_path = ckpt_dir_path / f"part_{i:06d}_{j:06d}.npz"
            _save_part_npz(part_path, emb)
            part_files.append(part_path)
        else:
            part_files.append(None)

        i = j
        processed = j
        if progress_cb:
            try:
                progress_cb(processed, total)
            except Exception:
                pass  # 콜백 예외는 무시

    # 파트 병합
    if ckpt_dir_path:
        files = _list_parts(ckpt_dir_path, resume=True)
        # 방금 런에서 만든 파트가 있다면 그것 기준으로 사용
        recent_parts = [p for p in part_files if isinstance(p, Path)]
        if recent_parts:
            files = sorted(set(list(files) + recent_parts))

        if not files:
            # (이례적) 파트가 전혀 없으면 한 번에 인코딩
            vectors = np.asarray(
                model.encode(cleaned.tolist(), show_progress_bar=False, batch_size=batch_size),
                dtype=np.float32,
            )
        else:
            # ① 레거시 .npy를 npz로 마이그레이션
            files = _migrate_legacy_npy_to_npz(files)

            # ② 전체 크기/차원 계산
            dims, rows_total = None, 0
            for p in files:
                with np.load(str(p)) as data:
                    arr = np.asarray(data["emb"], dtype=np.float32, copy=False)
                    if arr.ndim != 2:
                        raise RuntimeError(f"invalid part shape: {p} -> {arr.shape}")
                    r, d = arr.shape
                    rows_total += r
                    dims = d if dims is None else dims

            # ③ 병합(메모리 상에서)
            big = np.empty((rows_total, dims), dtype=np.float32)
            offset = 0
            for p in files:
                with np.load(str(p)) as data:
                    arr = np.asarray(data["emb"], dtype=np.float32, copy=False)
                    r = arr.shape[0]
                    big[offset:offset + r, :] = arr
                    offset += r

            # ④ 최종 체크포인트(압축 npz) 기록
            final_path = ckpt_dir_path / "embeddings.final.npz"
            _save_part_npz(final_path, big)
            vectors = big
    else:
        # 체크포인트 미사용
        chunks = []
        for i in range(0, total, batch_size):
            j = min(i + batch_size, total)
            batch_texts = cleaned.iloc[i:j].tolist()
            emb = model.encode(batch_texts, show_progress_bar=False, batch_size=batch_size)
            chunks.append(np.asarray(emb, dtype=np.float32))
        vectors = np.vstack(chunks) if len(chunks) > 1 else chunks[0]

    # 출력 DF 구성
    out = df.copy()
    # float → python list[float]
    out["embedding"] = [row.astype(np.float32, copy=False).tolist() for row in vectors]
    return out
