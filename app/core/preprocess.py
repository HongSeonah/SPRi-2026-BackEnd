# app/core/preprocess.py
from __future__ import annotations

import json
from typing import Iterable, IO, Optional, List, Tuple
import pandas as pd
from fastapi import APIRouter
from pathlib import Path
from typing import Sequence, Literal, Union

router = APIRouter()

# =========================================================
# JSONL 유틸 (update_date[:4].isdigit() + int < cutoff)
# =========================================================

def _iter_jsonl_lines(
    fp: IO[str],
    *,
    log_errors: bool = False
) -> Iterable[dict]:
    """
    JSON Lines를 한 줄씩 dict로 파싱.
    - 실패 라인은 skip. (log_errors=True면 에러를 print)
    - 원문 스크립트는 실패 시 continue 하므로 기본값은 조용히 skip.
    """
    for i, line in enumerate(fp, start=1):
        s = line.strip()
        if not s:
            continue
        try:
            yield json.loads(s)
        except Exception as e:
            if log_errors:
                print(f"[iter_jsonl] line {i} parse error: {e}")
            continue


def _year_from_update_date_like_original(update_date: str) -> int:
    """
    원문 방식 그대로: 앞 4자리 슬라이싱 → isdigit() → int 변환, 아니면 -1.
    (정규식 사용하지 않음)
    """
    s = str(update_date)
    y4 = s[:4]
    return int(y4) if y4.isdigit() else -1


def count_before_year_stream(fp: IO[str], cutoff_year: int) -> int:
    """
    ✅ 원문 로직 그대로: update_date[:4].isdigit() and int(...) < cutoff_year
    """
    cnt = 0
    for entry in _iter_jsonl_lines(fp):
        y = _year_from_update_date_like_original(entry.get("update_date", ""))
        if y >= 0 and y < int(cutoff_year):
            cnt += 1
    return cnt

def _concat_text(row: pd.Series, cols: Sequence[str]) -> str:
    """
    주어진 컬럼들을 공백으로 이어붙여 검색/유사도용 텍스트 생성.
    None/NaN은 빈 문자열로 처리.
    """
    parts = []
    for c in cols:
        if c in row and pd.notna(row[c]):
            parts.append(str(row[c]))
    return " ".join(parts).strip()

def filter_df_by_keywords_literal(
    df: pd.DataFrame,
    keywords: Sequence[str],
    *,
    text_cols: Sequence[str] = ("title", "abstract"),
    case_insensitive: bool = True,
    use_regex: bool = False
) -> pd.DataFrame:
    """
    문자열 매칭으로 키워드가 하나라도 포함된 행만 남김.
    - 기본: 대소문자 무시, 정규식 미사용(부분문자열 포함 검사).
    - text_cols 중 존재하는 컬럼만 사용.
    """
    use_cols = [c for c in text_cols if c in df.columns]
    if not use_cols or not keywords:
        return df.copy()

    proc = df.copy()
    if case_insensitive:
        kw_list = [str(k).lower() for k in keywords]
        def _hit(row: pd.Series) -> bool:
            text = _concat_text(row, use_cols).lower()
            if use_regex:
                import re
                return any(re.search(k, text) is not None for k in kw_list)
            return any(k in text for k in kw_list)
    else:
        kw_list = [str(k) for k in keywords]
        def _hit(row: pd.Series) -> bool:
            text = _concat_text(row, use_cols)
            if use_regex:
                import re
                return any(re.search(k, text) is not None for k in kw_list)
            return any(k in text for k in kw_list)

    mask = proc.apply(_hit, axis=1)
    return proc[mask].copy().reset_index(drop=True)

def filter_df_by_keywords_semantic(
    df: pd.DataFrame,
    keywords: Sequence[str],
    *,
    text_cols: Sequence[str] = ("title", "abstract"),
    model_name: str = "all-MiniLM-L6-v2",
    threshold: float = 0.30,
    normalize: bool = True,
    device: Optional[str] = None,
    add_score_cols: bool = True
) -> pd.DataFrame:
    """
    문장 임베딩 유사도로 키워드와 유사한 행만 남김.
    - 각 행의 텍스트(예: title+abstract) vs. 키워드 리스트의 유사도 중 최댓값이 threshold 이상이면 keep
    - 결과에 best_keyword, similarity 컬럼(옵션) 추가
    """
    use_cols = [c for c in text_cols if c in df.columns]
    if not use_cols or not keywords:
        return df.copy()

    try:
        from sentence_transformers import SentenceTransformer, util
        import torch
    except Exception as e:
        raise ImportError(
            "sentence-transformers 가 필요합니다. "
            "pip install sentence-transformers 로 설치 후 사용하세요."
        ) from e

    proc = df.copy()
    texts = proc.apply(lambda r: _concat_text(r, use_cols), axis=1).fillna("").astype(str).tolist()

    # 디바이스 결정
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_name, device=device)
    text_emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=normalize, show_progress_bar=False, device=device)
    kw_emb = model.encode(list(map(str, keywords)), convert_to_tensor=True, normalize_embeddings=normalize, show_progress_bar=False, device=device)

    sim = util.cos_sim(text_emb, kw_emb)  # [num_rows x num_keywords]
    best_scores, best_idx = sim.max(dim=1)

    keep_mask = best_scores >= float(threshold)
    kept = proc[keep_mask.cpu().numpy()].copy().reset_index(drop=True)

    if add_score_cols:
        kept["similarity"] = best_scores[keep_mask].cpu().numpy()
        kept["best_keyword"] = [keywords[i] for i in best_idx[keep_mask].cpu().tolist()]

    return kept



def filter_before_year_stream_to_df(fp: IO[str], cutoff_year: int) -> pd.DataFrame:
    """
    ✅ 원문 로직으로 필터링 → DataFrame 반환.
    """
    rows: List[dict] = []
    for entry in _iter_jsonl_lines(fp):
        y = _year_from_update_date_like_original(entry.get("update_date", ""))
        if y >= 0 and y < int(cutoff_year):
            rows.append(entry)
    return pd.DataFrame(rows)


# =========================================================
# 파일 경로 기반 I/O 유틸 (원문 스크립트와 1:1 대응)
# =========================================================

def count_until_year_from_path(input_path: str, cutoff_year: int = 2025) -> int:
    """
    원문 count 스크립트와 동일 동작:
    - update_date[:4].isdigit() and int(...) < cutoff_year
    - 예: cutoff_year=2026 → '2025까지' 카운트
    """
    p = Path(input_path)
    cnt = 0
    with p.open("r", encoding="utf-8") as infile:
        for line in infile:
            try:
                entry = json.loads(line)
                y = _year_from_update_date_like_original(entry.get("update_date", ""))
                if y >= 0 and y < int(cutoff_year):
                    cnt += 1
            except Exception:
                # 원문은 그냥 continue
                continue
    return cnt


def filter_jsonl_to_jsonl_with_cutoff(
    input_path: str,
    output_path: str,
    cutoff_year: int = 2026,
    *,
    log_errors: bool = False
) -> int:
    """
    원문 필터 스크립트와 동일 동작:
    - 조건: update_date[:4].isdigit() and int(...) < cutoff_year
    - 매칭 레코드를 JSONL로 저장, 저장한 개수 반환.
    - 원문은 파싱 오류 시 print 후 계속 진행 → log_errors=True로 설정 가능.
    """
    in_p = Path(input_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with in_p.open("r", encoding="utf-8") as infile, out_p.open("w", encoding="utf-8") as outfile:
        for i, line in enumerate(infile, start=1):
            try:
                entry = json.loads(line)
                y = _year_from_update_date_like_original(entry.get("update_date", ""))
                if y >= 0 and y < int(cutoff_year):
                    json.dump(entry, outfile, ensure_ascii=False)
                    outfile.write("\n")
                    count += 1
            except Exception as e:
                if log_errors:
                    print(f"[filter_jsonl] line {i} parse error: {e}")
                # 원문은 에러시 계속
                continue
    return count


def jsonl_to_csv(input_jsonl_path: str, output_csv_path: str) -> Tuple[int, str]:
    """
    원문: pd.read_json(..., lines=True) → df.to_csv(...)
    반환: (행 수, 출력 경로)
    """
    in_p = Path(input_jsonl_path)
    out_p = Path(output_csv_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(str(in_p), lines=True)
    df.to_csv(str(out_p), index=False)
    return len(df), str(out_p)


# =========================================================
# DataFrame 기반 파생/필터 (원문 방식과 동일성 유지)
# =========================================================

def _derive_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    update_date에서 앞 4자리 추출하여 'year' 파생.
    - 정규식이 아닌 '슬라이싱 + isdigit + int'로 원문과 완전 동일화.
    - 비유효값은 -1.
    """
    out = df.copy()
    if "year" not in out.columns:
        if "update_date" in out.columns:
            out["year"] = (
                out["update_date"]
                .astype(str)
                .map(lambda s: int(s[:4]) if s[:4].isdigit() else -1)
                .astype(int)
            )
        else:
            out["year"] = -1
    return out


def filter_df_before_year(df: pd.DataFrame, cutoff_year: int) -> pd.DataFrame:
    """
    ✅ 원문과 동일한 부등식: int(update_date[:4]) < cutoff_year
    """
    df2 = _derive_year(df)
    yrs = pd.to_numeric(df2["year"], errors="coerce").fillna(-1).astype(int)
    return df2[yrs < int(cutoff_year)].copy()


# =========================================================
# 공개 전처리 엔트리포인트
# =========================================================

def run_preprocess(
    df: pd.DataFrame,
    cutoff_year: int = 2025,
    *,
    # ▼ 새 옵션들
    keyword_mode: Optional[Literal["literal", "semantic"]] = None,
    keywords: Optional[Sequence[str]] = None,
    # literal 모드 옵션
    case_insensitive: bool = True,
    use_regex: bool = False,
    # semantic 모드 옵션
    semantic_threshold: float = 0.30,
    semantic_model_name: str = "all-MiniLM-L6-v2",
    semantic_normalize: bool = True,
    semantic_device: Optional[str] = None,
    add_score_cols: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    파이프라인에서 호출되는 전처리 함수.
    - year 파생
    - 제목/초록 모두 빈 행 제거
    - (옵션) 키워드 기반 필터링: literal 또는 semantic
    - *연도 필터(< cutoff_year)는 호출자에서 선택적으로 적용*
    """
    out = _derive_year(df)

    title_col = "title" if "title" in out.columns else None
    abstr_col = "abstract" if "abstract" in out.columns else None

    if title_col is None and abstr_col is None:
        # 텍스트 컬럼이 없으면 year 파생만 적용한 결과 반환
        return out.reset_index(drop=True)

    # 문자열화 + NaN -> ""
    if title_col:
        out[title_col] = out[title_col].astype(str).fillna("")
    if abstr_col:
        out[abstr_col] = out[abstr_col].astype(str).fillna("")

    # 제목/초록 모두 빈 문자열인 행 제거
    if title_col and abstr_col:
        mask_keep = (out[title_col].str.len() > 0) | (out[abstr_col].str.len() > 0)
        out = out[mask_keep]
    elif title_col:
        out = out[out[title_col].str.len() > 0]
    elif abstr_col:
        out = out[out[abstr_col].str.len() > 0]

    # ▼ 키워드 기반 필터(옵션)
    if keyword_mode and keywords:
        if keyword_mode == "literal":
            out = filter_df_by_keywords_literal(
                out, keywords,
                text_cols=tuple([c for c in ("title", "abstract") if c in out.columns]),
                case_insensitive=case_insensitive,
                use_regex=use_regex
            )
        elif keyword_mode == "semantic":
            out = filter_df_by_keywords_semantic(
                out, keywords,
                text_cols=tuple([c for c in ("title", "abstract") if c in out.columns]),
                model_name=semantic_model_name,
                threshold=semantic_threshold,
                normalize=semantic_normalize,
                device=semantic_device,
                add_score_cols=add_score_cols
            )
        else:
            raise ValueError("keyword_mode 는 None, 'literal', 'semantic' 중 하나여야 합니다.")

    return out.reset_index(drop=True)