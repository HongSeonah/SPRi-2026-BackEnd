# app/core/preprocess.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, IO, Optional, List, Tuple, Sequence, Callable

import pandas as pd
from fastapi import APIRouter

router = APIRouter()

# =========================================================
# 고정 키워드
# =========================================================
DEFAULT_KEYWORDS = ['data', 'algorithm', 'software', 'reality', 'virtual', 'augmented']

# =========================================================
# JSONL 유틸 (update_date[:4].isdigit() + int < cutoff)
# =========================================================
def _iter_jsonl_lines(
    fp: IO[str],
    *,
    log_errors: bool = False
) -> Iterable[dict]:
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
    s = str(update_date)
    y4 = s[:4]
    return int(y4) if y4.isdigit() else -1


def count_before_year_stream(fp: IO[str], cutoff_year: int) -> int:
    cnt = 0
    for entry in _iter_jsonl_lines(fp):
        y = _year_from_update_date_like_original(entry.get("update_date", ""))
        if y >= 0 and y < int(cutoff_year):
            cnt += 1
    return cnt


def filter_before_year_stream_to_df(fp: IO[str], cutoff_year: int) -> pd.DataFrame:
    rows: List[dict] = []
    for entry in _iter_jsonl_lines(fp):
        y = _year_from_update_date_like_original(entry.get("update_date", ""))
        if y >= 0 and y < int(cutoff_year):
            rows.append(entry)
    return pd.DataFrame(rows)

# =========================================================
# 파일 경로 기반 I/O (원문 스크립트와 동일)
# =========================================================
def count_until_year_from_path(input_path: str, cutoff_year: int = 2025) -> int:
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
                continue
    return cnt


def filter_jsonl_to_jsonl_with_cutoff(
    input_path: str,
    output_path: str,
    cutoff_year: int = 2026,
    *,
    log_errors: bool = False
) -> int:
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
                continue
    return count


def jsonl_to_csv(input_jsonl_path: str, output_csv_path: str) -> Tuple[int, str]:
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
    원문과 동일 부등식: int(update_date[:4]) < cutoff_year
    (연도 파싱 실패 -1 도 포함하는 원문 규칙 유지)
    """
    df2 = _derive_year(df)
    yrs = pd.to_numeric(df2["year"], errors="coerce").fillna(-1).astype(int)
    return df2[yrs < int(cutoff_year)].copy().reset_index(drop=True)

# =========================================================
# 키워드 필터 (literal 포함검사)
# =========================================================
def _concat_text(row: pd.Series, cols: Sequence[str]) -> str:
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

# =========================================================
# 공개 전처리 엔트리포인트
#  - 연도 파생/정리
#  - (항상) 키워드 리터럴 필터: DEFAULT_KEYWORDS
#  - progress_cb(processed:int, total:int, stage:str) 콜백 지원
# =========================================================
def run_preprocess(
    df: pd.DataFrame,
    cutoff_year: int = 2025,
    *,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    **kwargs
) -> pd.DataFrame:
    def _ping(proc: int, tot: int, stage: str):
        if progress_cb:
            try:
                progress_cb(int(proc), int(tot), str(stage))
            except Exception:
                pass

    # 0) 시작
    total_in = len(df)
    _ping(0, max(total_in, 1), "preprocess_start")

    # 1) year 파생 + 제목/초록 비어있는 행 제거 (원문 규칙)
    out = _derive_year(df)
    title_col = "title" if "title" in out.columns else None
    abstr_col = "abstract" if "abstract" in out.columns else None

    if title_col:
        out[title_col] = out[title_col].astype(str).fillna("")
    if abstr_col:
        out[abstr_col] = out[abstr_col].astype(str).fillna("")

    if title_col and abstr_col:
        mask_keep = (out[title_col].str.len() > 0) | (out[abstr_col].str.len() > 0)
        out = out[mask_keep]
    elif title_col:
        out = out[out[title_col].str.len() > 0]
    elif abstr_col:
        out = out[out[abstr_col].str.len() > 0]

    _ping(len(out), max(total_in, 1), "clean_done")

    # 2) (항상 적용) 키워드 리터럴 필터 — DEFAULT_KEYWORDS
    out = filter_df_by_keywords_literal(
        out,
        DEFAULT_KEYWORDS,
        text_cols=tuple([c for c in ("title", "abstract") if c in out.columns]),
        case_insensitive=True,
        use_regex=False,
    )
    _ping(len(out), max(total_in, 1), "keyword_filter_done")

    # 마무리
    out = out.drop(columns=["__text__"], errors="ignore").reset_index(drop=True)
    _ping(len(out), max(total_in, 1), "preprocess_done")
    return out
