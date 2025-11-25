from __future__ import annotations

import json
import re
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
# JSONL 유틸 (update_date)
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
    """
    구버전에서 사용했던 단순 함수 — 이제는 사용 빈도가 낮음.
    """
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
# 파일 기반 JSONL → JSONL (cutoff)
# =========================================================
def count_until_year_from_path(input_path: str, cutoff_year: int = 2026) -> int:
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
# --- 완전 강화된 year 파생 로직 ---
# =========================================================

RE_YEAR = re.compile(r"(19|20)\d{2}")
RE_ARXIV_OLD = re.compile(r"^(?P<yy>\d{2})(?P<mm>\d{2})\.\d+")
RE_ARXIV_NEW = re.compile(r"^(?P<yy>\d{2})(?P<mm>\d{2})\d{3,5}")

def _extract_year_from_arxiv_id(s: str) -> int:
    if not s:
        return -1
    s = s.strip()

    m = RE_ARXIV_OLD.match(s)
    if m:
        yy = int(m.group("yy"))
        return 2000 + yy

    m = RE_ARXIV_NEW.match(s)
    if m:
        yy = int(m.group("yy"))
        return 2000 + yy

    return -1


def _extract_year_from_any_date(s: str) -> int:
    if not s:
        return -1

    s = str(s)

    # 1) pandas datetime
    try:
        dt = pd.to_datetime(s, errors="raise")
        return int(dt.year)
    except:
        pass

    # 2) YYYYMMDD
    if s.isdigit() and len(s) == 8:
        return int(s[:4])

    # 3) 문자열 내부에서 4자리 연도
    m = RE_YEAR.search(s)
    if m:
        return int(m.group())

    return -1


def _derive_year(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 0) year 컬럼 있는 경우
    if "year" in out.columns:
        out["year"] = out["year"].apply(_extract_year_from_any_date)
        return out

    # 1) arXiv ID 기반
    id_candidates = [c for c in out.columns if "id" in c.lower()]
    for c in id_candidates:
        yrs = out[c].astype(str).apply(_extract_year_from_arxiv_id)
        if (yrs >= 2007).any():
            out["year"] = yrs
            return out

    # 2) date-like 컬럼
    date_candidates = []
    for c in out.columns:
        cl = c.lower()
        if "date" in cl or "time" in cl or "publish" in cl or "created" in cl:
            date_candidates.append(c)

    for c in date_candidates:
        yrs = out[c].astype(str).apply(_extract_year_from_any_date)
        if (yrs >= 1900).any():
            out["year"] = yrs
            return out

    # 3) fallback: 행 전체 문자열에서 연도 추출
    fallback_years = []
    for _, row in out.iterrows():
        y = -1
        for v in row:
            y = _extract_year_from_any_date(str(v))
            if y != -1:
                break
        fallback_years.append(y)

    out["year"] = fallback_years
    return out

# =========================================================
# DataFrame 기반 year-filter
# =========================================================
def filter_df_before_year(df: pd.DataFrame, cutoff_year: int) -> pd.DataFrame:
    df2 = _derive_year(df)
    yrs = pd.to_numeric(df2["year"], errors="coerce").fillna(-1).astype(int)
    return df2[yrs < int(cutoff_year)].copy().reset_index(drop=True)

# =========================================================
# 키워드 필터
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
                return any(re.search(k, text) for k in kw_list)
            return any(k in text for k in kw_list)
    else:
        kw_list = keywords

        def _hit(row: pd.Series) -> bool:
            text = _concat_text(row, use_cols)
            if use_regex:
                import re
                return any(re.search(k, text) for k in kw_list)
            return any(k in text for k in kw_list)

    mask = proc.apply(_hit, axis=1)
    return proc[mask].copy().reset_index(drop=True)

# =========================================================
# 공개 전처리 엔트리포인트
# =========================================================
def run_preprocess(
    df: pd.DataFrame,
    cutoff_year: int = 2026,
    *,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    **kwargs
) -> pd.DataFrame:

    def _ping(proc: int, tot: int, stage: str):
        if progress_cb:
            try:
                progress_cb(int(proc), int(tot), stage)
            except:
                pass

    total_in = len(df)
    _ping(0, max(total_in, 1), "preprocess_start")

    # 1) year 파생
    out = _derive_year(df)

    # title/abstract 존재 처리
    title_col = "title" if "title" in out.columns else None
    abstr_col = "abstract" if "abstract" in out.columns else None

    if title_col:
        out[title_col] = out[title_col].astype(str).fillna("")
    if abstr_col:
        out[abstr_col] = out[abstr_col].astype(str).fillna("")

    # 빈 제목/초록 제거
    if title_col and abstr_col:
        mask = (out[title_col].str.len() > 0) | (out[abstr_col].str.len() > 0)
        out = out[mask]
    elif title_col:
        out = out[out[title_col].str.len() > 0]
    elif abstr_col:
        out = out[out[abstr_col].str.len() > 0]

    _ping(len(out), max(total_in, 1), "clean_done")

    # 2) 키워드 필터
    out = filter_df_by_keywords_literal(
        out,
        DEFAULT_KEYWORDS,
        text_cols=[c for c in ("title", "abstract") if c in out.columns],
        case_insensitive=True,
        use_regex=False,
    )

    _ping(len(out), max(total_in, 1), "keyword_filter_done")

    # 정리
    out = out.reset_index(drop=True)

    # year 컬럼 손실 방지
    if "year" not in out.columns:
        out["year"] = -1

    return out
