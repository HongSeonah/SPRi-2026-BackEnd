# app/core/preprocess.py
from __future__ import annotations

import json
from typing import Iterable, IO
import pandas as pd
import re

# -----------------------------
# JSONL 유틸 (필요시 외부 스크립트와 동일 동작을 위한 도우미)
# -----------------------------
def _iter_jsonl_lines(fp: IO[str]) -> Iterable[dict]:
    """JSON Lines를 한 줄씩 dict로 파싱 (실패 라인은 조용히 skip)."""
    for line in fp:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def count_before_year_stream(fp: IO[str], cutoff_year: int) -> int:
    """
    ✅ 원문 로직 그대로:
       update_date[:4].isdigit() and int(...) < cutoff_year
    """
    cnt = 0
    for entry in _iter_jsonl_lines(fp):
        upd = entry.get("update_date", "")
        y4 = upd[:4]
        if y4.isdigit() and int(y4) < cutoff_year:
            cnt += 1
    return cnt


def filter_before_year_stream_to_df(fp: IO[str], cutoff_year: int) -> pd.DataFrame:
    """
    ✅ 원문 로직으로 필터링 → DataFrame 반환.
    """
    rows = []
    for entry in _iter_jsonl_lines(fp):
        upd = entry.get("update_date", "")
        y4 = upd[:4]
        if y4.isdigit() and int(y4) < cutoff_year:
            rows.append(entry)
    return pd.DataFrame(rows)


# -----------------------------
# DataFrame 기반 파생/필터
# -----------------------------
def _derive_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    update_date에서 앞 4자리 추출하여 'year' 파생.
    - 원문 스크립트의 연도 인식 방식과 동일(슬라이싱 + isdigit + int).
    - 텍스트는 변형하지 않음(결과 동일성 보장).
    """
    out = df.copy()
    if "year" not in out.columns:
        if "update_date" in out.columns:
            # 문자열화 후 앞 4자리 추출 (^\d{4} 만 허용)
            y4 = out["update_date"].astype(str).str.extract(r"^(\d{4})")[0]
            out["year"] = (
                y4.where(y4.str.fullmatch(r"\d{4}").fillna(False), "-1")
                .fillna("-1")
                .astype(int)
            )
        else:
            out["year"] = -1
    return out


def filter_df_before_year(df: pd.DataFrame, cutoff_year: int) -> pd.DataFrame:
    """
    ✅ 원문과 동일한 부등식: int(update_date[:4]) < cutoff_year
    - DataFrame에 'year'가 없다면 _derive_year로 생성한 후 필터.
    """
    df2 = _derive_year(df)
    yrs = pd.to_numeric(df2["year"], errors="coerce").fillna(-1).astype(int)
    return df2[yrs < int(cutoff_year)].copy()


# -----------------------------
# 공개 전처리 엔트리포인트
# -----------------------------
def run_preprocess(df: pd.DataFrame, cutoff_year: int = 2025, **kwargs) -> pd.DataFrame:
    """
    파이프라인에서 호출되는 전처리 함수.
    - 텍스트(제목/초록)는 그대로 둠 (결과 동일성 유지)
    - 'year' 파생만 수행
    - 제목/초록이 모두 비어있는 레코드는 제거(의미 없음)
    - 마지막에 '원문과 동일 부등식(< cutoff_year)'를 적용할 수 있도록
      별도의 필터는 pipeline 쪽(혹은 호출자)에서 선택적으로 수행.
    """
    out = _derive_year(df)

    # 흔히 사용하는 텍스트 컬럼 이름 보정(있을 때만)
    title_col = "title" if "title" in out.columns else None
    abstr_col = "abstract" if "abstract" in out.columns else None

    # 제목/초록 둘 다 전혀 없으면 그대로 반환(임베딩 단계가 처리하도록)
    if title_col is None and abstr_col is None:
        return out.reset_index(drop=True)

    # 문자열화 + NaN -> "" (내용 자체는 변경하지 않음)
    if title_col:
        out[title_col] = out[title_col].astype(str).fillna("")
    if abstr_col:
        out[abstr_col] = out[abstr_col].astype(str).fillna("")

    # 제목/초록이 모두 빈 문자열인 행은 제거
    if title_col and abstr_col:
        mask_keep = (out[title_col].str.len() > 0) | (out[abstr_col].str.len() > 0)
        out = out[mask_keep]
    elif title_col:
        out = out[out[title_col].str.len() > 0]
    elif abstr_col:
        out = out[out[abstr_col].str.len() > 0]

    return out.reset_index(drop=True)
