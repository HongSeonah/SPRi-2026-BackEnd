# app/core/preprocess.py
from __future__ import annotations

import json
from typing import Iterable, IO, Optional, List, Tuple
import pandas as pd
from fastapi import APIRouter
from pathlib import Path

router = APIRouter()

# =========================================================
# JSONL 유틸 (원문 동작과 동일성 보장: update_date[:4].isdigit() + int < cutoff)
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
    **kwargs
) -> pd.DataFrame:
    """
    파이프라인에서 호출되는 전처리 함수.
    - 텍스트(제목/초록)는 그대로 둠(내용 변형 없음)
    - 'year' 파생만 수행(슬라이싱 방식)
    - 제목/초록이 모두 비어있는 레코드는 제거(둘 다 공백/빈문자열이면 drop)
    - '연도 필터(< cutoff_year)'는 호출자에서 선택적으로 적용
    """
    out = _derive_year(df)

    title_col = "title" if "title" in out.columns else None
    abstr_col = "abstract" if "abstract" in out.columns else None

    # 둘 다 없으면 그대로 반환 (임베딩 단계 등에서 처리)
    if title_col is None and abstr_col is None:
        return out.reset_index(drop=True)

    # 문자열화 + NaN -> "" (내용 자체는 변경하지 않음)
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

    return out.reset_index(drop=True)


# =========================================================
# (선택) 간단한 CLI 유틸: 모듈을 단독 실행했을 때 원문 스크립트와 유사 동작
#   python -m app.core.preprocess count <input_jsonl> [cutoff_year]
#   python -m app.core.preprocess filter <input_jsonl> <output_jsonl> [cutoff_year]
#   python -m app.core.preprocess tocsv <input_jsonl> <output_csv>
# =========================================================

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if not args:
        print(
            "Usage:\n"
            "  python -m app.core.preprocess count <input_jsonl> [cutoff_year]\n"
            "  python -m app.core.preprocess filter <input_jsonl> <output_jsonl> [cutoff_year]\n"
            "  python -m app.core.preprocess tocsv <input_jsonl> <output_csv>\n"
        )
        sys.exit(0)

    cmd = args[0].lower()

    if cmd == "count":
        if len(args) < 2:
            print("count <input_jsonl> [cutoff_year]")
            sys.exit(1)
        input_path = args[1]
        cutoff = int(args[2]) if len(args) >= 3 else 2025
        n = count_until_year_from_path(input_path, cutoff)
        # 원문 출력 메시지 형식 맞춤
        print(f"✅ {cutoff-1}년까지의 논문 수: {n}")

    elif cmd == "filter":
        if len(args) < 3:
            print("filter <input_jsonl> <output_jsonl> [cutoff_year]")
            sys.exit(1)
        input_path = args[1]
        output_path = args[2]
        cutoff = int(args[3]) if len(args) >= 4 else 2025
        saved = filter_jsonl_to_jsonl_with_cutoff(input_path, output_path, cutoff_year=cutoff, log_errors=True)
        print(f"✅ 필터링 완료: {saved}개의 논문이 저장됨")

    elif cmd == "tocsv":
        if len(args) < 3:
            print("tocsv <input_jsonl> <output_csv>")
            sys.exit(1)
        input_path = args[1]
        output_csv = args[2]
        rows, path_out = jsonl_to_csv(input_path, output_csv)
        print(f"✅ CSV 저장 완료: {rows}행 → {path_out}")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
