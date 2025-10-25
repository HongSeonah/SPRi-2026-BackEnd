# app/api/pipeline.py
import asyncio
import io
import csv
import json
from typing import Any, Tuple

import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from app.core.preprocess import run_preprocess, filter_df_before_year
from app.core.embedding import run_embedding
from app.core.clustering import run_clustering
from app.core.tech_naming import run_tech_naming
from app.core.utils_prompt import build_user_prompt

router = APIRouter(tags=["Pipeline"])

# --------- 업로드 파일 → DataFrame 로더 (JSONL 우선) ---------
def _load_table_from_upload(
    file_bytes: bytes,
    filename: str | None = None,
    content_type: str | None = None,
    prefer_jsonl: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    업로드 파일을 DataFrame으로 로딩 (JSONL 우선, JSON/CSV/TSV/Excel도 지원)
    반환: (df, meta)
    """
    name = (filename or "").lower().strip()
    ctype = (content_type or "").lower().strip()
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1"]

    # 텍스트 스니핑
    sniff_text, sniff_enc = None, None
    for enc in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            sniff_text = file_bytes.decode(enc, errors="strict")
            sniff_enc = enc
            break
        except Exception:
            continue

    def _looks_like_json(txt: str) -> bool:
        s = txt.lstrip()
        return s.startswith("{") or s.startswith("[")

    def _looks_like_jsonl(txt: str) -> bool:
        if txt.count("\n") < 1:
            return False
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        sample = lines[: min(5, len(lines))]
        return len(sample) > 1 and all(ln.startswith("{") or ln.startswith("[") for ln in sample)

    # JSONL/JSON 내용 우선 감지
    if prefer_jsonl and sniff_text:
        if _looks_like_jsonl(sniff_text):
            try:
                df = pd.read_json(io.StringIO(sniff_text), lines=True)
                return df, {"format": "jsonl-sniff", "encoding": sniff_enc}
            except Exception:
                pass
        if _looks_like_json(sniff_text):
            try:
                obj = json.loads(sniff_text)
                if isinstance(obj, list):
                    return pd.DataFrame(obj), {"format": "json-sniff", "encoding": sniff_enc}
                elif isinstance(obj, dict):
                    for k in ("data", "items", "records", "rows", "result"):
                        if k in obj and isinstance(obj[k], list):
                            return pd.DataFrame(obj[k]), {"format": "json-sniff", "encoding": sniff_enc, "root": k}
                    return pd.json_normalize(obj), {"format": "json-sniff", "encoding": sniff_enc, "normalized": True}
            except Exception:
                pass

    # Excel
    is_xlsx = name.endswith(".xlsx") or "spreadsheetml" in ctype
    if is_xlsx:
        df_xlsx = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
        return df_xlsx, {"format": "excel"}

    # 확장자/타입 기반 JSON/JSONL
    is_jsonl = any(s in (name, ctype) for s in (".jsonl", ".ndjson", "ndjson"))
    is_json  = any(s in (name, ctype) for s in (".json", "application/json")) and not is_jsonl
    if is_jsonl or is_json:
        last_err = None
        for enc in encodings:
            try:
                txt = file_bytes.decode(enc, errors="strict")
            except Exception as e:
                last_err = e
                continue
            try:
                if is_jsonl:
                    df = pd.read_json(io.StringIO(txt), lines=True)
                    return df, {"format": "jsonl", "encoding": enc}
                else:
                    obj = json.loads(txt)
                    if isinstance(obj, list):
                        return pd.DataFrame(obj), {"format": "json", "encoding": enc}
                    elif isinstance(obj, dict):
                        for k in ("data", "items", "records", "rows", "result"):
                            if k in obj and isinstance(obj[k], list):
                                return pd.DataFrame(obj[k]), {"format": "json", "encoding": enc, "root": k}
                        return pd.json_normalize(obj), {"format": "json", "encoding": enc, "normalized": True}
                    else:
                        last_err = ValueError("Unsupported JSON top-level type")
                        continue
            except Exception as e:
                last_err = e
                continue
        raise ValueError(f"Failed to parse JSON: {last_err}")

    # CSV/TSV
    last_err = None
    sample = file_bytes[:20000]
    for enc in encodings:
        try:
            txt = file_bytes.decode(enc, errors="strict")
        except Exception as e:
            last_err = e
            continue
        try:
            dialect = csv.Sniffer().sniff(sample.decode("utf-8", errors="ignore"), delimiters=",;\t|")
            sep_guess = dialect.delimiter
        except Exception:
            sep_guess = None
        try:
            df = pd.read_csv(io.StringIO(txt), sep=sep_guess, engine="c", quotechar='"', escapechar="\\", on_bad_lines="error")
            return df, {"format": "csv", "encoding": enc, "sep": sep_guess, "engine": "c"}
        except Exception as e:
            last_err = e
        try:
            df = pd.read_csv(io.StringIO(txt), sep=sep_guess, engine="python", quotechar='"', escapechar="\\", on_bad_lines="skip")
            return df, {"format": "csv", "encoding": enc, "sep": sep_guess, "engine": "python", "on_bad_lines": "skip"}
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"Failed to parse as CSV/TSV. Last error: {last_err}")


# --------- 안전 리스트 변환 (요약값 방어) ---------
def _to_str_list(x: Any) -> list[str]:
    import pandas as pd
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    if isinstance(x, pd.Series):
        return x.dropna().astype(str).tolist()
    if isinstance(x, pd.DataFrame):
        return x.astype(str).stack().tolist()
    return [str(x)]


@router.post("/pipeline/run")
async def run_pipeline(
    file: UploadFile = File(...),
    cutoff_year: int = Form(2025),
    n_clusters: int = Form(100),
    model_name: str = Form("all-MiniLM-L6-v2"),
):
    # 업로드 전체 바이트를 미리 읽어둠 (stream 함수 내부에서 await 불가)
    file_bytes = await file.read()
    filename = file.filename
    content_type = file.content_type

    async def stream():
        try:
            # 0) 파일 로드
            yield json.dumps({"step": "파일 로드 중", "progress": 0}) + "\n"
            df, meta = await asyncio.to_thread(_load_table_from_upload, file_bytes, filename, content_type, True)
            yield json.dumps({"step": "파일 로드 완료", "progress": 5, "meta": {"filename": filename, "content_type": content_type, **meta}}) + "\n"

            # 1) 전처리 (연도 파생 포함)
            yield json.dumps({"step": "데이터 전처리 시작", "progress": 10}) + "\n"
            df_clean = await asyncio.to_thread(run_preprocess, df, int(cutoff_year))

            # 2) (선택) 원문 규칙과 동일 부등식(< cutoff_year)로 필터링이 필요하면 아래 사용
            #    외부 스크립트와 결과 동일성 필요할 때 켜면 됨.
            df_filtered = await asyncio.to_thread(filter_df_before_year, df_clean, int(cutoff_year))

            # 3) 임베딩
            yield json.dumps({"step": "임베딩 생성 중", "progress": 40}) + "\n"
            df_embed = await asyncio.to_thread(run_embedding, df_filtered, model_name)

            # 안전 확인
            if not isinstance(df_embed, pd.DataFrame):
                raise TypeError(f"run_embedding must return a DataFrame, got {type(df_embed)}")

            # 4) 클러스터링/추세
            yield json.dumps({"step": "클러스터링 및 추세 분석 중", "progress": 70}) + "\n"
            df_clustered, summary = await asyncio.to_thread(run_clustering, df_embed, n_clusters)

            # 5) 기술명명 (요약값 강제 정규화)
            yield json.dumps({"step": "기술명 생성 중", "progress": 90}) + "\n"

            if isinstance(summary, dict):
                kw_raw = summary.get("keywords", [])
                tt_raw = summary.get("titles", [])
            else:
                # dict가 아닌 경우(Series/DataFrame/Namespace 등) 속성으로 접근
                kw_raw = getattr(summary, "keywords", [])
                tt_raw = getattr(summary, "titles", [])

            keywords = _to_str_list(kw_raw)
            titles   = _to_str_list(tt_raw)

            year_val = 2024
            if isinstance(df_clustered, pd.DataFrame) and "year" in df_clustered.columns:
                try:
                    year_val = int(pd.to_numeric(df_clustered["year"], errors="coerce").max())
                except Exception:
                    year_val = 2024

            prompt = build_user_prompt(
                flow_id="AUTO_FLOW",
                cluster_name="자동생성_클러스터",
                year=year_val,
                keywords=keywords,
                rep_titles=titles
            )
            naming_result = await asyncio.to_thread(run_tech_naming, prompt)

            # 완료
            yield json.dumps({
                "step": "완료",
                "progress": 100,
                "result": {"summary": summary if isinstance(summary, dict) else str(type(summary)), "naming": naming_result}
            }) + "\n"

        except Exception as e:
            yield json.dumps({"step": "오류 발생", "progress": -1, "error": str(e)}) + "\n"

    return StreamingResponse(
        stream(),
        media_type="application/json",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
