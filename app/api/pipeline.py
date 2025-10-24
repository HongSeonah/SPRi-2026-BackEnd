from __future__ import annotations

import asyncio
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import io, csv, json
import pandas as pd

from app.core.preprocess import run_preprocess, filter_df_before_year
from app.core.embedding import run_embedding
from app.core.clustering import run_clustering
from app.core.tech_naming import run_tech_naming
from app.core.utils_prompt import build_user_prompt

router = APIRouter(tags=["Pipeline"])

# -----------------------------
# 업로드 파일 → DataFrame 로더
#  - JSONL 우선 감지
#  - JSON/CSV/TSV/Excel 보조
# -----------------------------
def _load_table_from_upload(
    file_bytes: bytes,
    filename: str | None = None,
    content_type: str | None = None,
    prefer_jsonl: bool = True,
):
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

    # 0) JSONL/JSON 내용 기반 우선 감지
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

    # 1) Excel
    is_xlsx = name.endswith(".xlsx") or "spreadsheetml" in ctype
    if is_xlsx:
        try:
            df_xlsx = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
            return df_xlsx, {"format": "excel"}
        except Exception as e:
            raise ValueError(f"Failed to read Excel: {e}")

    # 2) JSON/JSONL (확장자/컨텐트타입)
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

    # 3) CSV/TSV 자동 추정
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

        # (1) C 엔진
        try:
            df = pd.read_csv(
                io.StringIO(txt),
                sep=sep_guess,
                engine="c",
                quotechar='"',
                escapechar="\\",
                on_bad_lines="error",
            )
            return df, {"format": "csv", "encoding": enc, "sep": sep_guess, "engine": "c"}
        except Exception as e:
            last_err = e

        # (2) python 엔진 + skip
        try:
            df = pd.read_csv(
                io.StringIO(txt),
                sep=sep_guess,
                engine="python",
                quotechar='"',
                escapechar="\\",
                on_bad_lines="skip",
            )
            return df, {"format": "csv", "encoding": enc, "sep": sep_guess, "engine": "python", "on_bad_lines": "skip"}
        except Exception as e:
            last_err = e
            continue

    raise ValueError(f"Failed to parse as CSV/TSV. Last error: {last_err}")


# -----------------------------
# 파이프라인 엔드포인트
# -----------------------------
@router.post("/pipeline/run")
async def run_pipeline(
    file: UploadFile = File(...),
    cutoff_year: int = Form(2025),     # ✅ 의미: "< cutoff_year" (예: 2025 → 2024까지 사용)
    n_clusters: int = Form(100),
    model_name: str = Form("all-MiniLM-L6-v2"),
):
    file_bytes = await file.read()
    filename = file.filename
    content_type = file.content_type

    async def stream():
        try:
            # 0) 파일 로드
            yield json.dumps({"step": "파일 로드 중", "progress": 0}) + "\n"
            df, meta = await asyncio.to_thread(
                _load_table_from_upload, file_bytes, filename, content_type, True
            )
            yield json.dumps({
                "step": "파일 로드 완료", "progress": 5,
                "meta": {"filename": filename, "content_type": content_type, **meta}
            }) + "\n"

            # 1) 전처리(연도 파생만; 텍스트 변경X)
            yield json.dumps({"step": "데이터 전처리 시작", "progress": 10}) + "\n"
            df_clean = await asyncio.to_thread(run_preprocess, df)

            # 2) 연도 필터 (✅ 원문과 동일한 부등식: year < cutoff_year)
            #    update_date[:4]를 기반으로 만든 'year'에 대해 < cutoff_year만 유지
            df_filt = await asyncio.to_thread(filter_df_before_year, df_clean, cutoff_year)

            # 3) 임베딩
            yield json.dumps({"step": "임베딩 생성 중", "progress": 40}) + "\n"
            df_embed = await asyncio.to_thread(run_embedding, df_filt, model_name)

            # 4) 클러스터링 / 추세
            yield json.dumps({"step": "클러스터링 및 추세 분석 중", "progress": 70}) + "\n"
            df_clustered, summary = await asyncio.to_thread(run_clustering, df_embed, n_clusters)

            # 5) 기술명명
            yield json.dumps({"step": "기술명 생성 중", "progress": 90}) + "\n"
            keywords = summary.get("keywords", [])
            titles = summary.get("titles", [])
            year_val = int(df_clustered["year"].max()) if "year" in df_clustered.columns else 2024
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
                "result": {"summary": summary, "naming": naming_result}
            }) + "\n"

        except Exception as e:
            yield json.dumps({"step": "오류 발생", "progress": -1, "error": str(e)}) + "\n"

    return StreamingResponse(
        stream(),
        media_type="application/json",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
