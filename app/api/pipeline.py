# app/api/pipeline.py
import asyncio
import io
import os
import csv
import json
import uuid
import tempfile
from pathlib import Path
from typing import Any, Tuple, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from app.core.preprocess import run_preprocess, filter_df_before_year, get_cpc_path
from app.core.embedding import run_embedding
from app.core.clustering import run_clustering
from app.core.tech_naming import run_tech_naming
from app.core.component_tech_grouper import run_component_grouping, ComponentTechConfig
from app.core.component_tech_naming import generate_component_names_csv

router = APIRouter(tags=["Pipeline"])

# ============================================================
# 환경설정
# ============================================================
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/var/lib/app/outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------- 유틸: 포맷 스니핑 ----------------------
def _looks_like_json(txt: str) -> bool:
    s = txt.lstrip()
    return s.startswith("{") or s.startswith("[")

def _looks_like_jsonl(txt: str) -> bool:
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    sample = lines[: min(5, len(lines))]
    return len(sample) > 1 and all(ln.startswith("{") or ln.startswith("[") for ln in sample)

# ---------------------- 로더: 파일 경로 → DataFrame ----------------------
def _load_table_from_path(
    file_path: Path,
    prefer_jsonl: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    name = file_path.name.lower()
    meta: dict = {"filename": file_path.name}

    # Excel
    if name.endswith((".xlsx", ".xls")):
        df_xlsx = pd.read_excel(str(file_path), engine="openpyxl")
        meta.update({"format": "excel"})
        return df_xlsx, meta

    # Sniff small bytes
    sniff_bytes = file_path.read_bytes()[:100_000]
    sniff_text, sniff_enc = None, None
    for enc in ["utf-8-sig", "utf-8", "latin-1", "cp949", "euc-kr"]:
        try:
            sniff_text = sniff_bytes.decode(enc, errors="strict")
            sniff_enc = enc
            break
        except Exception:
            continue

    if sniff_text:
        if prefer_jsonl and _looks_like_jsonl(sniff_text):
            try:
                df = pd.read_json(str(file_path), lines=True)
                meta.update({"format": "jsonl-sniff", "encoding": sniff_enc})
                return df, meta
            except Exception:
                pass
        if _looks_like_json(sniff_text):
            try:
                with file_path.open("r", encoding=sniff_enc or "utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, list):
                    meta.update({"format": "json-sniff", "encoding": sniff_enc})
                    return pd.DataFrame(obj), meta
                elif isinstance(obj, dict):
                    for k in ("data", "items", "records", "rows", "result"):
                        if k in obj and isinstance(obj[k], list):
                            meta.update({"format": "json-sniff", "encoding": sniff_enc, "root": k})
                            return pd.DataFrame(obj[k]), meta
                    meta.update({"format": "json-sniff", "encoding": sniff_enc, "normalized": True})
                    return pd.json_normalize(obj), meta
            except Exception:
                pass

    # 확장자 직판
    if name.endswith((".jsonl", ".ndjson")):
        df = pd.read_json(str(file_path), lines=True)
        meta.update({"format": "jsonl"})
        return df, meta
    if name.endswith(".json"):
        with file_path.open("r", encoding=sniff_enc or "utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            meta.update({"format": "json"})
            return pd.DataFrame(obj), meta
        elif isinstance(obj, dict):
            for k in ("data", "items", "records", "rows", "result"):
                if k in obj and isinstance(obj[k], list):
                    meta.update({"format": "json", "root": k})
                    return pd.DataFrame(obj[k]), meta
            meta.update({"format": "json", "normalized": True})
            return pd.json_normalize(obj), meta

    # CSV/TSV
    try:
        dialect = csv.Sniffer().sniff(sniff_bytes.decode("utf-8", errors="ignore"), delimiters=",;\t|")
        sep_guess = dialect.delimiter
    except Exception:
        sep_guess = None
    try:
        df = pd.read_csv(str(file_path), sep=sep_guess, engine="c", quotechar='"', escapechar="\\", on_bad_lines="error")
        meta.update({"format": "csv", "sep": sep_guess, "engine": "c"})
        return df, meta
    except Exception:
        pass
    df = pd.read_csv(str(file_path), sep=sep_guess, engine="python", quotechar='"', escapechar="\\", on_bad_lines="skip")
    meta.update({"format": "csv", "sep": sep_guess, "engine": "python", "on_bad_lines": "skip"})
    return df, meta

# ---------------------- 안전 리스트 변환 ----------------------
def _to_str_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    if isinstance(x, pd.Series):
        return x.dropna().astype(str).tolist()
    if isinstance(x, pd.DataFrame):
        return x.astype(str).stack().tolist()
    return [str(x)]

# ---------------------- 업로드 스트리밍 → 임시파일 ----------------------
async def _save_upload_to_tempfile(file: UploadFile) -> Path:
    # 업로드되는 파일 확장자 보존(진단 편의)
    suffix = ""
    if file.filename and "." in file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(tmp.name)
    try:
        # 1MB 청크로 복사(메모리 폭주 방지)
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        tmp.flush()
        tmp.close()
        return tmp_path
    except Exception:
        try:
            tmp.close()
        finally:
            tmp_path.unlink(missing_ok=True)
        raise


@router.post("/pipeline/run")
async def run_pipeline(
    file: UploadFile = File(...),
    cutoff_year: int = Form(2025),
    n_clusters: int = Form(100),
    model_name: str = Form("all-MiniLM-L6-v2"),
    top_n: int = Form(100),
    run_id: str = Form(None),
):
    """
    영구 저장 없이 처리:
      - 업로드는 임시파일에만 저장(요청 끝나면 삭제)
      - 중간 산출물도 메모리/임시파일로만 사용
      - 진행상황을 JSON 라인으로 스트리밍
      - 결과물: OUTPUT_DIR에 CSV 2개 저장
          * 요소기술 네이밍: names_generated_flowagg.csv
          * 구성기술 네이밍: component_tech_names.csv
    """
    rid = run_id or uuid.uuid4().hex
    temp_paths: list[Path] = []

    async def stream():
        try:
            # 0) 업로드 → 임시파일
            yield json.dumps({"step": "파일 저장 시작", "progress": 0}, ensure_ascii=False) + "\n"
            src_path = await _save_upload_to_tempfile(file)
            temp_paths.append(src_path)
            yield json.dumps(
                {"step": "파일 저장 완료", "progress": 4, "meta": {"path": str(src_path), "filename": file.filename}},
                ensure_ascii=False,
            ) + "\n"

            # 1) 파일 로드
            yield json.dumps({"step": "파일 로드 중", "progress": 5}, ensure_ascii=False) + "\n"
            df, meta = await asyncio.to_thread(_load_table_from_path, src_path, True)
            yield json.dumps(
                {"step": "파일 로드 완료", "progress": 10, "meta": {"filename": file.filename, **meta}},
                ensure_ascii=False,
            ) + "\n"

            print(f"✅ 전처리 시작: {len(df):,}개의 데이터")

            # 2) 연도 필터링
            yield json.dumps({"step": "데이터 필터링 시작", "progress": 15}, ensure_ascii=False) + "\n"
            df_year = await asyncio.to_thread(filter_df_before_year, df, int(cutoff_year))

            # 3) 전처리
            yield json.dumps({"step": "데이터 전처리 시작", "progress": 20}, ensure_ascii=False) + "\n"
            df_clean = await asyncio.to_thread(
                run_preprocess,
                df_year,
                int(cutoff_year),
                do_cpc_match=True,
                cpc_csv_path=get_cpc_path(),
            )

            print(f"✅ 전처리 완료: {len(df_clean):,}개의 데이터")

            # 4) 임베딩 (하트비트 포함)
            yield json.dumps({"step": "임베딩 중", "progress": 40}, ensure_ascii=False) + "\n"

            progress_q: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_event_loop()
            last_progress: Optional[tuple[int, int]] = None  # (processed, total)

            def _progress_cb(processed: int, total: int):
                # run_embedding은 to_thread에서 돌기 때문에, 스레드-세이프하게 이벤트루프로 넘겨야 함
                try:
                    loop.call_soon_threadsafe(progress_q.put_nowait, (processed, total))
                except Exception:
                    pass

            # 임베딩 실행
            task_embed = asyncio.create_task(asyncio.to_thread(
                run_embedding,
                df_clean,
                model_name,
                # 체크포인트 사용 시 재개 가능 (문제 있으면 resume=False 권장)
                batch_size=512,
                checkpoint_dir=f"/tmp/emb_ckpt/{rid}",
                resume=True,
                progress_cb=_progress_cb,
            ))

            # 진행 상황 하트비트
            HB_INTERVAL = 2
            while not task_embed.done():
                try:
                    await asyncio.sleep(HB_INTERVAL)
                    while True:
                        last_progress = progress_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass

                if last_progress is not None:
                    processed, total = last_progress
                    pct = 40 + (processed / max(total, 1)) * 30  # 40~70
                    yield json.dumps({
                        "step": "ping",
                        "progress": int(pct),
                        "meta": {"stage": "embedding", "processed": processed, "total": total, "batch_size": 512}
                    }, ensure_ascii=False) + "\n"
                else:
                    yield json.dumps({"step": "ping", "progress": 41}, ensure_ascii=False) + "\n"

            # 임베딩 완료 결과 수집
            df_embed = await task_embed

            # 5) 클러스터링/요약
            yield json.dumps({"step": "클러스터링 및 추세 분석 중", "progress": 70}, ensure_ascii=False) + "\n"
            df_clustered, summary = await asyncio.to_thread(run_clustering, df_embed, n_clusters)
            if not isinstance(summary, dict) or "artifacts" not in summary:
                raise RuntimeError("artifacts 누락")

            # 6) 요소기술 네이밍 (flow-agg) → OUTPUT_DIR/names_generated_flowagg.csv
            yield json.dumps({"step": "기술명 생성 중", "progress": 85}, ensure_ascii=False) + "\n"
            naming_result = await asyncio.to_thread(
                run_tech_naming, None, artifacts=summary["artifacts"], top_n=int(top_n)
            )
            elem_csv_path = naming_result["paths"].get("flowagg_csv", "")

            # 7) 구성기술 묶기 (메모리 전용; 디스크 저장 없음)
            yield json.dumps({"step": "구성기술 묶는 중", "progress": 90}, ensure_ascii=False) + "\n"
            cfg = ComponentTechConfig(
                n_components=int(n_clusters),
                year_col="year",
                embed_col="embedding",
                cluster_col=summary["paths"]["label_col"],
                random_state=42,
            )
            df_component, comp_summary = await asyncio.to_thread(run_component_grouping, df_clustered, cfg)

            # 8) 구성기술 네이밍(전체) → OUTPUT_DIR/component_tech_names.csv
            yield json.dumps({"step": "구성기술 네이밍 중", "progress": 96}, ensure_ascii=False) + "\n"
            comp_csv_path = await asyncio.to_thread(
                generate_component_names_csv,
                df_component,
                label_col="component_tech_id",
                text_cols=("title",),
                output_csv_path=None,   # 내부에서 OUTPUT_DIR 사용
            )

            # 9) 완료(결과 경로 전송)
            keywords = _to_str_list(summary.get("keywords", []))[:100]
            titles = _to_str_list(summary.get("titles", []))[:100]
            yield json.dumps(
                {
                    "step": "완료",
                    "progress": 100,
                    "result": {
                        "outputs": {
                            "element_names_csv": elem_csv_path,
                            "component_names_csv": comp_csv_path,
                        },
                        "summary": {"keywords": keywords, "titles": titles, "paths": summary.get("paths", {})},
                        "run_id": rid,
                    },
                },
                ensure_ascii=False,
            ) + "\n"

        except Exception as e:
            yield json.dumps({"step": "오류 발생", "progress": -1, "error": str(e)}, ensure_ascii=False) + "\n"
        finally:
            # 임시파일 정리
            for p in temp_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

    return StreamingResponse(
        stream(),
        media_type="application/json",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
