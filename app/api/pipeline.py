# ============================================================
# app/api/pipeline.py — ETA + Heartbeat + PipelineState 통합 버전
# ============================================================

import asyncio
import os
import csv
import json
import traceback
import uuid
import tempfile
import base64
from io import StringIO
from pathlib import Path
from typing import Any, Tuple, Optional, List, Dict

import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse

# ---------------------- 코어 로직 ----------------------
from app.core.preprocess import run_preprocess, filter_df_before_year
from app.core.embedding import run_embedding
from app.core.clustering import run_clustering
from app.core.tech_naming import run_tech_naming, _panel_from_artifacts_edges
from app.core.component_tech_grouper import run_component_grouping, ComponentTechConfig
from app.core.component_tech_naming import generate_component_names_csv

# ---------------------- ETA & Progress ----------------------
from app.core.pipeline_steps.state import PipelineState
from app.core.pipeline_steps.heartbeats import make_heartbeat
from app.core.pipeline_steps.eta_format import format_eta

router = APIRouter(tags=["Pipeline"])

# ============================================================
# 환경설정
# ============================================================
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/var/lib/app/outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------- 유틸: 포맷 판별 ----------------------
def _looks_like_json(txt: str) -> bool:
    s = txt.lstrip()
    return s.startswith("{") or s.startswith("[")


def _looks_like_jsonl(txt: str) -> bool:
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    sample = lines[: min(5, len(lines))]
    return len(sample) > 1 and all(ln.startswith("{") or ln.startswith("[") for ln in sample)


# ---------------------- 파일 로더 ----------------------
def _load_table_from_path(file_path: Path, prefer_jsonl: bool = True) -> Tuple[pd.DataFrame, dict]:
    name = file_path.name.lower()
    meta: dict = {"filename": file_path.name}

    # Excel
    if name.endswith((".xlsx", ".xls")):
        df_xlsx = pd.read_excel(str(file_path), engine="openpyxl")
        meta.update({"format": "excel"})
        return df_xlsx, meta

    sniff_bytes = file_path.read_bytes()[:100_000]

    sniff_text, sniff_enc = None, None
    for enc in ["utf-8-sig", "utf-8", "latin-1", "cp949", "euc-kr"]:
        try:
            sniff_text = sniff_bytes.decode(enc, errors="strict")
            sniff_enc = enc
            break
        except Exception:
            continue

    # ----- JSONL 테스트 -----
    if sniff_text:
        if prefer_jsonl and _looks_like_jsonl(sniff_text):
            try:
                df = pd.read_json(str(file_path), lines=True)
                meta.update({"format": "jsonl-sniff", "encoding": sniff_enc})
                return df, meta
            except Exception:
                pass

    # ----- JSON 테스트 -----
    if sniff_text and _looks_like_json(sniff_text):
        try:
            with file_path.open("r", encoding=sniff_enc or "utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                meta.update({"format": "json-sniff"})
                return pd.DataFrame(obj), meta
            if isinstance(obj, dict):
                for key in ("data", "items", "records", "rows", "result"):
                    if key in obj and isinstance(obj[key], list):
                        meta.update({"format": "json-sniff", "root": key})
                        return pd.DataFrame(obj[key]), meta
                meta.update({"format": "json-sniff", "normalized": True})
                return pd.json_normalize(obj), meta
        except Exception:
            pass

    # JSON 확장자 직접 처리
    if name.endswith(".jsonl"):
        df = pd.read_json(str(file_path), lines=True)
        meta.update({"format": "jsonl"})
        return df, meta

    if name.endswith(".json"):
        with file_path.open("r", encoding=sniff_enc or "utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            meta.update({"format": "json"})
            return pd.DataFrame(obj), meta
        if isinstance(obj, dict):
            for key in ("data", "items", "records", "rows", "result"):
                if key in obj and isinstance(obj[key], list):
                    meta.update({"format": "json", "root": key})
                    return pd.DataFrame(obj[key]), meta
            meta.update({"format": "json", "normalized": True})
            return pd.json_normalize(obj), meta

    # CSV
    try:
        dialect = csv.Sniffer().sniff(sniff_bytes.decode("utf-8", errors="ignore"), delimiters=",;\t|")
        sep_guess = dialect.delimiter
    except Exception:
        sep_guess = None

    try:
        df = pd.read_csv(
            str(file_path),
            sep=sep_guess,
            engine="c",
            quotechar='"',
            escapechar="\\",
            on_bad_lines="error",
        )
        meta.update({"format": "csv", "sep": sep_guess})
        return df, meta
    except Exception:
        pass

    df = pd.read_csv(
        str(file_path),
        sep=sep_guess,
        engine="python",
        quotechar='"',
        escapechar="\\",
        on_bad_lines="skip",
    )
    meta.update({"format": "csv", "sep": sep_guess, "engine": "python"})
    return df, meta


# ---------------------- base64 ----------------------
def _b64(text: str | None) -> str:
    if not text:
        return ""
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


# ---------------------- 업로드 저장 ----------------------
async def _save_upload_to_tempfile(file: UploadFile) -> Path:
    suffix = ""
    if file.filename and "." in file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[-1]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(tmp.name)

    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        tmp.flush()
        tmp.close()
        return tmp_path
    except Exception:
        tmp.close()
        tmp_path.unlink(missing_ok=True)
        raise

# ============================================================
# 메인 파이프라인 엔드포인트
# ============================================================

@router.post("/pipeline/run")
async def run_pipeline(
    file: UploadFile = File(...),
    cutoff_year: int = Form(2026),
    n_clusters: int = Form(100),
    model_name: str = Form("all-MiniLM-L6-v2"),
    top_n: int = Form(100),
    run_id: str = Form(None)
):
    rid = run_id or uuid.uuid4().hex
    temp_paths: list[Path] = []

    # ETA + 진행도 통합 상태
    state = PipelineState()

    def j(obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    async def stream():
        try:
            # ============================================================
            # 0) 파일 업로드
            # ============================================================
            state.start_step("upload", "element")
            yield j(make_heartbeat(
                phase="element",
                step_name="upload",
                progress=0.0,
                element_progress=state.progress.element_progress(),
                component_progress=state.progress.component_progress(),
                overall_progress=state.progress.overall_progress(),
                eta_seconds=None,
                meta={"step_label": "파일 저장 시작"}
            ))

            src_path = await _save_upload_to_tempfile(file)
            temp_paths.append(src_path)

            state.update_progress(1.0)
            yield j(make_heartbeat(
                phase="element",
                step_name="upload",
                progress=1.0,
                element_progress=state.progress.element_progress(),
                component_progress=state.progress.component_progress(),
                overall_progress=state.progress.overall_progress(),
                eta_seconds=0,
                meta={"step_label": "파일 저장 완료", "filename": file.filename}
            ))

            # ============================================================
            # 1) 파일 로드
            # ============================================================
            state.start_step("load", "element")
            yield j(make_heartbeat(
                phase="element",
                step_name="load",
                progress=0.0,
                element_progress=state.progress.element_progress(),
                component_progress=state.progress.component_progress(),
                overall_progress=state.progress.overall_progress(),
                eta_seconds=None,
                meta={"step_label": "파일 로드 중"}
            ))

            df, meta = await asyncio.to_thread(_load_table_from_path, src_path, True)

            state.update_progress(1.0)
            yield j(make_heartbeat(
                phase="element",
                step_name="load",
                progress=1.0,
                element_progress=state.progress.element_progress(),
                component_progress=state.progress.component_progress(),
                overall_progress=state.progress.overall_progress(),
                eta_seconds=0,
                meta={"step_label": "파일 로드 완료", **meta}
            ))

            # ============================================================
            # 2) 연도 필터
            # ============================================================
            state.start_step("year_filter", "element")
            yield j(make_heartbeat(
                phase="element",
                step_name="year_filter",
                progress=0.0,
                element_progress=state.progress.element_progress(),
                component_progress=state.progress.component_progress(),
                overall_progress=state.progress.overall_progress(),
                eta_seconds=None,
                meta={"step_label": "연도 필터링 시작"}
            ))

            df_year = await asyncio.to_thread(filter_df_before_year, df, int(cutoff_year))

            state.update_progress(1.0)
            yield j(make_heartbeat(
                phase="element",
                step_name="year_filter",
                progress=1.0,
                element_progress=state.progress.element_progress(),
                component_progress=state.progress.component_progress(),
                overall_progress=state.progress.overall_progress(),
                eta_seconds=0,
                meta={"step_label": "연도 필터링 완료", "rows_after_filter": len(df_year)}
            ))

            # ============================================================
            # 3) 전처리 (heartbeat + ETA)
            # ============================================================
            state.start_step("preprocess", "element")

            yield j(make_heartbeat(
                phase="element",
                step_name="preprocess",
                progress=0.0,
                element_progress=state.progress.element_progress(),
                component_progress=state.progress.component_progress(),
                overall_progress=state.progress.overall_progress(),
                eta_seconds=None,
                meta={"step_label": "데이터 전처리 시작"}
            ))

            pre_q: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def _pre_cb(processed: int, total: int, stage: str):
                try:
                    loop.call_soon_threadsafe(pre_q.put_nowait, (processed, total, stage))
                except:
                    pass

            task_pre = asyncio.create_task(asyncio.to_thread(
                run_preprocess,
                df_year,
                int(cutoff_year),
                progress_cb=_pre_cb
            ))

            last_pre = None

            while not task_pre.done():
                await asyncio.sleep(1.0)

                try:
                    while True:
                        last_pre = pre_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass

                if last_pre is None:
                    yield j(make_heartbeat(
                        phase="element",
                        step_name="preprocess",
                        progress=state.progress._el_progress["preprocess"],
                        element_progress=state.progress.element_progress(),
                        component_progress=state.progress.component_progress(),
                        overall_progress=state.progress.overall_progress(),
                        eta_seconds=None,
                        meta={"step_label": "전처리 하트비트"}
                    ))
                    continue

                processed, total, stage = last_pre

                frac = processed / max(total, 1)
                state.update_progress(frac)
                eta_sec = state.timer.eta_seconds(processed, total)

                yield j(make_heartbeat(
                    phase="element",
                    step_name="preprocess",
                    progress=frac,
                    element_progress=state.progress.element_progress(),
                    component_progress=state.progress.component_progress(),
                    overall_progress=state.progress.overall_progress(),
                    eta_seconds=eta_sec,
                    meta={"stage": stage, "processed": processed, "total": total}
                ))

            df_clean = await task_pre
            state.update_progress(1.0)

            yield j(make_heartbeat(
                phase="element",
                step_name="preprocess",
                progress=1.0,
                element_progress=state.progress.element_progress(),
                component_progress=state.progress.component_progress(),
                overall_progress=state.progress.overall_progress(),
                eta_seconds=0,
                meta={"step_label": "전처리 완료", "rows": len(df_clean)}
            ))

            # ============================================================
            # 4) 임베딩 (heartbeat + ETA)
            # ============================================================
            state.start_step("embedding", "element")

            yield j(make_heartbeat(
                phase="element",
                step_name="embedding",
                progress=0.0,
                element_progress=state.progress.element_progress(),
                component_progress=state.progress.component_progress(),
                overall_progress=state.progress.overall_progress(),
                eta_seconds=None,
                meta={"step_label": "임베딩 시작"}
            ))

            emb_q: asyncio.Queue = asyncio.Queue()
            last_emb = None

            def _emb_cb(processed: int, total: int):
                try:
                    loop.call_soon_threadsafe(emb_q.put_nowait, (processed, total))
                except:
                    pass

            task_embed = asyncio.create_task(asyncio.to_thread(
                run_embedding,
                df_clean,
                model_name,
                batch_size=512,
                checkpoint_dir=f"/tmp/emb_ckpt/{rid}",
                resume=True,
                progress_cb=_emb_cb
            ))

            while not task_embed.done():
                await asyncio.sleep(1.0)

                try:
                    while True:
                        last_emb = emb_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass

                if last_emb is None:
                    yield j(make_heartbeat(
                        phase="element",
                        step_name="embedding",
                        progress=state.progress._el_progress["embedding"],
                        element_progress=state.progress.element_progress(),
                        component_progress=state.progress.component_progress(),
                        overall_progress=state.progress.overall_progress(),
                        eta_seconds=None,
                        meta={"step_label": "임베딩 하트비트"}
                    ))
                    continue

                processed, total = last_emb

                frac = processed / max(total, 1)
                state.update_progress(frac)
                eta_sec = state.timer.eta_seconds(processed, total)

                yield j(make_heartbeat(
                    phase="element",
                    step_name="embedding",
                    progress=frac,
                    element_progress=state.progress.element_progress(),
                    component_progress=state.progress.component_progress(),
                    overall_progress=state.progress.overall_progress(),
                    eta_seconds=eta_sec,
                    meta={"processed": processed, "total": total, "batch_size": 512}
                ))

            df_embed = await task_embed
            state.update_progress(1.0)

            yield j(make_heartbeat(
                phase="element",
                step_name="embedding",
                progress=1.0,
                element_progress=state.progress.element_progress(),
                component_progress=state.progress.component_progress(),
                overall_progress=state.progress.overall_progress(),
                eta_seconds=0,
                meta={"step_label": "임베딩 완료"}
            ))

            # ============================================================
            # 5) 클러스터링
            # ============================================================
            state.start_step("clustering", "element")

            yield j(make_heartbeat(
                phase="element",
                step_name="clustering",
                progress=0.0,
                element_progress=state.progress.element_progress(),
                component_progress=state.progress.component_progress(),
                overall_progress=state.progress.overall_progress(),
                eta_seconds=None,
                meta={"step_label": "클러스터링 시작"}
            ))

            # 클러스터링 실행
            df_clustered, summary = await asyncio.to_thread(run_clustering, df_embed, n_clusters)

            if not isinstance(summary, dict) or "artifacts" not in summary:
                raise RuntimeError("artifacts 누락")

            state.update_progress(1.0)
            yield j(make_heartbeat(
                phase="element",
                step_name="clustering",
                progress=1.0,
                element_progress=state.progress.element_progress(),
                component_progress=state.progress.component_progress(),
                overall_progress=state.progress.overall_progress(),
                eta_seconds=0,
                meta={"step_label": "클러스터링 완료"}
            ))

            # ============================================================
            # 6) 요소기술 네이밍
            # ============================================================
            state.start_step("tech_naming", "element")

            yield j(make_heartbeat(
                phase="element",
                step_name="tech_naming",
                progress=0.0,
                element_progress=state.progress.element_progress(),
                component_progress=state.progress.component_progress(),
                overall_progress=state.progress.overall_progress(),
                eta_seconds=None,
                meta={"step_label": "요소기술 네이밍 시작"}
            ))

            naming_result = await asyncio.to_thread(
                run_tech_naming,
                None,
                artifacts=summary["artifacts"],
                top_n=int(top_n)
            )

            naming_df: pd.DataFrame = naming_result.get("df", pd.DataFrame())
            elem_csv_text: str = naming_result.get("csv_text", "")

            # --- Top10 docs 정보 생성 ---
            panel = _panel_from_artifacts_edges(summary["artifacts"].get("flow_edges_df", pd.DataFrame()))
            if panel.empty:
                panel = pd.DataFrame(columns=["flow_id", "docs"])
            if "docs" not in panel.columns:
                panel["docs"] = 1

            flow_docs = (
                panel.groupby("flow_id", as_index=False)["docs"]
                .sum()
                .rename(columns={"flow_id": "id"})
            )

            elem_top = (
                naming_df.rename(columns={"flow_id": "id"})
                .merge(flow_docs, on="id", how="left")
                .sort_values("docs", ascending=False)
                .head(10)
            )

            elem_top10_payload = [
                {
                    "id": int(r.id),
                    "name_en": str(r.tech_name_en or ""),
                    "name_ko": str(r.tech_name_ko or ""),
                    "docs": int(r.docs or 0),
                }
                for r in elem_top.itertuples(index=False)
            ]

            elem_titles_en10 = [x["name_en"] for x in elem_top10_payload]
            elem_titles_ko10 = [x["name_ko"] for x in elem_top10_payload]

            state.update_progress(1.0)

            yield j({
                **make_heartbeat(
                    phase="element",
                    step_name="tech_naming",
                    progress=1.0,
                    element_progress=state.progress.element_progress(),
                    component_progress=state.progress.component_progress(),
                    overall_progress=state.progress.overall_progress(),
                    eta_seconds=0,
                    meta={"step_label": "요소기술 결과 전송"}
                ),
                "step": "element_result",
                "result": {
                    "outputs": {
                        "files": [
                            {
                                "filename": "names_generated_flowagg.csv",
                                "mime": "text/csv; charset=utf-8",
                                "encoding": "base64",
                                "content": _b64(elem_csv_text),
                            }
                        ]
                    },
                    "summary": {
                        "titles_en": elem_titles_en10,
                        "titles_ko": elem_titles_ko10,
                        "top10": elem_top10_payload,
                    },
                    "run_id": rid,
                }
            })

            # ============================================================
            # ----------- 구성기술 컴포넌트 단계 시작 ---------------------
            # ============================================================

            # 7) 구성기술 묶기
            state.start_step("component_grouping", "component")
            yield j(make_heartbeat(
                phase="component",
                step_name="component_grouping",
                progress=0.0,
                element_progress=state.progress.element_progress(),
                component_progress=state.progress.component_progress(),
                overall_progress=state.progress.overall_progress(),
                eta_seconds=None,
                meta={"step_label": "구성기술 묶기 시작"}
            ))

            cfg = ComponentTechConfig(
                n_components=int(n_clusters),
                year_col="year",
                embed_col="embedding",
                cluster_col=summary["paths"]["label_col"],
                random_state=42
            )

            df_component, comp_summary = await asyncio.to_thread(
                run_component_grouping, df_clustered, cfg
            )

            state.update_progress(1.0)
            yield j(make_heartbeat(
                phase="component",
                step_name="component_grouping",
                progress=1.0,
                element_progress=state.progress.element_progress(),
                component_progress=state.progress.component_progress(),
                overall_progress=state.progress.overall_progress(),
                eta_seconds=0,
                meta={"step_label": "구성기술 묶기 완료"}
            ))

            # ============================================================
            # 8) 구성기술 네이밍
            # ============================================================
            state.start_step("component_naming", "component")

            yield j(make_heartbeat(
                phase="component",
                step_name="component_naming",
                progress=0.0,
                element_progress=state.progress.element_progress(),
                component_progress=state.progress.component_progress(),
                overall_progress=state.progress.overall_progress(),
                eta_seconds=None,
                meta={"step_label": "구성기술 네이밍 시작"}
            ))

            comp_csv_text = await asyncio.to_thread(
                generate_component_names_csv,
                df_component,
                label_col="component_tech_id",
                text_cols=("title",),
                output_csv_path=None
            )

            comp_names_df = pd.read_csv(StringIO(comp_csv_text))

            comp_docs = (
                df_component.groupby("component_tech_id", as_index=False)
                .size()
                .rename(columns={"size": "docs", "component_tech_id": "id"})
            )

            comp_top = (
                comp_names_df.rename(columns={"component_tech_id": "id"})
                .merge(comp_docs, on="id", how="left")
                .sort_values("docs", ascending=False)
                .head(10)
            )

            comp_top10_payload = [
                {
                    "id": int(r.id),
                    "name_en": str(r.tech_name_en or ""),
                    "name_ko": str(r.tech_name_ko or ""),
                    "docs": int(r.docs or 0),
                }
                for r in comp_top.itertuples(index=False)
            ]

            component_titles_top10 = [x["name_en"] for x in comp_top10_payload]

            state.update_progress(1.0)

            yield j({
                **make_heartbeat(
                    phase="component",
                    step_name="component_naming",
                    progress=1.0,
                    element_progress=state.progress.element_progress(),
                    component_progress=state.progress.component_progress(),
                    overall_progress=state.progress.overall_progress(),
                    eta_seconds=0,
                    meta={"step_label": "구성기술 네이밍 완료"}
                ),
                "step": "component_result",
                "result": {
                    "outputs": {
                        "files": [
                            {
                                "filename": "component_tech_names.csv",
                                "mime": "text/csv; charset=utf-8",
                                "encoding": "base64",
                                "content": _b64(comp_csv_text),
                            }
                        ]
                    },
                    "summary": {
                        "titles": component_titles_top10,
                        "top10": comp_top10_payload,
                    },
                    "run_id": rid,
                }
            })

            # ============================================================
            # 9) 시각화 데이터 생성
            # ============================================================
            state.start_step("visualize", "component")

            yield j(make_heartbeat(
                phase="component",
                step_name="visualize",
                progress=0.0,
                element_progress=state.progress.element_progress(),
                component_progress=state.progress.component_progress(),
                overall_progress=state.progress.overall_progress(),
                eta_seconds=None,
                meta={"step_label": "계보도 데이터 생성 시작"}
            ))

            # Elements
            elements_all = []
            if not naming_df.empty:
                ndf = naming_df.rename(columns={"flow_id": "id"}).merge(flow_docs, on="id", how="left")
                for r in ndf.itertuples(index=False):
                    elements_all.append({
                        "id": int(r.id),
                        "name_en": str(r.tech_name_en or ""),
                        "name_ko": str(r.tech_name_ko or ""),
                        "docs": int(r.docs or 0),
                    })

            # Components
            components_all = []
            if not comp_names_df.empty:
                cdf = comp_names_df.rename(columns={"component_tech_id": "id"}).merge(comp_docs, on="id", how="left")
                for r in cdf.itertuples(index=False):
                    components_all.append({
                        "id": int(r.id),
                        "name_en": str(r.tech_name_en or ""),
                        "name_ko": str(r.tech_name_ko or ""),
                        "docs": int(r.docs or 0),
                    })

            # Edge
            orig_cluster_col = summary["paths"]["label_col"]
            if orig_cluster_col not in df_component.columns:
                fallback = next(
                    (c for c in df_component.columns if c.lower() in ["cluster_id", "orig_cluster_id", "source_cluster_id"]),
                    None
                )
                orig_cluster_col = fallback or orig_cluster_col

            edges_all = []
            if orig_cluster_col in df_component.columns:
                cross = (
                    df_component.groupby([orig_cluster_col, "component_tech_id"], as_index=False)
                    .size()
                    .rename(columns={"size": "weight"})
                )
                for r in cross.itertuples(index=False):
                    edges_all.append({
                        "from_element_id": int(getattr(r, orig_cluster_col)),
                        "to_component_id": int(r.component_tech_id),
                        "weight": int(r.weight),
                    })

            # 그래프 유틸 함수
            def induce_connected_only(
                    elements: List[Dict],
                    components: List[Dict],
                    edges: List[Dict],
                    *,
                    min_weight: int = 0,
                    top_edges_per_element: int | None = None,
                    top_edges_per_component: int | None = None,
            ):
                from collections import defaultdict

                # 0) 가중치 필터
                E = [e for e in edges if int(e.get("weight", 0)) >= int(min_weight)]

                # 1) 요소기술 기준 상위 엣지 제한(옵션)
                if top_edges_per_element:
                    by_elem = defaultdict(list)
                    for e in E:
                        by_elem[e["from_element_id"]].append(e)

                    E = [
                        ee
                        for _, lst in by_elem.items()
                        for ee in sorted(lst, key=lambda x: x.get("weight", 0), reverse=True)[:top_edges_per_element]
                    ]

                # 2) 구성기술 기준 상위 엣지 제한(옵션)
                if top_edges_per_component:
                    by_comp = defaultdict(list)
                    for e in E:
                        by_comp[e["to_component_id"]].append(e)

                    E = [
                        ee
                        for _, lst in by_comp.items()
                        for ee in sorted(lst, key=lambda x: x.get("weight", 0), reverse=True)[:top_edges_per_component]
                    ]

                elem_ids = {e["from_element_id"] for e in E}
                comp_ids = {e["to_component_id"] for e in E}

                # 필터된 요소기술 노드
                elements_out = [n for n in elements if n["id"] in elem_ids]
                # 필터된 구성기술 노드
                components_out = [n for n in components if n["id"] in comp_ids]

                elem_idset = {n["id"] for n in elements_out}
                comp_idset = {n["id"] for n in components_out}

                # 필터된 엣지
                edges_out = [
                    e for e in E
                    if e["from_element_id"] in elem_idset and e["to_component_id"] in comp_idset
                ]

                # 정렬 (선택)
                elements_out.sort(key=lambda x: (-int(x.get("docs", 0)), x["id"]))
                components_out.sort(key=lambda x: (-int(x.get("docs", 0)), x["id"]))
                edges_out.sort(
                    key=lambda x: (
                        -int(x.get("weight", 0)),
                        x["from_element_id"],
                        x["to_component_id"],
                    )
                )

                return elements_out, components_out, edges_out

            # 연결된 그래프만 필터링
            elements_v, components_v, edges_v = induce_connected_only(
                elements_all,
                components_all,
                edges_all,
                min_weight=0,
                top_edges_per_element=3,
                top_edges_per_component=None
            )

            state.update_progress(1.0)

            yield j({
                **make_heartbeat(
                    phase="component",
                    step_name="visualize",
                    progress=1.0,
                    element_progress=state.progress.element_progress(),
                    component_progress=state.progress.component_progress(),
                    overall_progress=state.progress.overall_progress(),
                    eta_seconds=0,
                    meta={"step_label": "계보도 데이터 전송"}
                ),
                "step": "visualize",
                "result": {
                    "summary": {
                        "components": components_v,
                        "elements": elements_v,
                        "edges": edges_v,
                    }
                },
                "run_id": rid,
            })

        # ============================================================
        # 에러 처리
        # ============================================================
        except Exception as e:
            tb = traceback.format_exc()
            yield j({
                "phase": state.phase,
                "step": "error",
                "step_label": "오류 발생",
                "step_progress": 0,
                "element_progress": int(state.progress.element_progress() * 100),
                "component_progress": int(state.progress.component_progress() * 100),
                "overall_progress": int(state.progress.overall_progress() * 100),
                "error": str(e),
                "traceback": tb[-1000:],
                "progress": -1
            })

        # ============================================================
        # 스트림 종료
        # ============================================================
        finally:
            try:
                yield j({
                    "phase": "component",
                    "step": "stream-close",
                    "step_label": "stream-close",
                    "step_progress": 100,
                    "element_progress": int(state.progress.element_progress() * 100),
                    "component_progress": int(state.progress.component_progress() * 100),
                    "overall_progress": int(state.progress.overall_progress() * 100),
                    "progress": -2
                })
            except:
                pass

            for p in temp_paths:
                try:
                    p.unlink(missing_ok=True)
                except:
                    pass

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
