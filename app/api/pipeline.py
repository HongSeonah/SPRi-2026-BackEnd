# app/api/pipeline.py
import asyncio
import os
import csv
import json
import traceback
import uuid
import tempfile
import base64
import io
from pathlib import Path
from typing import Any, Tuple, Optional, List

import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from app.core.preprocess import run_preprocess, filter_df_before_year
from app.core.embedding import run_embedding
from app.core.clustering import run_clustering
from app.core.tech_naming import run_tech_naming, _panel_from_artifacts_edges
from app.core.component_tech_grouper import run_component_grouping, ComponentTechConfig
from app.core.component_tech_naming import generate_component_names_csv

router = APIRouter(tags=["Pipeline"])

# ============================================================
# 환경설정
# ============================================================
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/var/lib/app/outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===================== 진행도 트래커 =========================
class ProgressTracker:
    """
    섹션(element, component)별 스텝 진행도/가중치를 통해
      - element_progress (0~1)
      - component_progress (0~1)
      - overall_progress (0~1)
    를 계산한다.
    각 step은 [0,1] 구간의 세부 진행도를 받는다.
    """

    # 전체 진행도에서 섹션 가중치
    ELEMENT_WEIGHT = 0.85
    COMPONENT_WEIGHT = 0.15

    # 섹션별 스텝 가중치 (합=1.0)
    ELEMENT_STEPS = [
        ("upload", 0.05),
        ("load", 0.05),
        ("year_filter", 0.10),
        ("preprocess", 0.20),
        ("embedding", 0.35),
        ("clustering", 0.15),
        ("tech_naming", 0.10),
    ]
    # 구성기술은 결과 전송(component_result)와 시각화(visualize)를 분리
    COMPONENT_STEPS = [
        ("component_grouping", 0.30),
        ("component_naming", 0.30),
        ("component_result", 0.10),
        ("visualize", 0.30),
    ]

    def __init__(self):
        self._el_progress = {name: 0.0 for name, _ in self.ELEMENT_STEPS}
        self._co_progress = {name: 0.0 for name, _ in self.COMPONENT_STEPS}

        # BUGFIX: 실제 weight 변수 사용
        self._el_weights = {name: weight for name, weight in self.ELEMENT_STEPS}
        self._co_weights = {name: weight for name, weight in self.COMPONENT_STEPS}

    # 현재 단계(step_name)의 진행도 갱신 (0~1)
    def update_step(self, step_name: str, frac: float):
        frac = max(0.0, min(1.0, float(frac)))
        if step_name in self._el_progress:
            self._el_progress[step_name] = frac
        elif step_name in self._co_progress:
            self._co_progress[step_name] = frac

    def _weighted_sum(self, progress_map, weight_map) -> float:
        return sum(progress_map[s] * weight_map[s] for s in progress_map)

    def element_progress(self) -> float:
        return self._weighted_sum(self._el_progress, self._el_weights)

    def component_progress(self) -> float:
        return self._weighted_sum(self._co_progress, self._co_weights)

    def overall_progress(self) -> float:
        return (
            self.ELEMENT_WEIGHT * self.element_progress()
            + self.COMPONENT_WEIGHT * self.component_progress()
        )

    def pack(self, current_step: str, phase: str, step_progress: float) -> dict:
        el = self.element_progress()
        co = self.component_progress()
        ov = self.overall_progress()
        return {
            "phase": phase,
            "step": current_step,
            "step_progress": int(round(step_progress * 100)),
            "element_progress": int(round(el * 100)),
            "component_progress": int(round(co * 100)),
            "overall_progress": int(round(ov * 100)),
            # 하위 호환 (기존 progress 필드)
            "progress": int(round(ov * 100)),
        }


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

# ---------------------- base64 유틸 ----------------------
def _b64(text: str | None) -> str:
    if not text:
        return ""
    return base64.b64encode(text.encode("utf-8")).decode("ascii")

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
    cutoff_year: int = Form(2026),
    n_clusters: int = Form(100),
    model_name: str = Form("all-MiniLM-L6-v2"),
    top_n: int = Form(100),
    run_id: str = Form(None),
):
    """
    스트림 순서:
      ① element_result  : 요소기술 CSV + titles_en/titles_ko(10)
      ② component_result: 구성기술 CSV + titles_en/titles_ko(10)
      ③ visualize       : 계보도 데이터(components/elements/edges)
    """
    rid = run_id or uuid.uuid4().hex
    temp_paths: list[Path] = []
    tracker = ProgressTracker()  # 진행도 트래커

    def j(obj):
        """JSON 한 줄 스트림 포맷"""
        return json.dumps(obj, ensure_ascii=False) + "\n"

    async def stream():
        try:
            # 0) 업로드 → 임시파일
            tracker.update_step("upload", 0.0)
            yield j({**tracker.pack("upload", "element", 0.0), "step_label": "파일 저장 시작"})
            src_path = await _save_upload_to_tempfile(file)
            temp_paths.append(src_path)
            tracker.update_step("upload", 1.0)
            yield j({
                **tracker.pack("upload", "element", 1.0),
                "step_label": "파일 저장 완료",
                "meta": {"path": str(src_path), "filename": file.filename}
            })

            # 1) 파일 로드
            tracker.update_step("load", 0.0)
            yield j({**tracker.pack("load", "element", 0.0), "step_label": "파일 로드 중"})
            df, meta = await asyncio.to_thread(_load_table_from_path, src_path, True)
            tracker.update_step("load", 1.0)
            yield j({
                **tracker.pack("load", "element", 1.0),
                "step_label": "파일 로드 완료",
                "meta": {"filename": file.filename, **meta}
            })

            # 2) 연도 필터링
            tracker.update_step("year_filter", 0.0)
            yield j({**tracker.pack("year_filter", "element", 0.0), "step_label": "데이터 연도 필터링 시작"})
            df_year = await asyncio.to_thread(filter_df_before_year, df, int(cutoff_year))
            tracker.update_step("year_filter", 1.0)
            yield j({**tracker.pack("year_filter", "element", 1.0), "step_label": "연도 필터링 완료",
                     "meta": {"rows_after_filter": len(df_year)}})

            # 3) 전처리 (콜백 + 하트비트)
            tracker.update_step("preprocess", 0.0)
            yield j({**tracker.pack("preprocess", "element", 0.0), "step_label": "데이터 전처리 시작"})

            pre_q: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def _pre_cb(processed: int, total: int, stage: str):
                try:
                    loop.call_soon_threadsafe(pre_q.put_nowait, (processed, total, stage))
                except Exception:
                    pass

            task_pre = asyncio.create_task(asyncio.to_thread(
                run_preprocess,
                df_year,
                int(cutoff_year),
                progress_cb=_pre_cb,  # 콜백만 유지
            ))

            HB_INTERVAL = 2
            last_pre: Optional[tuple[int, int, str]] = None
            while not task_pre.done():
                try:
                    await asyncio.sleep(HB_INTERVAL)
                    while True:
                        last_pre = pre_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass

                if last_pre is not None:
                    proc, tot, stage = last_pre
                    frac = (proc / max(tot, 1))
                    tracker.update_step("preprocess", frac)
                    yield j({
                        **tracker.pack("preprocess", "element", frac),
                        "step_label": "데이터 전처리 진행 중",
                        "meta": {"stage": stage, "processed": proc, "total": tot}
                    })
                else:
                    yield j({**tracker.pack("preprocess", "element", tracker._el_progress["preprocess"]),
                             "step_label": "데이터 전처리 하트비트", "meta": {"stage": "preprocess_idle"}})

            df_clean = await task_pre
            tracker.update_step("preprocess", 1.0)
            yield j({**tracker.pack("preprocess", "element", 1.0),
                     "step_label": "데이터 전처리 완료",
                     "meta": {"rows": len(df_clean)}})

            # 4) 임베딩 (하트비트 포함)
            tracker.update_step("embedding", 0.0)
            yield j({**tracker.pack("embedding", "element", 0.0), "step_label": "임베딩 시작"})
            progress_q: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_event_loop()
            last_progress: Optional[tuple[int, int]] = None

            def _progress_cb(processed: int, total: int):
                try:
                    loop.call_soon_threadsafe(progress_q.put_nowait, (processed, total))
                except Exception:
                    pass

            task_embed = asyncio.create_task(asyncio.to_thread(
                run_embedding,
                df_clean,
                model_name,
                batch_size=512,
                checkpoint_dir=f"/tmp/emb_ckpt/{rid}",
                resume=True,
                progress_cb=_progress_cb,
            ))

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
                    frac = (processed / max(total, 1))
                    tracker.update_step("embedding", frac)
                    yield j({
                        **tracker.pack("embedding", "element", frac),
                        "step_label": "임베딩 진행 중",
                        "meta": {
                            "stage": "embedding",
                            "processed": processed,
                            "total": total,
                            "batch_size": 512
                        }
                    })
                else:
                    yield j({**tracker.pack("embedding", "element", tracker._el_progress["embedding"]),
                             "step_label": "임베딩 하트비트"})

            # 임베딩 완료
            df_embed = await task_embed
            tracker.update_step("embedding", 1.0)
            yield j({**tracker.pack("embedding", "element", 1.0), "step_label": "임베딩 완료"})

            # 5) 클러스터링/요약
            tracker.update_step("clustering", 0.0)
            yield j({**tracker.pack("clustering", "element", 0.0), "step_label": "클러스터링 및 추세 분석 시작"})
            df_clustered, summary = await asyncio.to_thread(run_clustering, df_embed, n_clusters)
            if not isinstance(summary, dict) or "artifacts" not in summary:
                raise RuntimeError("artifacts 누락")
            tracker.update_step("clustering", 1.0)
            yield j({**tracker.pack("clustering", "element", 1.0), "step_label": "클러스터링 및 추세 분석 완료"})

            # 6) 요소기술 네이밍 → 완료 즉시 파일/타이틀 송신
            tracker.update_step("tech_naming", 0.0)
            yield j({**tracker.pack("tech_naming", "element", 0.0), "step_label": "요소기술 네이밍 시작"})
            naming_result = await asyncio.to_thread(
                run_tech_naming, None, artifacts=summary["artifacts"], top_n=int(top_n)
            )
            elem_csv_text: str = naming_result.get("csv_text", "")
            naming_df: pd.DataFrame = naming_result.get("df", pd.DataFrame())
            tracker.update_step("tech_naming", 1.0)

            elem_csv_b64 = _b64(elem_csv_text)
            elem_titles_en10: List[str] = []
            elem_titles_ko10: List[str] = []
            if not naming_df.empty:
                if "tech_name_en" in naming_df.columns:
                    elem_titles_en10 = (
                        naming_df["tech_name_en"].fillna("").astype(str).map(str.strip).replace("", pd.NA).dropna().head(10).tolist()
                    )
                if "tech_name_ko" in naming_df.columns:
                    elem_titles_ko10 = (
                        naming_df["tech_name_ko"].fillna("").astype(str).map(str.strip).replace("", pd.NA).dropna().head(10).tolist()
                    )

            # ① 요소기술 결과 이벤트
            yield j({
                **tracker.pack("tech_naming", "element", 1.0),
                "step_label": "요소기술 결과 전송",
                "step": "element_result",
                "result": {
                    "outputs": {
                        "files": [
                            {
                                "filename": "names_generated_flowagg.csv",
                                "mime": "text/csv; charset=utf-8",
                                "encoding": "base64",
                                "content": elem_csv_b64
                            }
                        ]
                    },
                    "summary": {
                        "titles_en": elem_titles_en10,
                        "titles_ko": elem_titles_ko10
                    },
                    "run_id": rid,
                },
            })

            # -------- 여기부터 구성기술 섹션 --------
            # 7) 구성기술 묶기
            tracker.update_step("component_grouping", 0.0)
            yield j({**tracker.pack("component_grouping", "component", 0.0), "step_label": "구성기술 묶는 중 시작"})
            cfg = ComponentTechConfig(
                n_components=int(n_clusters),
                year_col="year",
                embed_col="embedding",
                cluster_col=summary["paths"]["label_col"],
                random_state=42,
            )
            df_component, comp_summary = await asyncio.to_thread(run_component_grouping, df_clustered, cfg)
            tracker.update_step("component_grouping", 1.0)
            yield j({**tracker.pack("component_grouping", "component", 1.0),
                     "step_label": "구성기술 묶기 완료"})

            # 8) 구성기술 네이밍
            tracker.update_step("component_naming", 0.0)
            yield j({**tracker.pack("component_naming", "component", 0.0), "step_label": "구성기술 네이밍 시작"})
            comp_csv_text: str = await asyncio.to_thread(
                generate_component_names_csv,
                df_component,
                label_col="component_tech_id",
                text_cols=("title",),
                output_csv_path=None,
            )
            tracker.update_step("component_naming", 1.0)
            yield j({**tracker.pack("component_naming", "component", 1.0), "step_label": "구성기술 네이밍 완료"})

            # 구성기술 CSV base64 + titles 10개
            comp_csv_b64 = _b64(comp_csv_text)
            comp_df = pd.read_csv(io.StringIO(comp_csv_text)) if comp_csv_text else pd.DataFrame()
            comp_titles_en10: List[str] = []
            comp_titles_ko10: List[str] = []
            if not comp_df.empty:
                if "tech_name_en" in comp_df.columns:
                    comp_titles_en10 = (
                        comp_df["tech_name_en"].fillna("").astype(str).map(str.strip).replace("", pd.NA).dropna().head(10).tolist()
                    )
                if "tech_name_ko" in comp_df.columns:
                    comp_titles_ko10 = (
                        comp_df["tech_name_ko"].fillna("").astype(str).map(str.strip).replace("", pd.NA).dropna().head(10).tolist()
                    )

            # ② 구성기술 결과 이벤트
            tracker.update_step("component_result", 1.0)
            yield j({
                **tracker.pack("component_result", "component", 1.0),
                "step_label": "구성기술 결과 전송",
                "step": "component_result",
                "result": {
                    "outputs": {
                        "files": [
                            {
                                "filename": "component_tech_names.csv",
                                "mime": "text/csv; charset=utf-8",
                                "encoding": "base64",
                                "content": comp_csv_b64
                            }
                        ]
                    },
                    "summary": {
                        "titles_en": comp_titles_en10,
                        "titles_ko": comp_titles_ko10
                    },
                    "run_id": rid,
                },
            })

            # --- 계보도 데이터 구성 ---
            # (0) 유틸: 구성클러스터 컬럼 자동 탐색
            def _resolve_cluster_col(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
                cand = []
                if preferred:
                    cand.append(preferred)
                cand += ["cluster_id", "label", "cluster", "cluster_idx", "kmeans_label"]
                for c in cand:
                    if c in df.columns:
                        return c
                for c in df.columns:
                    if "cluster" in c.lower():
                        return c
                raise KeyError(f"cannot resolve cluster column from df_component.columns={list(df.columns)}")

            # (1) 요소 elements: flow docs 집계 + 이름
            artifacts = summary["artifacts"]
            flow_panel = _panel_from_artifacts_edges(artifacts["flow_edges_df"])
            if "docs" not in flow_panel.columns:
                flow_panel["docs"] = 1
            flow_docs = flow_panel.groupby("flow_id", as_index=False)["docs"].sum()

            elements_payload = []
            if not naming_df.empty:
                exist_cols = [c for c in ["flow_id", "tech_name_en", "tech_name_ko"] if c in naming_df.columns]
                merged = naming_df[exist_cols].merge(flow_docs, on="flow_id", how="left")
                for _, r in merged.iterrows():
                    elements_payload.append({
                        "id": int(r["flow_id"]),
                        "name_en": str(r.get("tech_name_en", "") or ""),
                        "name_ko": str(r.get("tech_name_ko", "") or ""),
                        "docs": int(r.get("docs", 0) or 0),
                    })

            # (2) 구성 components: component_tech_id 문서수 + 이름
            comp_docs = (
                df_component.groupby("component_tech_id", as_index=False)
                .size()
                .rename(columns={"size": "docs"})
            )
            components_payload = []
            if not comp_df.empty:
                comp_nodes = comp_docs.merge(
                    comp_df[["component_tech_id", "tech_name_en", "tech_name_ko"]],
                    on="component_tech_id", how="left"
                )
            else:
                comp_nodes = comp_docs.assign(tech_name_en="", tech_name_ko="")
            for _, r in comp_nodes.iterrows():
                components_payload.append({
                    "id": int(r["component_tech_id"]),
                    "name_en": str(r.get("tech_name_en", "") or ""),
                    "name_ko": str(r.get("tech_name_ko", "") or ""),
                    "docs": int(r.get("docs", 0) or 0),
                })

            # (3) edges: (flow_id, year, cluster_id) ↔ component_tech_id 매핑 후 groupby count
            try:
                preferred_label = summary.get("paths", {}).get("label_col")
            except Exception:
                preferred_label = None
            cluster_col_in_component = _resolve_cluster_col(df_component, preferred_label)

            needed_cols = ["year", cluster_col_in_component, "component_tech_id"]
            missing = [c for c in needed_cols if c not in df_component.columns]
            if missing:
                raise KeyError(f"df_component missing columns {missing}; available={list(df_component.columns)}")

            comp_map = (
                df_component[needed_cols]
                .rename(columns={cluster_col_in_component: "cluster_id"})
                .dropna()
                .copy()
            )
            comp_map["year"] = comp_map["year"].astype(int)
            comp_map["cluster_id"] = comp_map["cluster_id"].astype(int)

            flow_panel_edges = flow_panel.copy()
            flow_panel_edges["year"] = flow_panel_edges["year"].astype(int)
            flow_panel_edges["cluster_id"] = flow_panel_edges["cluster_id"].astype(int)

            joined = flow_panel_edges.merge(comp_map, on=["year", "cluster_id"], how="left")
            edges_df = (
                joined.dropna(subset=["component_tech_id"])
                .groupby(["flow_id", "component_tech_id"], as_index=False)
                .size()
                .rename(columns={"size": "weight"})
            )
            edges_payload = [
                {
                    "from_element_id": int(r["flow_id"]),
                    "to_component_id": int(r["component_tech_id"]),
                    "weight": int(r["weight"]),
                }
                for _, r in edges_df.iterrows()
            ]

            # ③ 시각화(계보도) 이벤트
            tracker.update_step("visualize", 1.0)
            yield j({
                **tracker.pack("visualize", "component", 1.0),
                "step_label": "계보도 데이터 전송",
                "step": "visualize",
                "result": {
                    "summary": {
                        "components": components_payload,
                        "elements": elements_payload,
                        "edges": edges_payload
                    },
                    "run_id": rid,
                },
            })

        except asyncio.CancelledError:
            raise
        except Exception as e:
            tb = traceback.format_exc()
            print("[STREAM ERROR]", tb)
            # 오류 시에도 그 시점의 진행도를 포함
            yield j({
                "phase": "element",
                "step": "error",
                "step_label": "오류 발생",
                "step_progress": 0,
                "element_progress": int(tracker.element_progress() * 100),
                "component_progress": int(tracker.component_progress() * 100),
                "overall_progress": int(tracker.overall_progress() * 100),
                "progress": -1,  # 하위 호환
                "error": str(e),
                "traceback": tb[-1000:],
            })
        finally:
            try:
                yield j({
                    "phase": "component",
                    "step": "stream-close",
                    "step_label": "stream-close",
                    "step_progress": 100,
                    "element_progress": int(tracker.element_progress() * 100),
                    "component_progress": int(tracker.component_progress() * 100),
                    "overall_progress": int(tracker.overall_progress() * 100),
                    "progress": -2
                })
            except Exception:
                pass
            # 임시파일 정리
            for p in temp_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

    # --- Streaming Response 설정 ---
    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
