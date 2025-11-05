# app/api/pipeline.py
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
    요소기술/구성기술 섹션별 스텝 가중치로
    - element_progress (0~1)
    - component_progress (0~1)
    - overall_progress (0~1)
    계산
    """

    ELEMENT_WEIGHT = 0.85
    COMPONENT_WEIGHT = 0.15

    ELEMENT_STEPS = [
        ("upload",       0.05),
        ("load",         0.05),
        ("year_filter",  0.10),
        ("preprocess",   0.20),
        ("embedding",    0.35),
        ("clustering",   0.15),
        ("tech_naming",  0.10),
    ]
    # 구성기술 30/60/70/100%를 맞추기 위한 가중 배분
    # grouping=0.30, naming=0.30, component_result=0.10, visualize=0.30  => 총 1.0
    COMPONENT_STEPS = [
        ("component_grouping", 0.30),
        ("component_naming",   0.30),
        ("component_result",   0.10),
        ("visualize",          0.30),
    ]

    def __init__(self):
        self._el_progress = {name: 0.0 for name, _w in self.ELEMENT_STEPS}
        self._co_progress = {name: 0.0 for name, _w in self.COMPONENT_STEPS}
        self._el_weights = {name: w for name, w in self.ELEMENT_STEPS}
        self._co_weights = {name: w for name, w in self.COMPONENT_STEPS}

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
            "progress": int(round(ov * 100)),  # 하위 호환
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

# ---------------------- base64 유틸 ----------------------
def _b64(text: str | None) -> str:
    if not text:
        return ""
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


# ---------------------- 시각화: 연결된 노드만 남기기 ----------------------
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
        E = [ee
             for _, lst in by_elem.items()
             for ee in sorted(lst, key=lambda x: x.get("weight", 0), reverse=True)[:top_edges_per_element]]

    # 2) 구성기술 기준 상위 엣지 제한(옵션)
    if top_edges_per_component:
        by_comp = defaultdict(list)
        for e in E:
            by_comp[e["to_component_id"]].append(e)
        E = [ee
             for _, lst in by_comp.items()
             for ee in sorted(lst, key=lambda x: x.get("weight", 0), reverse=True)[:top_edges_per_component]]

    elem_ids = {e["from_element_id"] for e in E}
    comp_ids = {e["to_component_id"] for e in E}

    elements_out = [n for n in elements if n["id"] in elem_ids]
    components_out = [n for n in components if n["id"] in comp_ids]

    elem_idset = {n["id"] for n in elements_out}
    comp_idset = {n["id"] for n in components_out}
    edges_out = [e for e in E
                 if e["from_element_id"] in elem_idset and e["to_component_id"] in comp_idset]

    # 정렬(선택)
    elements_out.sort(key=lambda x: (-int(x.get("docs", 0)), x["id"]))
    components_out.sort(key=lambda x: (-int(x.get("docs", 0)), x["id"]))
    edges_out.sort(key=lambda x: (-int(x.get("weight", 0)),
                                  x["from_element_id"], x["to_component_id"]))
    return elements_out, components_out, edges_out


# ---------------------- 업로드 스트리밍 → 임시파일 ----------------------
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
    전 과정 스트리밍:
      - 요소기술: CSV + 이름 Top10(영/한, docs 포함)
      - 구성기술: CSV + 이름 Top10(docs 포함)
      - 시각화: 연결된 노드만 필터링한 elements/components/edges
    """
    rid = run_id or uuid.uuid4().hex
    temp_paths: list[Path] = []
    tracker = ProgressTracker()

    def j(obj):
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

            # 2) 연도 필터
            tracker.update_step("year_filter", 0.0)
            yield j({**tracker.pack("year_filter", "element", 0.0), "step_label": "연도 필터링 시작"})
            df_year = await asyncio.to_thread(filter_df_before_year, df, int(cutoff_year))
            tracker.update_step("year_filter", 1.0)
            yield j({
                **tracker.pack("year_filter", "element", 1.0),
                "step_label": "연도 필터링 완료",
                "meta": {"rows_after_filter": len(df_year)}
            })

            # 3) 전처리
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
                run_preprocess, df_year, int(cutoff_year), progress_cb=_pre_cb,
            ))

            last_pre: Optional[tuple[int, int, str]] = None
            while not task_pre.done():
                try:
                    await asyncio.sleep(2)
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
                    yield j({
                        **tracker.pack("preprocess", "element", tracker._el_progress["preprocess"]),
                        "step_label": "데이터 전처리 하트비트",
                        "meta": {"stage": "preprocess_idle"}
                    })

            df_clean = await task_pre
            tracker.update_step("preprocess", 1.0)
            yield j({
                **tracker.pack("preprocess", "element", 1.0),
                "step_label": "데이터 전처리 완료",
                "meta": {"rows": len(df_clean)}
            })

            # 4) 임베딩
            tracker.update_step("embedding", 0.0)
            yield j({**tracker.pack("embedding", "element", 0.0), "step_label": "임베딩 시작"})
            progress_q: asyncio.Queue = asyncio.Queue()
            last_progress: Optional[tuple[int, int]] = None

            def _progress_cb(processed: int, total: int):
                try:
                    loop.call_soon_threadsafe(progress_q.put_nowait, (processed, total))
                except Exception:
                    pass

            task_embed = asyncio.create_task(asyncio.to_thread(
                run_embedding, df_clean, model_name,
                batch_size=512, checkpoint_dir=f"/tmp/emb_ckpt/{rid}",
                resume=True, progress_cb=_progress_cb,
            ))

            while not task_embed.done():
                try:
                    await asyncio.sleep(2)
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
                        "meta": {"processed": processed, "total": total, "batch_size": 512}
                    })
                else:
                    yield j({
                        **tracker.pack("embedding", "element", tracker._el_progress["embedding"]),
                        "step_label": "임베딩 하트비트"
                    })

            df_embed = await task_embed
            tracker.update_step("embedding", 1.0)
            yield j({**tracker.pack("embedding", "element", 1.0), "step_label": "임베딩 완료"})

            # 5) 클러스터링
            tracker.update_step("clustering", 0.0)
            yield j({**tracker.pack("clustering", "element", 0.0), "step_label": "클러스터링 및 추세 분석 시작"})
            df_clustered, summary = await asyncio.to_thread(run_clustering, df_embed, n_clusters)
            if not isinstance(summary, dict) or "artifacts" not in summary:
                raise RuntimeError("artifacts 누락")
            tracker.update_step("clustering", 1.0)
            yield j({**tracker.pack("clustering", "element", 1.0), "step_label": "클러스터링 및 추세 분석 완료"})

            # 6) 요소기술 네이밍 (TopN + docs 동봉)
            tracker.update_step("tech_naming", 0.0)
            yield j({**tracker.pack("tech_naming", "element", 0.0), "step_label": "요소기술 네이밍 시작"})
            naming_result = await asyncio.to_thread(
                run_tech_naming, None, artifacts=summary["artifacts"], top_n=int(top_n)
            )
            naming_df: pd.DataFrame = naming_result.get("df", pd.DataFrame())  # columns: flow_id, tech_name_en, tech_name_ko,...
            elem_csv_text: str = naming_result.get("csv_text", "")

            # 요소기술 Top10(doc 기준) 구성
            panel = _panel_from_artifacts_edges(summary["artifacts"].get("flow_edges_df", pd.DataFrame()))
            if panel.empty:
                panel = pd.DataFrame(columns=["flow_id", "docs"])
            if "docs" not in panel.columns:
                panel["docs"] = 1
            flow_docs = (
                panel.groupby("flow_id", as_index=False)["docs"]
                .sum().rename(columns={"flow_id": "id"})
            )
            elem_top = (
                naming_df.rename(columns={"flow_id": "id"})
                .merge(flow_docs, on="id", how="left")
                .sort_values("docs", ascending=False)
                .head(10)
            )
            elem_titles_en10 = elem_top["tech_name_en"].fillna("").astype(str).tolist()
            elem_titles_ko10 = elem_top["tech_name_ko"].fillna("").astype(str).tolist()
            elem_top10_payload = [
                {
                    "id": int(r.id),
                    "name_en": str(r.tech_name_en or ""),
                    "name_ko": str(r.tech_name_ko or ""),
                    "docs": int(r.docs) if pd.notna(r.docs) else 0,
                }
                for r in elem_top.itertuples(index=False)
            ]

            tracker.update_step("tech_naming", 1.0)
            # 요소기술 CSV 전송 (partial)
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
                                "content": _b64(elem_csv_text),
                            }
                        ]
                    },
                    "summary": {
                        "titles_en": elem_titles_en10,   # 호환
                        "titles_ko": elem_titles_ko10,   # 호환
                        "top10": elem_top10_payload,     # 신규(논문수 포함)
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
                cluster_col=summary["paths"]["label_col"],  # 원본 클러스터 라벨 컬럼명
                random_state=42,
            )
            df_component, comp_summary = await asyncio.to_thread(run_component_grouping, df_clustered, cfg)
            tracker.update_step("component_grouping", 1.0)
            yield j({**tracker.pack("component_grouping", "component", 1.0), "step_label": "구성기술 묶기 완료"})

            # 8) 구성기술 네이밍 (CSV 생성 + Top10)
            tracker.update_step("component_naming", 0.0)
            yield j({**tracker.pack("component_naming", "component", 0.0), "step_label": "구성기술 네이밍 시작"})
            comp_csv_text: str = await asyncio.to_thread(
                generate_component_names_csv,
                df_component,
                label_col="component_tech_id",
                text_cols=("title",),
                output_csv_path=None,
            )

            # CSV → DF 파싱(tech_name_en/ko)
            comp_names_df = pd.read_csv(StringIO(comp_csv_text))  # [component_tech_id, tech_name_en, tech_name_ko, ...]
            comp_docs = (
                df_component.groupby("component_tech_id", as_index=False)
                .size().rename(columns={"size": "docs", "component_tech_id": "id"})
            )
            comp_top = (
                comp_names_df.rename(columns={"component_tech_id": "id"})
                .merge(comp_docs, on="id", how="left")
                .sort_values("docs", ascending=False)
                .head(10)
            )
            component_titles_top10 = comp_top["tech_name_en"].fillna("").astype(str).tolist()
            comp_top10_payload = [
                {
                    "id": int(r.id),
                    "name_en": str(r.tech_name_en or ""),
                    "name_ko": str(r.tech_name_ko or ""),
                    "docs": int(r.docs) if pd.notna(r.docs) else 0,
                }
                for r in comp_top.itertuples(index=False)
            ]

            tracker.update_step("component_naming", 1.0)
            # (70%) 구성기술 결과 전송
            tracker.update_step("component_result", 1.0)  # 누적 0.30 + 0.30 + 0.10 = 0.70
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
                                "content": _b64(comp_csv_text),
                            }
                        ]
                    },
                    "summary": {
                        "titles": component_titles_top10,  # 호환
                        "top10": comp_top10_payload,       # 신규(논문수 포함)
                    },
                    "run_id": rid,
                },
            })

            # 9) 시각화 페이로드 생성 (연결된 것만)
            # 요소기술 노드: naming_df (flow_id 기반)
            # docs: flow_docs 사용
            elements_all = []
            if not naming_df.empty:
                ndf = naming_df.rename(columns={"flow_id": "id"})
                ndf = ndf.merge(flow_docs, on="id", how="left")
                for r in ndf.itertuples(index=False):
                    elements_all.append({
                        "id": int(r.id),
                        "name_en": str(getattr(r, "tech_name_en", "") or ""),
                        "name_ko": str(getattr(r, "tech_name_ko", "") or ""),
                        "docs": int(getattr(r, "docs", 0) or 0),
                    })

            # 구성기술 노드: comp_names_df + comp_docs
            components_all = []
            if not comp_names_df.empty:
                cdf = comp_names_df.rename(columns={"component_tech_id": "id"}).merge(comp_docs, on="id", how="left")
                for r in cdf.itertuples(index=False):
                    components_all.append({
                        "id": int(r.id),
                        "name_en": str(getattr(r, "tech_name_en", "") or ""),
                        "name_ko": str(getattr(r, "tech_name_ko", "") or ""),
                        "docs": int(getattr(r, "docs", 0) or 0),
                    })

            # 엣지: (원본 클러스터 → component_tech_id) 빈도
            orig_cluster_col = summary["paths"]["label_col"]  # 예: "cluster_id"
            if orig_cluster_col not in df_component.columns:
                # run_component_grouping이 원본 클러스터 id를 'cluster_id'로 보존하지 않았다면 폴백 방지:
                # 존재하는 라벨 컬럼 중 하나를 찾되, 없으면 엣지를 비움.
                fallback_col = next((c for c in df_component.columns if c.lower() in ["cluster_id", "orig_cluster_id", "source_cluster_id"]), None)
                orig_cluster_col = fallback_col or orig_cluster_col

            edges_all = []
            if orig_cluster_col in df_component.columns:
                cross = (
                    df_component
                    .groupby([orig_cluster_col, "component_tech_id"], as_index=False)
                    .size().rename(columns={"size": "weight"})
                )
                for r in cross.itertuples(index=False):
                    edges_all.append({
                        "from_element_id": int(getattr(r, orig_cluster_col)),
                        "to_component_id": int(getattr(r, "component_tech_id")),
                        "weight": int(getattr(r, "weight")),
                    })
            else:
                # 엣지 구성 불가 시 빈 그래프 (프론트 안정성용)
                edges_all = []

            # 연결된 것만 남겨서 전송 (필요시 파라미터 조정)
            elements_v, components_v, edges_v = induce_connected_only(
                elements_all,
                components_all,
                edges_all,
                min_weight=0,                # 필요시 50~200 등으로 올려 그래프 간소화
                top_edges_per_element=3,     # 요소기술당 상위 3개 엣지
                top_edges_per_component=None # 구성기술 기준 제한은 옵션
            )

            tracker.update_step("visualize", 1.0)  # (100%)
            yield j({
                **tracker.pack("visualize", "component", 1.0),
                "step": "visualize",
                "step_label": "계보도 데이터 전송",
                "result": {
                    "summary": {
                        "components": components_v,
                        "elements": elements_v,
                        "edges": edges_v,
                    }
                },
                "run_id": rid,
            })

        except asyncio.CancelledError:
            raise
        except Exception as e:
            tb = traceback.format_exc()
            print("[STREAM ERROR]", tb)
            yield j({
                "phase": "element",
                "step": "error",
                "step_label": "오류 발생",
                "step_progress": 0,
                "element_progress": int(tracker.element_progress() * 100),
                "component_progress": int(tracker.component_progress() * 100),
                "overall_progress": int(tracker.overall_progress() * 100),
                "progress": -1,
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
            for p in temp_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
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
