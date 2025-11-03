# app/api/pipeline.py
import asyncio
import os
import csv
import json
import traceback
import uuid
import tempfile
import base64
from pathlib import Path
from typing import Any, Tuple, Optional

import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from app.core.preprocess import run_preprocess, filter_df_before_year
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
    COMPONENT_STEPS = [
        ("component_grouping", 0.40),
        ("component_naming", 0.50),
        ("finalize", 0.10),
    ]

    def __init__(self):
        self._el_progress = {name: 0.0 for name, _ in self.ELEMENT_STEPS}
        self._co_progress = {name: 0.0 for name, _ in self.COMPONENT_STEPS}

        self._el_weights = {name: w for name, w in self.ELEMENT_STEPS}
        self._co_weights = {name: w for name, w in self.COMPONENT_STEPS}

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
    cutoff_year: int = Form(2026),
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
      - 결과물: 프론트에 CSV(base64)로 직접 전달
        * 요소기술 네이밍: names_generated_flowagg.csv
        * 구성기술 네이밍: component_tech_names.csv
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
            # print(f"✅ 전처리 시작: {len(df):,}개의 데이터")

            # 2) 연도 필터링
            tracker.update_step("year_filter", 0.0)
            yield j({**tracker.pack("year_filter", "element", 0.0), "step_label": "데이터 연도 필터링 시작"})
            df_year = await asyncio.to_thread(filter_df_before_year, df, int(cutoff_year))
            tracker.update_step("year_filter", 1.0)
            yield j({**tracker.pack("year_filter", "element", 1.0), "step_label": "연도 필터링 완료",
                     "meta": {"rows_after_filter": len(df_year)}})
            # print(f"✅ 연도 필터링 완료: {len(df_year):,}개의 데이터")

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

            # ✅ CPC 관련 옵션 제거된 run_preprocess 호출
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
                    # 하트비트
                    yield j({**tracker.pack("preprocess", "element", tracker._el_progress["preprocess"]),
                             "step_label": "데이터 전처리 하트비트", "meta": {"stage": "preprocess_idle"}})

            df_clean = await task_pre
            tracker.update_step("preprocess", 1.0)
            yield j({**tracker.pack("preprocess", "element", 1.0),
                     "step_label": "데이터 전처리 완료",
                     "meta": {"rows": len(df_clean)}})
            # print(f"✅ 전처리 완료: {len(df_clean):,}개의 데이터")

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

            # 임베딩 실행
            task_embed = asyncio.create_task(asyncio.to_thread(
                run_embedding,
                df_clean,
                model_name,
                batch_size=512,
                checkpoint_dir=f"/tmp/emb_ckpt/{rid}",
                resume=True,
                progress_cb=_progress_cb,
            ))

            # 하트비트
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
            # (여기선 내부 진행 콜백이 없으므로 단계 단위로 0→1 처리)
            df_clustered, summary = await asyncio.to_thread(run_clustering, df_embed, n_clusters)
            if not isinstance(summary, dict) or "artifacts" not in summary:
                raise RuntimeError("artifacts 누락")
            tracker.update_step("clustering", 1.0)
            yield j({**tracker.pack("clustering", "element", 1.0), "step_label": "클러스터링 및 추세 분석 완료"})

            # 6) 요소기술 네이밍
            tracker.update_step("tech_naming", 0.0)
            yield j({**tracker.pack("tech_naming", "element", 0.0), "step_label": "요소기술 네이밍 시작"})
            # (동기 함수라 하트비트 없음 → 시작/완료로 표기)
            naming_result = await asyncio.to_thread(
                run_tech_naming, None, artifacts=summary["artifacts"], top_n=int(top_n)
            )
            elem_csv_text: str = naming_result.get("csv_text", "")
            tracker.update_step("tech_naming", 1.0)
            yield j({**tracker.pack("tech_naming", "element", 1.0), "step_label": "요소기술 네이밍 완료"})

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

            # (참고) 저장 로직은 더 이상 사용하지 않지만, 혹시 복구할 수 있도록 주석으로 남겨둡니다.
            # -----------------------------------------------------------------------
            # elem_csv_path = OUTPUT_DIR / "names_generated_flowagg.csv"
            # (OUTPUT_DIR / "names_generated_flowagg.csv").write_text(elem_csv_text, encoding="utf-8-sig")
            # comp_csv_path = OUTPUT_DIR / "component_tech_names.csv"
            # (OUTPUT_DIR / "component_tech_names.csv").write_text(comp_csv_text, encoding="utf-8-sig")
            # -----------------------------------------------------------------------

            # 8.5) CSV 텍스트 → base64 (바이너리 안전)
            def _b64(text: str) -> str:
                if text is None:
                    return ""
                return base64.b64encode(text.encode("utf-8")).decode("ascii")

            elem_csv_b64 = _b64(elem_csv_text)
            comp_csv_b64 = _b64(comp_csv_text)

            # 9) 완료
            tracker.update_step("finalize", 1.0)
            keywords = _to_str_list(summary.get("keywords", []))[:100]
            titles = _to_str_list(summary.get("titles", []))[:100]
            yield j({
                **tracker.pack("finalize", "component", 1.0),
                "step_label": "파이프라인 완료",
                "result": {
                    "outputs": {
                        "files": [
                            {
                                "filename": "names_generated_flowagg.csv",
                                "mime": "text/csv; charset=utf-8",
                                "encoding": "base64",
                                "content": elem_csv_b64
                            },
                            {
                                "filename": "component_tech_names.csv",
                                "mime": "text/csv; charset=utf-8",
                                "encoding": "base64",
                                "content": comp_csv_b64
                            }
                        ]
                    },
                    "summary": {
                        "keywords": keywords,
                        "titles": titles,
                        "paths": summary.get("paths", {})
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
                **tracker.pack("error", "element", 0.0),
                "step_label": "오류 발생",
                "progress": -1,  # 하위 호환
                "error": str(e),
                "traceback": tb[-1000:],
            })
        finally:
            try:
                yield j({**tracker.pack("stream-close", "component", 1.0),
                         "step_label": "stream-close", "progress": -2})
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
