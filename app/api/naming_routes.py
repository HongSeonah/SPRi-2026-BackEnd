from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from app.services.naming_service import run_naming_pipeline
import os

router = APIRouter(prefix="/naming", tags=["naming"])

@router.post("/run")
async def run_naming_api(
    mode: str = Form("both"),                  # "hybrid" | "flowagg" | "both"
    label_suffix: str = Form("k100"),
    method: str = Form("A"),
    year_pred: int = Form(2024),               # 타깃 연도 (WEAK/UPGRADE 대상)
    topk_keywords: int = Form(40),             # hybrid 키워드 개수
    n_rep_titles: int = Form(5),               # hybrid 대표 타이틀 개수
    use_embeddings: bool = Form(True),         # hybrid 임베딩 우선
    agg_window: int | None = Form(None),       # flowagg 집계 최근 N년 (None=전체)
    topk_per_cluster: int = Form(80),          # flowagg 연/클러스터당 키워드 수
    topk_flow_keywords: int = Form(60),        # flowagg 최종 상위 키워드 수
    doc_weight_norm_by_year: bool = Form(True),
    model: str = Form("gpt-4o-mini")           # OpenAI Chat Completions 모델
):
    try:
        out = await run_naming_pipeline(
            mode=mode,
            label_suffix=label_suffix,
            method=method,
            year_pred=int(year_pred),
            topk_keywords=int(topk_keywords),
            n_rep_titles=int(n_rep_titles),
            use_embeddings=bool(use_embeddings),
            agg_window=None if agg_window in (None, "", "None") else int(agg_window),
            topk_per_cluster=int(topk_per_cluster),
            topk_flow_keywords=int(topk_flow_keywords),
            doc_weight_norm_by_year=bool(doc_weight_norm_by_year),
            model=model,
        )
        return JSONResponse({"ok": True, **out})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download")
def download(path: str):
    if not os.path.exists(path):
        raise HTTPException(404, "file not found")
    return FileResponse(path, filename=os.path.basename(path))
