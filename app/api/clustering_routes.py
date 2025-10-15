from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from app.services.clustering_service import run_clustering_pipeline
import os

router = APIRouter(prefix="/clustering", tags=["clustering"])

@router.post("/run")
async def run_clustering_api(
    data_csv: UploadFile = File(...),
    embed_col: str = Form("embedding"),
    year_col: str = Form("year"),
    text_cols: str = Form("title"),                # 콤마구분: "title,abstract"
    pca_dim: int = Form(50),
    k_list: str = Form("100,150,200,300,400,600,800"),
    final_k: int | None = Form(None),              # None이면 연도별 best-k 자동
    do_tfidf: bool = Form(True),
    min_docs_per_cluster: int = Form(5),           # 너무 작은 클러스터는 요약 제외
    similarity_top1: bool = Form(True),            # prev→next 1:1 매칭
):
    try:
        out_dir, summary = await run_clustering_pipeline(
            data_csv=data_csv,
            embed_col=embed_col,
            year_col=year_col,
            text_cols=[c.strip() for c in text_cols.split(",") if c.strip()],
            pca_dim=int(pca_dim),
            k_list=[int(x) for x in k_list.split(",") if x.strip()],
            final_k=None if final_k in (None, "") else int(final_k),
            do_tfidf=bool(do_tfidf),
            min_docs_per_cluster=int(min_docs_per_cluster),
            similarity_top1=bool(similarity_top1),
        )
        return JSONResponse({"ok": True, "out_dir": out_dir, **summary})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download")
def download(path: str):
    if not os.path.exists(path):
        raise HTTPException(404, "file not found")
    return FileResponse(path, filename=os.path.basename(path))
