from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from app.services.preprocessing_service import run_preprocessing

router = APIRouter(prefix="/preprocessing", tags=["preprocessing"])

@router.post("/run")
async def run_preprocessing_api(
    arxiv_jsonl: UploadFile = File(...),   # arxiv 원본 JSONL
    keywords: str = Form("data,algorithm,software,reality,virtual,augmented"),
    model_name: str = Form("all-MiniLM-L6-v2"),
    cutoff_year: int = Form(2025)
):
    """
    전체 전처리 과정 실행:
    1. arXiv 필터링 (2025년 이전)
    2. CPC 전처리 및 키워드 추출
    3. 논문 ↔ CPC 매칭
    4. 유사도 기반 임계점 계산
    """
    out_csv, knee_threshold = await run_preprocessing(
        arxiv_jsonl, keywords, model_name, cutoff_year
    )

    return FileResponse(
        out_csv,
        filename="preprocessed_results.csv",
        media_type="text/csv",
        headers={"X-Knee-Threshold": f"{knee_threshold:.4f}"}
    )
