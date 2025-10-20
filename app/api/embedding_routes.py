from fastapi import APIRouter, UploadFile, File, Form
from app.services.embedding_service import run_embedding

router = APIRouter()

@router.post("/run")
async def run_embedding_api(
    input_csv: UploadFile = File(...),
    title_col: str = Form(...),
    date_col: str = Form(...),
    model_name: str = Form("all-MiniLM-L6-v2"),
    out_parquet_name: str = Form("df_all_for_pipeline.parquet"),
):
    out_path, rows, y_min, y_max = await run_embedding(
        input_csv=input_csv,
        title_col=title_col,
        date_col=date_col,
        model_name=model_name,
        out_parquet_name=out_parquet_name,
    )
    return {
        "out_parquet": out_path,
        "rows": rows,
        "year_min": y_min,
        "year_max": y_max,
    }
