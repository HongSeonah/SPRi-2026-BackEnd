from fastapi import APIRouter, UploadFile, File, Form
from app.services.cluster_service import run_cluster_pipeline

router = APIRouter()

@router.post("/run")
async def run_cluster_api(
    df_all_parquet: UploadFile = File(...),
    label_k: int = Form(100),
    pca_dim: int = Form(50),
    min_df: int = Form(3),
    max_df: float = Form(0.9),
    lsa_dim: int = Form(256),
    sim_threshold: float = Form(0.08),
    label_suffix: str = Form("k100"),
    method_tag: str = Form("A"),
):
    out = await run_cluster_pipeline(
        df_all_parquet=df_all_parquet,
        k=label_k,
        pca_dim=pca_dim,
        min_df=min_df,
        max_df=max_df,
        lsa_dim=lsa_dim,
        sim_thr=sim_threshold,
        label_suffix=label_suffix,
        method_tag=method_tag,
    )
    return out
