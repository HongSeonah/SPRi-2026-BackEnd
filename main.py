from fastapi import FastAPI
from app.api.preprocessing_routes import router as preproc_router
from app.api.clustering_routes import router as cluster_router
from app.api.naming_routes import router as naming_router

app = FastAPI(title="Preprocessing API")
app.include_router(preproc_router)
app.include_router(cluster_router)
app.include_router(naming_router)

@app.get("/")
def health():
    return {"status": "ok"}
