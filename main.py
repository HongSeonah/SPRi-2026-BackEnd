from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# HuggingFace tokenizer 멀티프로세스 경고 방지
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

load_dotenv()  # .env 파일 로드

from app.api import pipeline
app = FastAPI(title="Spri AI Pipeline", version="1.0")

# CORS 설정 (React용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 특정 도메인만 지정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
# app.include_router(preprocess.router, prefix="/api")
# app.include_router(embedding.router, prefix="/api")
# app.include_router(clustering.router, prefix="/api")
# app.include_router(tech_naming.router, prefix="/api")

app.include_router(pipeline.router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Spri AI Pipeline API is running 🚀"}
