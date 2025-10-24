# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) 의존성만 먼저 복사
COPY requirements.txt .

# 2) pip 캐시 마운트 + 설치 (requirements 바뀌지 않으면 레이어 캐시 재사용)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# 3) 그 다음에 소스 복사 (코드만 바뀌면 위 레이어 재사용)
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
