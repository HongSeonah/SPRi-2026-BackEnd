# 1️⃣ Python 베이스 이미지
FROM python:3.11-slim

# 2️⃣ 작업 디렉토리 설정
WORKDIR /app

# 3️⃣ requirements 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4️⃣ 코드 복사
COPY . .

# 5️⃣ FastAPI 실행 (Uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
