from fastapi import APIRouter
import json
import pandas as pd
from pathlib import Path

router = APIRouter()

def run_preprocess(input_path: str, cutoff_year: int):
    """
    1. JSON/CSV 불러오기
    2. cutoff_year 이전 데이터 필터링
    3. CSV 저장
    """
    p = Path(input_path)
    out_path = Path(f"./app/data/processed/filtered_until_{cutoff_year}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    entries = []

    if p.suffix == ".json" or p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as infile:
            for line in infile:
                try:
                    entry = json.loads(line)
                    update = entry.get("update_date", "")
                    if update[:4].isdigit() and int(update[:4]) < cutoff_year:
                        entries.append(entry)
                        count += 1
                except Exception:
                    continue
        df = pd.DataFrame(entries)
    else:
        df = pd.read_csv(p)
        df = df[df["update_date"].astype(str).str[:4].astype(int) < cutoff_year]
        count = len(df)

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return {
        "message": f"✅ {cutoff_year}년까지 {count}개 데이터 필터링 완료",
        "output_path": str(out_path),
        "rows": count
    }
