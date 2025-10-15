import os
import io
import re
import json
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from fastapi import UploadFile
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SK_STOPWORDS

from app.utils.io_utils import tmp_path

# ─────────────────────────────────────────────────────────────────────
# 전역 설정
# ─────────────────────────────────────────────────────────────────────

# 불용어: NLTK 제거 → scikit-learn 내장 불용어 사용 (다운로드 불필요)
SW = set(SK_STOPWORDS)

# 고정 CPC 파일 경로 (프로젝트 구조에 맞춰 상대경로 → 절대경로로 변환)
CPC_CSV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "cpc_data_with_titles.csv")
)


# ─────────────────────────────────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────────────────────────────────

def load_cpc_flex(path: str) -> pd.DataFrame:
    """
    CPC CSV를 읽어서 심볼/제목 컬럼명을 표준화(SYMBOL, cpc_title)해서 반환.
    - 인코딩: utf-8-sig → utf-8 → cp949 순으로 시도
    - 제목 후보: cpc_title / Description / description / 대분류_DESCRIPTION / title / TITLE
    - 심볼 후보: SYMBOL / symbol / Symbol / 코드 / 코드명
    """
    encodings = ["utf-8-sig", "utf-8", "cp949"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception as e:
            last_err = e
    else:
        raise last_err

    title_candidates = ["cpc_title", "Description", "description",
                        "대분류_DESCRIPTION", "title", "TITLE"]
    symbol_candidates = ["SYMBOL", "symbol", "Symbol", "코드", "코드명"]

    title_col = next((c for c in title_candidates if c in df.columns), None)
    symbol_col = next((c for c in symbol_candidates if c in df.columns), None)

    if symbol_col is None:
        raise ValueError(f"CPC CSV에서 심볼 컬럼을 찾지 못했습니다. 후보: {symbol_candidates}")
    if title_col is None:
        raise ValueError(f"CPC CSV에서 제목 컬럼을 찾지 못했습니다. 후보: {title_candidates}")

    df = df.rename(columns={symbol_col: "SYMBOL", title_col: "cpc_title"})
    return df[["SYMBOL", "cpc_title"]].copy()


def clean_title(text: str) -> str:
    """
    알파벳 3자 이상 토큰만 남기고 불용어 제거 → 공백 단일화.
    """
    words = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", str(text).lower()) if w not in SW]
    return " ".join(words)


def stream_filter_arxiv_jsonl(
    upload: UploadFile,
    out_path: str,
    cutoff_year: int
) -> None:
    """
    업로드된 JSONL(라인별 JSON)을 스트리밍으로 읽어,
    update_date < cutoff_year 인 라인만 out_path 로 필터 저장.

    청크 경계에서 라인이 잘리지 않도록 버퍼링 처리.
    """
    # 업로드 스트림은 비동기여서 호출부에서 await로 파일을 넘기는 대신,
    # 여기서는 동기적으로 처리하기 위해 .read()를 chunk 단위로 await 해주는 쪽이 자연스럽다.
    # 이 함수는 run_preprocessing() 내부에서 호출되므로 여기는 sync지만,
    # 실제 read는 run_preprocessing에서 수행하고 전달하는 형태로도 가능.
    raise NotImplementedError("This function is intended to be used via the async wrapper below.")


async def async_stream_filter_arxiv_jsonl(
    upload: UploadFile,
    out_path: str,
    cutoff_year: int
) -> None:
    """
    비동기 UploadFile 을 스트리밍으로 읽어 JSONL 필터링.
    청크 경계 보정 버퍼(buffer)를 사용.
    """
    buffer = ""  # 덜 끝난 라인을 담아두는 버퍼
    with open(out_path, "w", encoding="utf-8") as out:
        while True:
            chunk = await upload.read(1024 * 1024)  # 1 MiB
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="ignore")

            # 줄 끝까지 끊고, 마지막 조각은 버퍼에 남김
            lines = buffer.split("\n")
            buffer = lines.pop()  # 마지막(미완성) 라인

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    year = str(entry.get("update_date", ""))[:4]
                    if year.isdigit() and int(year) < cutoff_year:
                        out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                except Exception:
                    # 깨진 라인은 무시
                    continue

        # 파일 종료 시 버퍼에 남은 마지막 라인 처리
        last = buffer.strip()
        if last:
            try:
                entry = json.loads(last)
                year = str(entry.get("update_date", ""))[:4]
                if year.isdigit() and int(year) < cutoff_year:
                    out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass


def top1_batched(
    query_emb: torch.Tensor,
    corpus_emb: torch.Tensor,
    batch_size: int = 512
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    코사인 유사도(정규화 임베딩의 내적)를 전체 행렬로 만들지 않고,
    query를 배치로 나눠 각 쿼리의 Top-1 (score, index)만 계산.
    - query_emb: [Q, D]
    - corpus_emb: [C, D]
    반환: (best_scores[Q], best_indices[Q])
    """
    corpus_T = corpus_emb.T.contiguous()  # [D, C]
    qn = query_emb.size(0)
    best_s_list: List[torch.Tensor] = []
    best_i_list: List[torch.Tensor] = []

    for i in range(0, qn, batch_size):
        q = query_emb[i:i + batch_size]          # [b, D]
        sims = torch.matmul(q, corpus_T)         # [b, C]  (코사인=내적)
        bs, bi = sims.max(dim=1)                 # 각 쿼리별 최고점/인덱스
        best_s_list.append(bs)
        best_i_list.append(bi)

        # 메모리 해제 유도
        del q, sims, bs, bi
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best_scores = torch.cat(best_s_list, dim=0)
    best_idx = torch.cat(best_i_list, dim=0)
    return best_scores, best_idx


# ─────────────────────────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────────────────────────

async def run_preprocessing(
    arxiv_jsonl: UploadFile,
    keywords: str,
    model_name: str,
    cutoff_year: int
) -> Tuple[str, float]:
    """
    전체 전처리 수행 (CPC는 내부 고정 파일 사용)

    1) arXiv JSONL 필터링 (update_date < cutoff_year)
    2) CPC 로드(유연 매핑) + 텍스트 클린/키워드 매칭
    3) 임베딩 매칭(논문 abstract ↔ CPC title) - 배치 Top-1
    4) 임계점(knee) 계산
    5) CSV 저장 후 경로와 knee 반환
    """

    # 1) arXiv 필터링 (스트리밍)
    tmp_jsonl = tmp_path(".jsonl")
    await async_stream_filter_arxiv_jsonl(arxiv_jsonl, tmp_jsonl, cutoff_year)

    # 필터 결과를 DataFrame으로 적재 (테스트 용도로는 충분; 대용량이면 청크처리 고려)
    rows = []
    with open(tmp_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        raise ValueError("필터링 결과가 비었습니다. update_date 조건을 확인하세요.")

    df_paper = pd.DataFrame(rows)
    if "abstract" not in df_paper.columns:
        raise ValueError("arxiv 데이터에 'abstract' 필드가 없습니다.")

    # 2) CPC 로드(유연 매핑) + 클린/키워드
    if not os.path.exists(CPC_CSV_PATH):
        raise FileNotFoundError(f"CPC CSV 파일이 없습니다: {CPC_CSV_PATH}")
    df_cpc = load_cpc_flex(CPC_CSV_PATH)

    kw_list = [k.strip().lower() for k in keywords.split(",") if k.strip()]

    def match_kw(t: str):
        t = (t or "").lower()
        hits = [k for k in kw_list if k in t]
        return ", ".join(hits) if hits else None

    df_cpc["cpc_title"] = df_cpc["cpc_title"].apply(clean_title)
    df_cpc["matched_keyword"] = df_cpc["cpc_title"].apply(match_kw)

    # 3) 임베딩 매칭 (메모리 안전: 배치 Top-1)
    #  - CPU 먼저 권장; GPU가 안정되면 device="cuda"로 교체 가능
    device = "cpu"
    model = SentenceTransformer(model_name, device=device)

    cpc_emb = model.encode(
        df_cpc["cpc_title"].tolist(),
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        device=device,
    )
    abs_emb = model.encode(
        df_paper["abstract"].fillna("").tolist(),
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        device=device,
    )

    best_scores, best_idx = top1_batched(abs_emb, cpc_emb, batch_size=512)

    df_paper["similarity"] = best_scores.cpu().numpy()
    idx_list = best_idx.cpu().tolist()
    df_paper["matched_cpc"] = [df_cpc.iloc[i]["SYMBOL"] for i in idx_list]
    df_paper["matched_cpc_description"] = [df_cpc.iloc[i]["cpc_title"] for i in idx_list]

    # 4) 임계점(knee) 계산 (간단: 2차 차분 최대 지점)
    thresholds = np.linspace(0, 0.6, 61)
    counts = np.array([(df_paper["similarity"] >= t).sum() for t in thresholds])
    second = np.gradient(np.gradient(counts))
    knee_idx = int(np.argmax(second))
    knee = float(thresholds[knee_idx])

    df_paper["selected"] = df_paper["similarity"] >= knee

    # 5) 저장
    out_csv = tmp_path(".csv")
    df_paper.to_csv(out_csv, index=False)
    return out_csv, knee
