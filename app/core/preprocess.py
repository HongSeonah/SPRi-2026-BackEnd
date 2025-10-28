# app/core/preprocess.py
from __future__ import annotations

import json
from typing import Iterable, IO, Optional, List, Tuple, Sequence, Literal, Union, Callable
from pathlib import Path
from pathlib import Path
import os

import numpy as np
import pandas as pd

# ▼ CPC 매칭용 (semantic)
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
except Exception:
    SentenceTransformer = None
    util = None
    torch = None

# FastAPI 라우터(기존 유지)
from fastapi import APIRouter
router = APIRouter()

# =========================================================
# JSONL 유틸 (update_date[:4].isdigit() + int < cutoff)
# =========================================================

def _iter_jsonl_lines(
    fp: IO[str],
    *,
    log_errors: bool = False
) -> Iterable[dict]:
    for i, line in enumerate(fp, start=1):
        s = line.strip()
        if not s:
            continue
        try:
            yield json.loads(s)
        except Exception as e:
            if log_errors:
                print(f"[iter_jsonl] line {i} parse error: {e}")
            continue


def _year_from_update_date_like_original(update_date: str) -> int:
    s = str(update_date)
    y4 = s[:4]
    return int(y4) if y4.isdigit() else -1


def count_before_year_stream(fp: IO[str], cutoff_year: int) -> int:
    cnt = 0
    for entry in _iter_jsonl_lines(fp):
        y = _year_from_update_date_like_original(entry.get("update_date", ""))
        if y >= 0 and y < int(cutoff_year):
            cnt += 1
    return cnt


def filter_before_year_stream_to_df(fp: IO[str], cutoff_year: int) -> pd.DataFrame:
    rows: List[dict] = []
    for entry in _iter_jsonl_lines(fp):
        y = _year_from_update_date_like_original(entry.get("update_date", ""))
        if y >= 0 and y < int(cutoff_year):
            rows.append(entry)
    return pd.DataFrame(rows)

# =========================================================
# 파일 경로 기반 I/O (원문 스크립트와 동일)
# =========================================================

def count_until_year_from_path(input_path: str, cutoff_year: int = 2025) -> int:
    p = Path(input_path)
    cnt = 0
    with p.open("r", encoding="utf-8") as infile:
        for line in infile:
            try:
                entry = json.loads(line)
                y = _year_from_update_date_like_original(entry.get("update_date", ""))
                if y >= 0 and y < int(cutoff_year):
                    cnt += 1
            except Exception:
                continue
    return cnt


def filter_jsonl_to_jsonl_with_cutoff(
    input_path: str,
    output_path: str,
    cutoff_year: int = 2026,
    *,
    log_errors: bool = False
) -> int:
    in_p = Path(input_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with in_p.open("r", encoding="utf-8") as infile, out_p.open("w", encoding="utf-8") as outfile:
        for i, line in enumerate(infile, start=1):
            try:
                entry = json.loads(line)
                y = _year_from_update_date_like_original(entry.get("update_date", ""))
                if y >= 0 and y < int(cutoff_year):
                    json.dump(entry, outfile, ensure_ascii=False)
                    outfile.write("\n")
                    count += 1
            except Exception as e:
                if log_errors:
                    print(f"[filter_jsonl] line {i} parse error: {e}")
                continue
    return count


def jsonl_to_csv(input_jsonl_path: str, output_csv_path: str) -> Tuple[int, str]:
    in_p = Path(input_jsonl_path)
    out_p = Path(output_csv_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(str(in_p), lines=True)
    df.to_csv(str(out_p), index=False)
    return len(df), str(out_p)

# =========================================================
# DataFrame 기반 파생/필터 (원문 방식과 동일성 유지)
# =========================================================

def _derive_year(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "year" not in out.columns:
        if "update_date" in out.columns:
            out["year"] = (
                out["update_date"]
                .astype(str)
                .map(lambda s: int(s[:4]) if s[:4].isdigit() else -1)
                .astype(int)
            )
        else:
            out["year"] = -1
    return out


def filter_df_before_year(df: pd.DataFrame, cutoff_year: int) -> pd.DataFrame:
    df2 = _derive_year(df)
    yrs = pd.to_numeric(df2["year"], errors="coerce").fillna(-1).astype(int)
    return df2[yrs < int(cutoff_year)].copy().reset_index(drop=True)

# =========================================================
# 키워드 필터(원문 보강)
# =========================================================

def _concat_text(row: pd.Series, cols: Sequence[str]) -> str:
    parts = []
    for c in cols:
        if c in row and pd.notna(row[c]):
            parts.append(str(row[c]))
    return " ".join(parts).strip()


def filter_df_by_keywords_literal(
    df: pd.DataFrame,
    keywords: Sequence[str],
    *,
    text_cols: Sequence[str] = ("title", "abstract"),
    case_insensitive: bool = True,
    use_regex: bool = False
) -> pd.DataFrame:
    use_cols = [c for c in text_cols if c in df.columns]
    if not use_cols or not keywords:
        return df.copy()

    proc = df.copy()
    if case_insensitive:
        kw_list = [str(k).lower() for k in keywords]
        def _hit(row: pd.Series) -> bool:
            text = _concat_text(row, use_cols).lower()
            if use_regex:
                import re
                return any(re.search(k, text) is not None for k in kw_list)
            return any(k in text for k in kw_list)
    else:
        kw_list = [str(k) for k in keywords]
        def _hit(row: pd.Series) -> bool:
            text = _concat_text(row, use_cols)
            if use_regex:
                import re
                return any(re.search(k, text) is not None for k in kw_list)
            return any(k in text for k in kw_list)

    mask = proc.apply(_hit, axis=1)
    return proc[mask].copy().reset_index(drop=True)


def filter_df_by_keywords_semantic(
    df: pd.DataFrame,
    keywords: Sequence[str],
    *,
    text_cols: Sequence[str] = ("title", "abstract"),
    model_name: str = "all-MiniLM-L6-v2",
    threshold: float = 0.30,
    normalize: bool = True,
    device: Optional[str] = None,
    add_score_cols: bool = True
) -> pd.DataFrame:
    use_cols = [c for c in text_cols if c in df.columns]
    if not use_cols or not keywords:
        return df.copy()

    if SentenceTransformer is None:
        raise ImportError("sentence-transformers 가 필요합니다. pip install sentence-transformers")

    import numpy as np

    proc = df.copy()
    texts = proc.apply(lambda r: _concat_text(r, use_cols), axis=1).fillna("").astype(str).tolist()

    if device is None:
        device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"

    model = SentenceTransformer(model_name, device=device)
    text_emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=normalize, show_progress_bar=False, device=device)
    kw_emb = model.encode(list(map(str, keywords)), convert_to_tensor=True, normalize_embeddings=normalize, show_progress_bar=False, device=device)

    sim = util.cos_sim(text_emb, kw_emb)  # [num_rows x num_keywords]
    best_scores, best_idx = sim.max(dim=1)
    keep_mask = best_scores >= float(threshold)
    kept = proc[keep_mask.cpu().numpy()].copy().reset_index(drop=True)

    if add_score_cols:
        kept["similarity"] = best_scores[keep_mask].cpu().numpy()
        kept["best_keyword"] = [keywords[i] for i in best_idx[keep_mask].cpu().tolist()]

    # 메모리 정리
    try:
        del text_emb, kw_emb, sim
        if torch and device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    return kept

# =========================================================
# CPC 전처리 & 매칭 (추가)
# =========================================================

import re

def _clean_text_for_match(text: str, *, keep_signs: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    if keep_signs:
        # 알파벳/숫자 + 의미 있는 부호(#+-/. slash/dot)만 허용
        text = re.sub(r"[^a-z0-9#\+\/\-\.\s]", " ", text)
    else:
        text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_cpc_df(
    df_cpc: pd.DataFrame,
    *,
    title_col: str = "cpc_title",
    symbol_col: str = "SYMBOL",
    keep_signs: bool = True
) -> pd.DataFrame:
    if title_col not in df_cpc.columns or symbol_col not in df_cpc.columns:
        raise ValueError(f"df_cpc에 '{symbol_col}', '{title_col}' 컬럼이 필요합니다.")
    out = df_cpc.copy()
    out[title_col] = out[title_col].fillna("").astype(str).map(lambda s: _clean_text_for_match(s, keep_signs=keep_signs))
    return out.reset_index(drop=True)


def match_cpc_to_papers(
    df_paper: pd.DataFrame,
    df_cpc: pd.DataFrame,
    *,
    paper_text_cols: Sequence[str] = ("abstract",),
    cpc_title_col: str = "cpc_title",
    cpc_symbol_col: str = "SYMBOL",
    model_name: str = "all-MiniLM-L6-v2",
    threshold: float = 0.30,
    normalize: bool = True,
    device: Optional[str] = None,
    batch_size: int = 2048
) -> pd.DataFrame:
    """
    논문 텍스트(기본 abstract) ↔ CPC 타이틀 매칭.
    - 출력: df_paper에 matched_cpc, matched_cpc_description, similarity, final_cpc, final_cpc_description 추가
    - 메모리 안전을 위해 논문 임베딩/유사도는 배치로 처리
    """
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers 가 필요합니다. pip install sentence-transformers")

    use_cols = [c for c in paper_text_cols if c in df_paper.columns]
    if not use_cols:
        # 텍스트가 없으면 그대로 반환
        return df_paper.copy().reset_index(drop=True)

    proc_p = df_paper.copy()
    proc_p["__paper_text__"] = proc_p.apply(lambda r: _concat_text(r, use_cols), axis=1).fillna("").astype(str)
    proc_p["__paper_text__"] = proc_p["__paper_text__"].map(lambda s: _clean_text_for_match(s, keep_signs=True))

    proc_c = preprocess_cpc_df(df_cpc, title_col=cpc_title_col, symbol_col=cpc_symbol_col, keep_signs=True)

    if device is None:
        device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"

    model = SentenceTransformer(model_name, device=device)
    # CPC는 전체 한 번에
    cpc_emb = model.encode(
        proc_c[cpc_title_col].tolist(),
        convert_to_tensor=True, normalize_embeddings=normalize, show_progress_bar=False, device=device
    )

    # 논문은 배치
    n = len(proc_p)
    best_sim = []
    best_idx = []

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        texts = proc_p["__paper_text__"].iloc[s:e].tolist()
        abs_emb = model.encode(
            texts,
            convert_to_tensor=True, normalize_embeddings=normalize, show_progress_bar=False, device=device
        )
        sim = util.cos_sim(abs_emb, cpc_emb)  # [batch x num_cpc]
        bs, bi = torch.max(sim, dim=1)
        best_sim.append(bs.detach().cpu())
        best_idx.append(bi.detach().cpu())

        # 메모리 정리
        try:
            del abs_emb, sim, bs, bi
            if torch and device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    import numpy as np
    best_scores = np.concatenate([x.numpy() for x in best_sim], axis=0)
    best_index = np.concatenate([x.numpy() for x in best_idx], axis=0)

    SYMS   = proc_c[cpc_symbol_col].tolist()
    TITLES = proc_c[cpc_title_col].tolist()

    proc_p["matched_cpc"] = [SYMS[i] for i in best_index.tolist()]
    proc_p["matched_cpc_description"] = [TITLES[i] for i in best_index.tolist()]
    proc_p["similarity"] = best_scores

    # threshold 적용 → exploration 라벨링
    def _final(row):
        if row["similarity"] >= float(threshold):
            return row["matched_cpc"], row["matched_cpc_description"]
        return "exploration", "Exploratory Topic (Uncertain CPC Match)"

    out_final = proc_p.apply(_final, axis=1, result_type="expand")
    proc_p["final_cpc"] = out_final[0]
    proc_p["final_cpc_description"] = out_final[1]

    # 내부 열 정리
    proc_p = proc_p.drop(columns=["__paper_text__"], errors="ignore")
    return proc_p.reset_index(drop=True)

def get_cpc_path() -> str:
    """
    로컬/서버 환경 모두에서 CPC CSV 경로를 안전하게 반환.
    1. 환경변수 CPC_CSV_PATH 가 있으면 그걸 우선 사용.
    2. macOS 로컬 기본 경로 확인.
    3. 프로젝트 내 /app/data/ 폴더 확인.
    4. 없으면 에러 발생.
    """
    # ① 환경변수 지정 시 우선 사용
    env_path = os.getenv("CPC_CSV_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # ② 로컬 macOS 기본 경로
    local_path = Path("/Users/hongseonah/PycharmProjects/SPRI2026/app/data/cpc_data_with_titles.csv")
    if local_path.exists():
        return str(local_path)

    # ③ 프로젝트 상대경로 (서버용 기본)
    project_path = Path(__file__).resolve().parent.parent / "data" / "cpc_data_with_titles.csv"
    if project_path.exists():
        return str(project_path)

    raise FileNotFoundError(
        "❌ CPC 데이터 파일을 찾을 수 없습니다.\n"
        " - 로컬에서는 app/data/cpc_data_with_titles.csv 에 두거나,\n"
        " - 서버에서는 환경변수 CPC_CSV_PATH 로 경로를 지정하세요."
    )


# =========================================================
# 공개 전처리 엔트리포인트 (확장: CPC 매칭 포함 가능)
# =========================================================

# app/core/preprocess.py (상단 import에 추가)
def run_preprocess(
    df: pd.DataFrame,
    cutoff_year: int = 2025,
    *,
    # ▼ 새 옵션들
    keyword_mode: Optional[Literal["literal", "semantic"]] = None,
    keywords: Optional[Sequence[str]] = None,
    # literal 모드 옵션
    case_insensitive: bool = True,
    use_regex: bool = False,
    # semantic 모드 옵션
    semantic_threshold: float = 0.30,
    semantic_model_name: str = "all-MiniLM-L6-v2",
    semantic_normalize: bool = True,
    semantic_device: Optional[str] = None,
    add_score_cols: bool = True,
    # ▼ CPC 매칭(새)
    do_cpc_match: bool = False,
    cpc_csv_path: Optional[str] = None,
    cpc_batch_size: int = 512,
    cpc_threshold: float = 0.30,
    cpc_device: Optional[str] = None,
    # ▼ 진행 콜백(새): (processed, total, stage)
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    전처리 + (옵션) 키워드 필터 + (옵션) CPC 매칭.
    progress_cb(processed:int, total:int, stage:str) 형태로 진행 상황 콜백을 호출.
    """
    def _ping(proc: int, tot: int, stage: str):
        if progress_cb:
            try:
                progress_cb(int(proc), int(tot), str(stage))
            except Exception:
                pass

    # 0) 시작
    total_in = len(df)
    _ping(0, max(total_in, 1), "preprocess_start")

    # 1) year 파생 + 빈행 제거
    out = _derive_year(df)
    title_col = "title" if "title" in out.columns else None
    abstr_col = "abstract" if "abstract" in out.columns else None

    if title_col:
        out[title_col] = out[title_col].astype(str).fillna("")
    if abstr_col:
        out[abstr_col] = out[abstr_col].astype(str).fillna("")

    if title_col and abstr_col:
        mask_keep = (out[title_col].str.len() > 0) | (out[abstr_col].str.len() > 0)
        out = out[mask_keep]
    elif title_col:
        out = out[out[title_col].str.len() > 0]
    elif abstr_col:
        out = out[out[abstr_col].str.len() > 0]

    _ping(len(out), max(total_in, 1), "clean_done")

    # 2) 키워드 필터(옵션)
    if keyword_mode and keywords:
        if keyword_mode == "literal":
            out = filter_df_by_keywords_literal(
                out, keywords,
                text_cols=tuple([c for c in ("title", "abstract") if c in out.columns]),
                case_insensitive=case_insensitive,
                use_regex=use_regex
            )
        elif keyword_mode == "semantic":
            # semantic은 기존 함수를 그대로 쓰되 완료 신호만 보냄
            out = filter_df_by_keywords_semantic(
                out, keywords,
                text_cols=tuple([c for c in ("title", "abstract") if c in out.columns]),
                model_name=semantic_model_name,
                threshold=semantic_threshold,
                normalize=semantic_normalize,
                device=semantic_device,
                add_score_cols=add_score_cols
            )
        else:
            raise ValueError("keyword_mode 는 None, 'literal', 'semantic' 중 하나여야 합니다.")
        _ping(len(out), max(total_in, 1), "keyword_filter_done")

    # 3) CPC 매칭(옵션) — 배치 진행률 콜백
    if do_cpc_match:
        if not cpc_csv_path:
            raise ValueError("do_cpc_match=True 인데 cpc_csv_path가 없습니다.")
        df_cpc = pd.read_csv(cpc_csv_path)
        if "SYMBOL" not in df_cpc.columns or "cpc_title" not in df_cpc.columns:
            raise ValueError("CPC CSV에는 'SYMBOL', 'cpc_title' 컬럼이 필요합니다.")

        # 클린
        def _clean(txt: str) -> str:
            import re
            if not isinstance(txt, str):
                return ""
            txt = txt.lower()
            txt = re.sub(r"[^a-z0-9#\+\/\-\.\s]", " ", txt)
            txt = re.sub(r"\s+", " ", txt).strip()
            return txt

        df_cpc = df_cpc.copy()
        df_cpc["cpc_title_clean"] = df_cpc["cpc_title"].fillna("").astype(str).map(_clean)

        use_cols = [c for c in ("title", "abstract") if c in out.columns]
        out = out.copy()
        out["__text__"] = out.apply(lambda r: " ".join([str(r[c]) for c in use_cols]), axis=1).fillna("").astype(str).map(_clean)

        # 임베딩 & 배치 매칭
        from sentence_transformers import SentenceTransformer, util
        import torch

        device = cpc_device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

        cpc_emb = model.encode(
            df_cpc["cpc_title_clean"].tolist(),
            convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False, device=device
        )
        SYMS = df_cpc["SYMBOL"].tolist()
        TITLES = df_cpc["cpc_title"].tolist()

        texts = out["__text__"].tolist()
        T = len(texts)
        _ping(0, max(T, 1), "cpc_match")

        bs = int(cpc_batch_size)
        best_sym = np.empty(T, dtype=object)
        best_desc = np.empty(T, dtype=object)
        best_score = np.empty(T, dtype=np.float32)

        for i in range(0, T, bs):
            j = min(i + bs, T)
            batch = texts[i:j]
            abs_emb = model.encode(
                batch, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False, device=device
            )
            sim = util.cos_sim(abs_emb, cpc_emb)            # [B x N_cpc]
            scores, idx = torch.max(sim, dim=1)

            idx_np = idx.detach().cpu().numpy()
            scores_np = scores.detach().cpu().numpy().astype(np.float32)

            best_sym[i:j] = [SYMS[k] for k in idx_np]
            best_desc[i:j] = [TITLES[k] for k in idx_np]
            best_score[i:j] = scores_np

            _ping(j, max(T, 1), "cpc_match")               # 🔹 배치 진행률 콜백

        out["matched_cpc"] = best_sym
        out["matched_cpc_description"] = best_desc
        out["similarity"] = best_score

        # threshold 적용
        if cpc_threshold is not None:
            def _final_row(sim, sym, desc):
                if float(sim) >= float(cpc_threshold):
                    return sym, desc
                return "exploration", "Exploratory Topic (Uncertain CPC Match)"
            final = [ _final_row(s, a, b) for s, a, b in zip(best_score, best_sym, best_desc) ]
            out["final_cpc"] = [x[0] for x in final]
            out["final_cpc_description"] = [x[1] for x in final]
        _ping(T, max(T, 1), "cpc_match_done")

    # 마무리
    out = out.drop(columns=["__text__"], errors="ignore").reset_index(drop=True)
    _ping(len(out), max(total_in, 1), "preprocess_done")
    return out
