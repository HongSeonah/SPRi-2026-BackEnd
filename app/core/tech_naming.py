# app/core/tech_naming.py
# ============================================================
#  클러스터 네이밍 생성기 (Hybrid + Flow-Aggregated)
#  - Pipeline과 호환: run_tech_naming(prompt: str | None) -> dict
#  - 출력:
#      ./cluster_out/compare_methods_{LABEL_SUFFIX}/grading_v3/model_weak_upgrade/names_generated_hybrid.csv
#      ./cluster_out/compare_methods_{LABEL_SUFFIX}/grading_v3/model_weak_upgrade/names_generated_flowagg.csv
#  - PANEL(패널) 자동 탐지 지원: flow_year_overall_* 또는 flow_year_metrics_* 파일 스캔
# ============================================================

from __future__ import annotations

import os
import re
import json
import time
from glob import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

# ---------------- Env / Paths ----------------
LABEL_SUFFIX = os.getenv("LABEL_SUFFIX", "k100")   # ex) k50 / k100 / varK
METHOD       = os.getenv("METHOD", "A")
ROOT         = Path(f"./cluster_out/compare_methods_{LABEL_SUFFIX}").resolve()
YEARLY_DIRS  = [
    Path("./cluster_out/yearly_v4").resolve(),
    Path("./cluster_out/yearly_v3").resolve(),
    Path("./cluster_out/yearly_v2").resolve(),
]

# 기본 파일 경로(실행 시 _resolve_paths로 최신 패널/루트 재설정)
PANEL        = ROOT / f"flow_year_overall_{METHOD}_{LABEL_SUFFIX}.csv"
OUT_HYBRID   = ROOT / "grading_v3/model_weak_upgrade/names_generated_hybrid.csv"
OUT_FLOWAG   = ROOT / "grading_v3/model_weak_upgrade/names_generated_flowagg.csv"

# 약신호/승격 후보 목록(없을 수 있음)
WEAK_FILE    = ROOT / "grading_v3/model_weak_upgrade/pred_WEAK_2024.csv"
UPGR_FILE    = ROOT / "grading_v3/model_weak_upgrade/pred_upgrade_from_WEAK_2024.csv"

# ---------------- OpenAI Client ----------------
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")

def _init_openai():
    """
    OpenAI Chat Completions 클라이언트 초기화.
    'openai' 레거시와 'openai.OpenAI' 신버전 모두 지원.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("환경변수 OPENAI_API_KEY 가 설정되어 있지 않습니다.")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        return ("v1", client, None)
    except Exception:
        import openai
        openai.api_key = OPENAI_API_KEY
        return ("legacy", None, openai)

_MODE, _CLIENT_V1, _OPENAI_LEGACY = _init_openai()

# ---------------- 경로 자동 탐지 ----------------
def _autodetect_root_and_panel() -> Tuple[Path, Path]:
    # 현재 설정 우선
    cur_root = ROOT
    cur_panel = cur_root / f"flow_year_overall_{METHOD}_{LABEL_SUFFIX}.csv"
    if cur_panel.exists():
        return cur_root, cur_panel

    # 후보: overall → metrics 순으로 스캔
    candidates: List[Path] = []
    candidates += [Path(p) for p in glob("cluster_out/compare_methods_*/flow_year_overall_*_k*.csv")]
    if not candidates:
        candidates += [Path(p) for p in glob("cluster_out/compare_methods_*/flow_year_metrics_*_k*.csv")]

    if not candidates:
        return cur_root, cur_panel  # 발견 실패 → 기존 값 유지

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    chosen_panel = candidates[0]
    chosen_root = chosen_panel.parent
    return chosen_root.resolve(), chosen_panel.resolve()

def _refresh_outputs_for_root() -> None:
    global OUT_HYBRID, OUT_FLOWAG
    OUT_HYBRID = ROOT / "grading_v3/model_weak_upgrade/names_generated_hybrid.csv"
    OUT_FLOWAG = ROOT / "grading_v3/model_weak_upgrade/names_generated_flowagg.csv"

def _resolve_paths() -> None:
    """현재 LABEL_SUFFIX/METHOD와 무관하게 실제 생성된 최신 패널을 자동 인식."""
    global ROOT, PANEL
    det_root, det_panel = _autodetect_root_and_panel()
    if det_panel.exists():
        ROOT = det_root
        PANEL = det_panel
    _refresh_outputs_for_root()

# ---------------- 공통 유틸 ----------------
TITLE_COLS_CAND = ["title", "paper_title", "doc_title"]
ABSTR_COLS_CAND = ["abstract", "summary", "doc_abstract"]

def _short(s: str, L:int=28) -> str:
    s = str(s) if s is not None else ""
    return s if len(s) <= L else s[:L] + "…"

def _first_existing_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None

def _find_year_clustered_file(year: int) -> Optional[Tuple[Path, str]]:
    """
    해당 연도 폴더에서 *_clustered_k*.csv 중 하나를 찾고,
    라벨 컬럼명(cluster_k###)을 함께 반환.
    """
    for base in YEARLY_DIRS:
        ydir = base / str(year)
        if not ydir.exists():
            continue
        cands = sorted(ydir.glob("*clustered_k*.csv"))
        for f in cands:
            try:
                dfh = pd.read_csv(f, nrows=2)
                lab = next((c for c in dfh.columns if re.fullmatch(r"cluster_k\d+", c)), None)
                if lab:
                    return f, lab
            except Exception:
                continue
    return None

def _find_tfidf_file(year: int) -> Optional[Path]:
    # 1) 고정 접미 우선
    for base in YEARLY_DIRS:
        f = base / str(year) / f"tfidf_top_terms_{LABEL_SUFFIX}.csv"
        if f.exists(): return f
    # 2) 가변 k 지원
    for base in YEARLY_DIRS:
        hits = sorted((base/str(year)).glob("tfidf_top_terms_k*.csv"))
        if hits: return hits[0]
    return None

def _find_counts_file(year: int) -> Optional[Path]:
    for base in YEARLY_DIRS:
        f = base/str(year)/f"cluster_counts_{LABEL_SUFFIX}.csv"
        if f.exists(): return f
    for base in YEARLY_DIRS:
        hits = sorted((base/str(year)).glob("cluster_counts_k*.csv"))
        if hits: return hits[0]
    return None

def _load_counts_map(year: int) -> Dict[int, int]:
    f = _find_counts_file(year)
    if not f: return {}
    df = pd.read_csv(f)
    if "cluster_id" not in df.columns or "count" not in df.columns:
        return {}
    return {int(r["cluster_id"]): int(r["count"]) for _, r in df.iterrows()}

def _parse_embedding(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, str):
        x = x.strip()
        try:
            return np.asarray(json.loads(x), dtype=np.float32)
        except Exception:
            try:
                import ast
                return np.asarray(ast.literal_eval(x), dtype=np.float32)
            except Exception:
                return None
    return None

# ---------------- GPT 호출 ----------------
_SYS_PROMPT_HYBRID = (
    "당신은 기술 네이밍 비서입니다. 입력된 클러스터의 키워드와 대표 타이틀을 보고 "
    "① 기술의 '목적', ② '구현 방법', ③ '신규 기여'를 간결히 도출한 다음, "
    "이를 근거로 한국어/영문 기술명을 제안하세요.\n\n"
    "출력은 반드시 JSON만 반환:\n"
    "{"
    "\"tech_name_ko\":\"(18자 이내, 고유명/축약 가능)\","
    "\"tech_name_en\":\"(3~5 words, 명료한 표현)\","
    "\"purpose\":\"~을/를 위한 ~\","
    "\"method\":\"핵심 접근/알고리즘/데이터/시스템\","
    "\"novelty\":\"기존 대비 차별점(1~2문장)\","
    "\"rationale\":\"이 이름을 선택한 이유(한국어 1문장)\""
    "}\n"
    "금지: 과도한 일반명사, 과장, 특정 상표명 차용."
)

_SYS_PROMPT_FLOWAG = (
    "역할: 기술 네이밍 컨설턴트.\n"
    "입력된 '흐름 전체(여러 연도·클러스터)에서 집계한 상위 키워드'를 보고 "
    "① 기술의 목적, ② 구현 방법, ③ 신규 기여 를 간결하게 요약하라.\n"
    "그 요약을 근거로 **세부 도메인(특정 질병/제품/업체/수치/데이터셋명 등)은 피하고**, "
    "**상위 개념·보편적 표현**으로 한국어/영문 기술명을 1개씩 제안하라.\n\n"
    "반드시 JSON만 출력:\n"
    "{"
    "\"tech_name_ko\":\"(12~18자, 과도한 세부명사 금지)\","
    "\"tech_name_en\":\"(3~5 words, domain-agnostic)\","
    "\"purpose\":\"~을/를 위한 ~\","
    "\"method\":\"핵심 접근/알고리즘/데이터/시스템\","
    "\"novelty\":\"기존 대비 차별점(1~2문장)\","
    "\"rationale\":\"왜 이런 이름인지 (한국어 1문장)\""
    "}"
)

def _extract_json(txt: str) -> str:
    m = re.search(r"\{.*\}", txt, flags=re.S)
    return m.group(0) if m else txt

def _call_gpt(user_prompt: str, sys_prompt: str, retry:int=3, sleep:float=1.4) -> Dict:
    last_err = None
    for i in range(retry):
        try:
            if _MODE == "v1":
                resp = _CLIENT_V1.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.35,
                    response_format={"type":"json_object"},
                )
                txt = resp.choices[0].message.content or "{}"
            else:
                resp = _OPENAI_LEGACY.ChatCompletion.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.35,
                )
                txt = resp["choices"][0]["message"]["content"]
            return json.loads(_extract_json(txt))
        except Exception as e:
            last_err = e
            time.sleep(sleep * (i+1))
    raise last_err

# ---------------- Hybrid: 최신년 클러스터 단위 ----------------
TOPK_KEYWORDS      = 40   # 클러스터 키워드 수
N_REP_TITLES       = 5    # 대표 타이틀 수
USE_EMBEDDINGS     = True # 임베딩 있으면 사용
AGG_WINDOW: Optional[int] = None  # FlowAgg에서만 사용

def _load_top_terms(year: int, cluster_id: int, topk:int=TOPK_KEYWORDS) -> List[str]:
    f = _find_tfidf_file(year)
    if not f:
        return []
    df = pd.read_csv(f)
    if not {"cluster_id","term","tfidf"} <= set(df.columns):
        return []
    sub = df[df["cluster_id"] == cluster_id].sort_values("tfidf", ascending=False).head(topk)
    return sub["term"].astype(str).tolist()

def _load_cluster_docs(year: int, cluster_id: int) -> pd.DataFrame:
    found = _find_year_clustered_file(year)
    if not found:
        return pd.DataFrame()
    f, lab = found
    df = pd.read_csv(f)
    if lab not in df.columns:
        return pd.DataFrame()
    return df[df[lab] == cluster_id].copy()

def _rep_titles_via_embeddings(df_c: pd.DataFrame, n:int=N_REP_TITLES) -> List[str]:
    emb_col = next((c for c in df_c.columns if c.lower() == "embedding"), None)
    if emb_col is None:
        return []
    title_col = _first_existing_col(df_c, TITLE_COLS_CAND)
    if title_col is None:
        return []
    embs = []
    for _, r in df_c.iterrows():
        v = _parse_embedding(r[emb_col])
        if v is not None:
            embs.append(v)
    if not embs:
        return []
    E = np.vstack(embs).astype(np.float32)
    centroid = E.mean(axis=0)
    sims = (E @ centroid) / (np.linalg.norm(E, axis=1) * (np.linalg.norm(centroid) + 1e-9) + 1e-9)
    df_c2 = df_c.iloc[:len(sims)].copy()
    df_c2["__sim__"] = sims
    top = df_c2.sort_values("__sim__", ascending=False).head(n)
    return top[title_col].astype(str).tolist()

def _rep_titles_via_tfidf(df_c: pd.DataFrame, keywords: List[str], n:int=N_REP_TITLES) -> List[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    title_col = _first_existing_col(df_c, TITLE_COLS_CAND)
    abstr_col = _first_existing_col(df_c, ABSTR_COLS_CAND)
    if title_col is None and abstr_col is None:
        return []
    texts = (df_c[title_col].fillna("") + " " + (df_c[abstr_col].fillna("") if abstr_col else "")).astype(str)
    if texts.empty:
        return []
    vec = TfidfVectorizer(max_features=120000, ngram_range=(1,2), min_df=2, max_df=0.95)
    X = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()
    if not keywords:
        scores = np.asarray(X.mean(axis=1)).ravel()
    else:
        kw_set = set([k.lower() for k in keywords])
        w = np.zeros(len(vocab), dtype=np.float32)
        for i, t in enumerate(vocab):
            if t in kw_set:
                w[i] = 1.0
        scores = (X @ w).A.ravel() if hasattr(X @ w, "A") else np.asarray(X @ w).ravel()
    order = np.argsort(-scores)[:n]
    return df_c.iloc[order][title_col].astype(str).tolist()

def _latest_info(panel: pd.DataFrame, flow_id: str) -> Optional[Dict]:
    rows = panel[panel["flow_id"] == flow_id]
    if rows.empty:
        return None
    r = rows.sort_values("year").iloc[-1]
    year = int(r["year"])
    cid_col = next((c for c in rows.columns if re.fullmatch(r"cluster_id|cluster_k\d+", c)), None)
    cid = int(r[cid_col]) if cid_col and not pd.isna(r[cid_col]) else None
    name = r["cluster_name"] if "cluster_name" in rows.columns else None
    return dict(year=year, cluster_id=cid, cluster_name=name)

def _build_prompt_hybrid(flow_id: str, cluster_name: Optional[str], year: int,
                         keywords: List[str], rep_titles: List[str]) -> str:
    kw = ", ".join(keywords[:TOPK_KEYWORDS]) if keywords else ""
    tl = "; ".join([_short(t, 90) for t in rep_titles[:N_REP_TITLES]]) if rep_titles else ""
    title = cluster_name or flow_id
    return (
        f"[기술 흐름] {flow_id}\n"
        f"[최신 연도] {year}\n"
        f"[클러스터명(있으면)] {title}\n"
        f"[핵심 키워드] {kw}\n"
        f"[대표 타이틀] {tl}\n\n"
        "위 내용을 바탕으로 ①목적 ②구현 방법 ③신규 기여를 먼저 요약하고, "
        "그 3요소를 반영한 한국어/영문 기술명을 만들어 JSON으로만 출력하세요."
    )

def _run_hybrid() -> Path:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "grading_v3/model_weak_upgrade").mkdir(parents=True, exist_ok=True)

    # 입력 로드
    if not PANEL.exists():
        # 패널이 없으면 생성 불가 → 빈 CSV라도 생성
        pd.DataFrame(columns=["flow_id","status"]).to_csv(OUT_HYBRID, index=False, encoding="utf-8-sig")
        return OUT_HYBRID

    panel = pd.read_csv(PANEL)
    if "year" in panel.columns:
        panel["year"] = panel["year"].astype(int)

    weak = pd.read_csv(WEAK_FILE) if WEAK_FILE.exists() else pd.DataFrame(columns=["flow_id"])
    upgr = pd.read_csv(UPGR_FILE) if UPGR_FILE.exists() else pd.DataFrame(columns=["flow_id"])

    def _top20(df: pd.DataFrame, score_cols: List[str]) -> pd.DataFrame:
        cols = list(set(df.columns) & set(score_cols))
        if not cols:
            return df.head(0).assign(list_type="EMPTY")
        return df.sort_values(cols, ascending=False).head(20)

    weak_top20 = _top20(weak, ["p_WEAK_y"]).assign(list_type="WEAK_TOP20")
    upgr_top20 = _top20(upgr, ["p_upgrade_2y"]).assign(list_type="UPGRADE_TOP20")

    targets = (pd.concat([
        weak_top20[["flow_id","cluster_name","list_type"]],
        upgr_top20[["flow_id","cluster_name","list_type"]],
    ], ignore_index=True).drop_duplicates(subset=["flow_id","list_type"]))

    rows_out: List[Dict[str, Any]] = []
    for _, r in targets.iterrows():
        fid = r["flow_id"]
        info = _latest_info(panel, fid)
        if info is None:
            rows_out.append({"flow_id": fid, "list_type": r["list_type"], "status": "no_panel_info"})
            continue
        year = info["year"]; cid = info["cluster_id"]; cname = info["cluster_name"] or r.get("cluster_name")
        if cid is None:
            rows_out.append({"flow_id": fid, "list_type": r["list_type"], "year": year,
                             "cluster_name_src": cname, "status": "no_cluster_id"})
            continue

        keywords = _load_top_terms(year, cid, topk=TOPK_KEYWORDS)

        df_c = _load_cluster_docs(year, cid)
        rep_titles: List[str] = []
        if not df_c.empty:
            if USE_EMBEDDINGS:
                rep_titles = _rep_titles_via_embeddings(df_c, n=N_REP_TITLES)
            if not rep_titles:
                rep_titles = _rep_titles_via_tfidf(df_c, keywords, n=N_REP_TITLES)

        prompt = _build_prompt_hybrid(fid, cname, year, keywords, rep_titles)

        try:
            out = _call_gpt(prompt, _SYS_PROMPT_HYBRID)
            rows_out.append({
                "flow_id": fid,
                "list_type": r["list_type"],
                "year": year,
                "cluster_id": cid,
                "cluster_name_src": cname,
                "tech_name_ko": out.get("tech_name_ko"),
                "tech_name_en": out.get("tech_name_en"),
                "purpose": out.get("purpose"),
                "method": out.get("method"),
                "novelty": out.get("novelty"),
                "rationale": out.get("rationale"),
                "keywords_used": ", ".join(keywords[:TOPK_KEYWORDS]),
                "rep_titles_used": "; ".join(rep_titles[:N_REP_TITLES]),
                "model": OPENAI_MODEL,
                "status": "ok"
            })
        except Exception as e:
            rows_out.append({
                "flow_id": fid, "list_type": r["list_type"], "year": year,
                "cluster_id": cid, "cluster_name_src": cname,
                "status": f"error: {type(e).__name__}: {e}"
            })

    OUT_HYBRID.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(OUT_HYBRID, index=False, encoding="utf-8-sig")
    return OUT_HYBRID

# ---------------- Flow-Aggregated: 흐름 단위 집계 ----------------
TOPK_PER_CLUSTER     = 80
TOPK_FLOW_KEYWORDS   = 60
DOC_WEIGHT_NORM_BY_YEAR = True

def _aggregate_flow_keywords(panel: pd.DataFrame, flow_id: str) -> Tuple[List[Tuple[str, float]], int, int]:
    rows = panel[panel["flow_id"] == flow_id].copy()
    if rows.empty:
        return [], 0, 0
    rows["year"] = rows["year"].astype(int)
    rows = rows.sort_values("year")

    # AGG_WINDOW 적용 (None이면 전체)
    if isinstance(AGG_WINDOW, int) and AGG_WINDOW > 0:
        y_max = rows["year"].max()
        rows = rows[rows["year"] >= (y_max - AGG_WINDOW + 1)]

    cid_col = next((c for c in rows.columns if re.fullmatch(r"cluster_id|cluster_k\d+", c)), None)
    if cid_col is None:
        return [], 0, 0

    term_score: Dict[str, float] = {}
    year_used, pairs_used = set(), 0

    for y, g in rows.groupby("year"):
        f_tfidf = _find_tfidf_file(int(y))
        if not f_tfidf:
            continue
        df_tf = pd.read_csv(f_tfidf)
        if not {"cluster_id","term","tfidf"} <= set(df_tf.columns):
            continue

        doc_map = _load_counts_map(int(y))
        maxd = max(doc_map.values()) if (DOC_WEIGHT_NORM_BY_YEAR and doc_map) else 1

        for _, rr in g.iterrows():
            cid = rr[cid_col]
            if pd.isna(cid):
                continue
            cid = int(cid)
            sub = (df_tf[df_tf["cluster_id"] == cid]
                   .sort_values("tfidf", ascending=False)
                   .head(TOPK_PER_CLUSTER))
            if sub.empty:
                continue
            w_docs = (doc_map.get(cid, 1) / maxd) if DOC_WEIGHT_NORM_BY_YEAR else (doc_map.get(cid, 1) or 1)
            for _, s in sub.iterrows():
                t = str(s["term"]).lower()
                sc = float(s["tfidf"]) * float(w_docs)
                term_score[t] = term_score.get(t, 0.0) + sc
            year_used.add(int(y))
            pairs_used += 1

    if not term_score:
        return [], len(year_used), pairs_used

    items = sorted(term_score.items(), key=lambda x: x[1], reverse=True)
    return items[:TOPK_FLOW_KEYWORDS], len(year_used), pairs_used

def _build_prompt_flowagg(flow_id: str, flow_keywords: List[Tuple[str, float]], meta_line: str) -> str:
    kw = ", ".join([t for t, _ in flow_keywords])
    return (
        f"[기술 흐름] {flow_id}\n"
        f"[메타] {meta_line}\n"
        f"[흐름-집계 키워드 (넓은 범위)] {kw}\n\n"
        "위 키워드는 여러 연도·클러스터에서 집계한 상위 개념 용어입니다. "
        "지나치게 특정 응용(질병명/제품명/업체명/데이터셋명/수치 등)은 피하고, "
        "상위 기술 개념을 중심으로 ①목적 ②구현방법 ③신규기여를 간단히 요약한 뒤, "
        "그 3요소를 반영한 한국어/영문 기술명을 만들어 JSON으로만 출력하세요."
    )

def _run_flow_agg() -> Path:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "grading_v3/model_weak_upgrade").mkdir(parents=True, exist_ok=True)

    if not PANEL.exists():
        pd.DataFrame(columns=["flow_id","status"]).to_csv(OUT_FLOWAG, index=False, encoding="utf-8-sig")
        return OUT_FLOWAG

    panel = pd.read_csv(PANEL)
    if "year" in panel.columns:
        panel["year"] = panel["year"].astype(int)

    weak = pd.read_csv(WEAK_FILE) if WEAK_FILE.exists() else pd.DataFrame(columns=["flow_id"])
    upgr = pd.read_csv(UPGR_FILE) if UPGR_FILE.exists() else pd.DataFrame(columns=["flow_id"])

    def _top20(df: pd.DataFrame, score_cols: List[str]) -> pd.DataFrame:
        cols = list(set(df.columns) & set(score_cols))
        if not cols:
            return df.head(0).assign(list_type="EMPTY")
        return df.sort_values(cols, ascending=False).head(20)

    weak_top20 = _top20(weak, ["p_WEAK_y"]).assign(list_type="WEAK_TOP20")
    upg_top20  = _top20(upgr, ["p_upgrade_2y"]).assign(list_type="UPGRADE_TOP20")

    targets = (pd.concat([
        weak_top20[["flow_id","cluster_name","list_type"]],
        upg_top20[["flow_id","cluster_name","list_type"]],
    ], ignore_index=True).drop_duplicates(subset=["flow_id","list_type"]))

    rows_out: List[Dict[str, Any]] = []
    for _, r in targets.iterrows():
        fid = r["flow_id"]
        agg_terms, n_years, n_pairs = _aggregate_flow_keywords(panel, fid)
        if not agg_terms:
            rows_out.append({"flow_id": fid, "list_type": r["list_type"], "status": "no_terms_collected"})
            continue

        meta_line = f"years_used={n_years}, pairs_used={n_pairs}, top_terms={len(agg_terms)}"
        prompt = _build_prompt_flowagg(fid, agg_terms, meta_line)

        try:
            out = _call_gpt(prompt, _SYS_PROMPT_FLOWAG)
            rows_out.append({
                "flow_id": fid,
                "list_type": r["list_type"],
                "tech_name_ko": out.get("tech_name_ko"),
                "tech_name_en": out.get("tech_name_en"),
                "purpose": out.get("purpose"),
                "method": out.get("method"),
                "novelty": out.get("novelty"),
                "rationale": out.get("rationale"),
                "keywords_used": ", ".join([t for t,_ in agg_terms]),
                "n_years_used": n_years,
                "n_pairs_used": n_pairs,
                "model": OPENAI_MODEL,
                "status": "ok"
            })
        except Exception as e:
            rows_out.append({
                "flow_id": fid, "list_type": r["list_type"],
                "status": f"error: {type(e).__name__}: {e}"
            })

    OUT_FLOWAG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(OUT_FLOWAG, index=False, encoding="utf-8-sig")
    return OUT_FLOWAG

# ---------------- Public API ----------------
def run_tech_naming(_prompt_ignored: str | None = None) -> Dict:
    """
    Pipeline 호환 API.
    - 내부적으로 Hybrid / Flow-Aggregated 두 방식을 모두 실행
    - 반환: {"paths": {...}} (파이프라인에서는 경로만 저장하면 됨)
    """
    _resolve_paths()  # 최신 PANEL/ROOT 자동 인식
    hybrid_path = _run_hybrid()
    flowag_path = _run_flow_agg()
    return {
        "paths": {
            "hybrid_csv": str(hybrid_path),
            "flowagg_csv": str(flowag_path),
        }
    }
