# app/core/tech_naming.py
# ============================================================
#  기술 네이밍 생성기 (Hybrid + Flow-Aggregated)
#  - flow_id가 없는 패널(전이 엣지 표)도 자동으로 flow 재구성
#  - 상위 N개(문서수 기준) flow만 네이밍
# ============================================================

from __future__ import annotations
import os, re, json, time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd

# ---------------- Env / Paths ----------------
LABEL_SUFFIX = os.getenv("LABEL_SUFFIX", "k100")   # k5, k50, k100 ...
METHOD       = os.getenv("METHOD", "A")
ROOT         = Path(f"./cluster_out/compare_methods_{LABEL_SUFFIX}").resolve()
YEARLY_DIRS  = [Path("./cluster_out/yearly_v4").resolve(),
                Path("./cluster_out/yearly_v3").resolve(),
                Path("./cluster_out/yearly_v2").resolve()]

PANEL_EDGES  = ROOT / f"flow_year_overall_{METHOD}_{LABEL_SUFFIX}.csv"
OUT_HYBRID   = ROOT / "grading_v3/model_weak_upgrade/names_generated_hybrid.csv"
OUT_FLOWAG   = ROOT / "grading_v3/model_weak_upgrade/names_generated_flowagg.csv"

# ---------------- OpenAI ----------------
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_MODE = None
_CLIENT_V1 = None
_OPENAI_LEGACY = None

def _init_openai():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("환경변수 OPENAI_API_KEY 를 설정하세요.")
    try:
        from openai import OpenAI
        return "v1", OpenAI(api_key=key), None
    except Exception:
        import openai
        openai.api_key = key
        return "legacy", None, openai

def _ensure_openai():
    """지연 초기화: import 시점이 아닌 GPT 호출 직전 초기화"""
    global _MODE, _CLIENT_V1, _OPENAI_LEGACY
    if _MODE is None:
        _MODE, _CLIENT_V1, _OPENAI_LEGACY = _init_openai()

# ---------------- GPT 호출 및 파싱 ----------------
REQ_KEYS = ["tech_name_ko","tech_name_en","purpose","method","novelty","rationale"]

def _normalize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    alias = {
        "ko_name": "tech_name_ko", "kr_name": "tech_name_ko", "name_ko": "tech_name_ko",
        "en_name": "tech_name_en", "name_en": "tech_name_en",
        "summary": "purpose", "approach": "method",
        "innovation": "novelty", "justification": "rationale", "why": "rationale",
    }
    out = {}
    for k, v in d.items():
        kk = alias.get(k.strip(), k.strip())
        out[kk] = v
    for k in REQ_KEYS:
        out.setdefault(k, "")
        if out[k] is None: out[k] = ""
        if not isinstance(out[k], str):
            try: out[k] = json.dumps(out[k], ensure_ascii=False)
            except Exception: out[k] = str(out[k])
    return out

def _extract_json(txt: str) -> Dict[str, Any]:
    # 코드블록 제거 후 첫 {..} 파싱, 실패 시 key:value 라인 폴백
    t = re.sub(r"```(?:json)?\s*|\s*```", "", txt, flags=re.I)
    m = re.search(r"\{[\s\S]*?\}", t)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    kv = {}
    for line in t.splitlines():
        mm = re.match(r"\s*([A-Za-z0-9_]+)\s*:\s*(.+)", line)
        if mm:
            kv[mm.group(1)] = mm.group(2).strip()
    return kv if kv else {}

def _call_gpt(user_prompt: str, sys_prompt: str, retry=5, sleep=2.0) -> Dict:
    _ensure_openai()
    last_err = None
    for i in range(retry):
        try:
            if _MODE == "v1":
                resp = _CLIENT_V1.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role":"system","content":sys_prompt},
                              {"role":"user","content":user_prompt}],
                    temperature=0.35,
                    response_format={"type":"json_object"},
                    timeout=90,
                )
                raw = resp.choices[0].message.content or "{}"
            else:
                resp = _OPENAI_LEGACY.ChatCompletion.create(
                    model=OPENAI_MODEL,
                    messages=[{"role":"system","content":sys_prompt},
                              {"role":"user","content":user_prompt}],
                    temperature=0.35,
                    request_timeout=90
                )
                raw = resp["choices"][0]["message"]["content"]
            try: data = json.loads(raw)
            except Exception: data = _extract_json(raw)
            data = _normalize_keys(data); data["_raw_text"] = raw
            return data
        except Exception as e:
            last_err = e
            time.sleep(sleep*(i+1))
    raise last_err

# ---------------- 파일 탐색 유틸 ----------------
def _find_tfidf_file(year: int) -> Optional[Path]:
    for base in YEARLY_DIRS:
        f = base/str(year)/f"tfidf_top_terms_{LABEL_SUFFIX}.csv"
        if f.exists(): return f
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

def _find_year_clustered_file(year: int) -> Optional[Tuple[Path,str]]:
    for base in YEARLY_DIRS:
        ydir = base/str(year)
        if not ydir.exists(): continue
        for f in sorted(ydir.glob("*clustered_k*.csv")):
            try:
                dfh = pd.read_csv(f, nrows=2)
                lab = next((c for c in dfh.columns if re.fullmatch(r"cluster_k\d+", c)), None)
                if lab: return f, lab
            except: pass
    return None

# ---------------- 패널 재구성 ----------------
def _build_panel_from_edges(df_edges: pd.DataFrame) -> pd.DataFrame:
    need = {"year_from","year_to","prev_id","next_id"}
    if not need.issubset(df_edges.columns):
        raise ValueError(f"엣지 표에 필수 컬럼 부족: {need - set(df_edges.columns)}")
    forward, reverse, docs_map = {}, {}, {}
    for _, r in df_edges.iterrows():
        y1, y2 = int(r["year_from"]), int(r["year_to"])
        c1, c2 = int(r["prev_id"]), int(r["next_id"])
        forward[(y1, c1)] = (y2, c2)
        reverse[(y2, c2)] = (y1, c1)
        if "docs_from" in df_edges.columns and not pd.isna(r.get("docs_from", np.nan)):
            docs_map[(y1, c1)] = int(r["docs_from"])
        if "docs_to" in df_edges.columns and not pd.isna(r.get("docs_to", np.nan)):
            docs_map[(y2, c2)] = int(r["docs_to"])
    sources = [(y,c) for (y,c) in forward.keys() if (y,c) not in reverse]
    visited, rows, flow_counter = set(), [], 0
    def follow(start):
        nonlocal flow_counter
        path, cur = [], start
        while cur and cur not in visited:
            path.append(cur); visited.add(cur)
            cur = forward.get(cur)
        for (y, cid) in path:
            rows.append({"flow_id": flow_counter, "year": y, "cluster_id": cid})
        flow_counter += 1
    for s in sorted(sources): follow(s)
    all_nodes = set(list(forward.keys()) + list(forward.values()))
    for node in sorted(all_nodes):
        if node in visited: continue
        cur = node
        while cur in reverse: cur = reverse[cur]
        follow(cur)
    panel = pd.DataFrame(rows)
    if panel.empty: return panel
    panel["docs"] = panel.apply(lambda r: docs_map.get((int(r["year"]), int(r["cluster_id"])), np.nan), axis=1)
    missing = panel[panel["docs"].isna()]
    if not missing.empty:
        cache_counts: Dict[int, pd.DataFrame] = {}
        for y in sorted(panel["year"].unique()):
            f = _find_counts_file(int(y))
            if f and f.exists():
                dfc = pd.read_csv(f)
                if {"cluster_id","count"}.issubset(dfc.columns):
                    cache_counts[int(y)] = dfc[["cluster_id","count"]].rename(columns={"count":"docs"})
        def _fill_doc(row):
            if not pd.isna(row["docs"]): return row["docs"]
            y = int(row["year"]); cid = int(row["cluster_id"])
            dfc = cache_counts.get(y)
            if dfc is None: return np.nan
            hit = dfc[dfc["cluster_id"]==cid]
            return int(hit["docs"].values[0]) if not hit.empty else np.nan
        panel["docs"] = panel.apply(_fill_doc, axis=1)
    return panel

def _load_panel_or_build_flows() -> pd.DataFrame:
    if not PANEL_EDGES.exists():
        raise FileNotFoundError(f"패널/엣지 파일이 없습니다: {PANEL_EDGES}")
    df = pd.read_csv(PANEL_EDGES)
    if "flow_id" in df.columns and {"year","cluster_id"}.issubset(df.columns):
        return df[["flow_id","year","cluster_id"] + ([c for c in ["docs"] if c in df.columns])].copy()
    panel = _build_panel_from_edges(df)
    if panel.empty:
        raise RuntimeError("엣지 표로부터 flow 패널을 구성하지 못했습니다 (빈 결과).")
    return panel

# ---------------- 상위 Flow 선정 ----------------
MAX_FLOWS_FOR_TEST = 100
def _select_top_flows(panel: pd.DataFrame, top_n:int=MAX_FLOWS_FOR_TEST) -> List[int]:
    if "docs" not in panel.columns:
        panel["docs"] = 1
    agg = (panel.groupby("flow_id", as_index=False)["docs"].sum()
           .sort_values("docs", ascending=False)
           .head(top_n))
    return agg["flow_id"].astype(int).tolist()

# ---------------- 텍스트/키워드 로딩 ----------------
TITLE_COLS_CAND = ["title","paper_title","doc_title"]
ABSTR_COLS_CAND = ["abstract","summary","doc_abstract"]

def _first_existing_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns: return c
    return None

def _parse_embedding(x):
    if isinstance(x,(list,tuple,np.ndarray)): return np.asarray(x,dtype=np.float32)
    if isinstance(x,str):
        x=x.strip()
        try: return np.asarray(json.loads(x),dtype=np.float32)
        except:
            try:
                import ast
                return np.asarray(ast.literal_eval(x),dtype=np.float32)
            except: return None
    return None

def _load_cluster_docs(year:int, cluster_id:int) -> pd.DataFrame:
    found = _find_year_clustered_file(year)
    if not found: return pd.DataFrame()
    f, lab = found
    df = pd.read_csv(f)
    if lab not in df.columns: return pd.DataFrame()
    return df[df[lab]==cluster_id].copy()

def _load_top_terms(year:int, cluster_id:int, topk:int=40) -> List[str]:
    f = _find_tfidf_file(year)
    if not f: return []
    df = pd.read_csv(f)
    if not {"cluster_id","term","tfidf"}.issubset(df.columns): return []
    sub = df[df["cluster_id"]==cluster_id].sort_values("tfidf",ascending=False).head(topk)
    return sub["term"].astype(str).tolist()

def _rep_titles_via_embeddings(df_c: pd.DataFrame, n:int=3) -> List[str]:
    emb_col = next((c for c in df_c.columns if c.lower()=="embedding"), None)
    title_col = _first_existing_col(df_c, TITLE_COLS_CAND)
    if emb_col is None or title_col is None: return []
    embs = [v for v in df_c[emb_col].map(_parse_embedding) if v is not None]
    if not embs: return []
    E = np.vstack(embs).astype(np.float32)
    centroid = E.mean(axis=0)
    sims = (E @ centroid) / (np.linalg.norm(E,axis=1)*(np.linalg.norm(centroid)+1e-9)+1e-9)
    df2 = df_c.iloc[:len(sims)].copy()
    df2["__sim__"] = sims
    return df2.sort_values("__sim__",ascending=False)[title_col].head(n).astype(str).tolist()

# ---------------- 네이밍 실행 ----------------
TOPK_KEYWORDS = 40
N_REP_TITLES  = 3

SYS_HYBRID = (
    "당신은 기술 네이밍 비서입니다. 입력된 클러스터의 키워드와 대표 타이틀을 보고 "
    "① 목적, ② 구현 방법, ③ 신규 기여를 간결히 요약하고, 한국어/영문 기술명을 JSON으로만 출력하세요.\n"
    "반드시 아래 키를 포함한 JSON만 출력하세요:\n"
    '{"tech_name_ko":"","tech_name_en":"","purpose":"","method":"","novelty":"","rationale":""}'
)

SYS_FLOWAG = (
    "당신은 기술 네이밍 컨설턴트입니다. 흐름 전체의 상위 키워드를 보고 "
    "목적/방법/신규기여를 요약하고 한국어/영문 기술명을 JSON으로만 출력하세요.\n"
    "반드시 아래 키를 포함한 JSON만 출력하세요:\n"
    '{"tech_name_ko":"","tech_name_en":"","purpose":"","method":"","novelty":"","rationale":""}'
)

def _run_hybrid(panel: pd.DataFrame, target_flows: List[int]) -> Path:
    rows_out = []
    last_nodes = (panel.sort_values("year").groupby("flow_id").tail(1).reset_index(drop=True))
    last_nodes = last_nodes[last_nodes["flow_id"].isin(target_flows)]
    for _, r in last_nodes.iterrows():
        fid = int(r["flow_id"]); y = int(r["year"]); cid = int(r["cluster_id"])
        keywords = _load_top_terms(y, cid, topk=TOPK_KEYWORDS)
        docs_df = _load_cluster_docs(y, cid)
        titles  = _rep_titles_via_embeddings(docs_df, n=N_REP_TITLES) if not docs_df.empty else []
        prompt = (
            f"[flow_id] {fid}\n[year] {y}\n"
            f"[keywords] {', '.join(keywords[:TOPK_KEYWORDS])}\n"
            f"[rep_titles] {'; '.join(titles[:N_REP_TITLES])}"
        )
        try:
            out = _call_gpt(prompt, SYS_HYBRID)
            rows_out.append({
                "flow_id": fid, "year": y, "cluster_id": cid,
                "tech_name_ko": out.get("tech_name_ko",""),
                "tech_name_en": out.get("tech_name_en",""),
                "purpose": out.get("purpose",""),
                "method": out.get("method",""),
                "novelty": out.get("novelty",""),
                "rationale": out.get("rationale",""),
                "raw_text": out.get("_raw_text",""),
                "model": OPENAI_MODEL, "status": "ok"
            })
        except Exception as e:
            rows_out.append({"flow_id": fid, "year": y, "cluster_id": cid,
                             "status": f"error: {type(e).__name__}: {e}"})
        time.sleep(1.5)
    pd.DataFrame(rows_out).to_csv(OUT_HYBRID, index=False, encoding="utf-8-sig")
    return OUT_HYBRID

def _run_flowagg(panel: pd.DataFrame, target_flows: List[int]) -> Path:
    rows_out = []
    for fid in target_flows:
        sub = panel[panel["flow_id"]==fid].sort_values("year")
        term_score: Dict[str,float] = {}
        for y, g in sub.groupby("year"):
            f = _find_tfidf_file(int(y))
            if not f: continue
            df = pd.read_csv(f)
            if not {"cluster_id","term","tfidf"}.issubset(df.columns): continue
            for cid in g["cluster_id"].astype(int).tolist():
                top = df[df["cluster_id"]==cid].sort_values("tfidf",ascending=False).head(80)
                for _, s in top.iterrows():
                    t = str(s["term"]).lower()
                    term_score[t] = term_score.get(t,0.0) + float(s["tfidf"])
        if not term_score:
            rows_out.append({"flow_id": fid, "status":"no_terms"}); continue
        terms = sorted(term_score.items(), key=lambda x:x[1], reverse=True)[:60]
        prompt = f"[flow_id] {fid}\n[flow_keywords] {', '.join([t for t,_ in terms])}"
        try:
            out = _call_gpt(prompt, SYS_FLOWAG)
            rows_out.append({
                "flow_id": fid,
                "tech_name_ko": out.get("tech_name_ko",""),
                "tech_name_en": out.get("tech_name_en",""),
                "purpose": out.get("purpose",""),
                "method": out.get("method",""),
                "novelty": out.get("novelty",""),
                "rationale": out.get("rationale",""),
                "raw_text": out.get("_raw_text",""),
                "model": OPENAI_MODEL, "status": "ok"
            })
        except Exception as e:
            rows_out.append({"flow_id": fid, "status": f"error: {type(e).__name__}: {e}"})
        time.sleep(1.5)
    pd.DataFrame(rows_out).to_csv(OUT_FLOWAG, index=False, encoding="utf-8-sig")
    return OUT_FLOWAG

# ---------------- 진입점 ----------------
def run_tech_naming(_prompt_ignored: str | None = None) -> Dict:
    # 1) 패널 로드(또는 엣지→패널 변환)
    panel = _load_panel_or_build_flows()
    panel["year"] = panel["year"].astype(int)

    # 2) 문서수 합 기준 상위 N개 flow 선정
    top_flows = _select_top_flows(panel, top_n=10)

    # 3) 하이브리드 & 플로우집계 네이밍 실행
    OUT_HYBRID.parent.mkdir(parents=True, exist_ok=True)
    OUT_FLOWAG.parent.mkdir(parents=True, exist_ok=True)
    hybrid_path = _run_hybrid(panel, top_flows)
    flowag_path = _run_flowagg(panel, top_flows)

    return {"paths": {"hybrid_csv": str(hybrid_path), "flowagg_csv": str(flowag_path)}}
