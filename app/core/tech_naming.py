# app/core/tech_naming.py
# ============================================================
#  기술 네이밍 생성기 (Flow-Aggregated)
#  - OpenAI 호출 병렬화로 처리시간 단축 → 프록시/서버 타임아웃 리스크 완화
#  - 재시도: 지수 백오프 + 지터, 429/5xx 내성 강화
#  - 최종 네이밍 결과를 DataFrame/CSV 텍스트로 반환 (저장 로직은 주석 처리)
#  - 주요 튜닝 ENV:
#     * OUTPUT_DIR       (기본: /var/lib/app/outputs)
#     * LABEL_SUFFIX     (메타)
#     * METHOD           (A|H|F, 기본 A)
#     * OPENAI_MODEL     (기본: gpt-4o-mini)
#     * MAX_WORKERS      (동시 호출 수, 기본 4)
#     * REQUEST_TIMEOUT  (OpenAI 단건 타임아웃, 기본 60s)
#     * GPT_RETRY        (기본 4)
#     * BACKOFF_BASE     (기본 1.5)
# ============================================================

from __future__ import annotations
import os, re, json, time, random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- Env / Paths ----------------
OUTPUT_DIR   = Path(os.getenv("OUTPUT_DIR", "/var/lib/app/outputs")).resolve()
LABEL_SUFFIX = os.getenv("LABEL_SUFFIX", "k100")   # 메타 표기용
METHOD       = os.getenv("METHOD", "A").upper().strip()
OUT_HYBRID   = OUTPUT_DIR / "names_generated_hybrid.csv"   # (미사용: 호환 위해 남김)
OUT_FLOWAG   = OUTPUT_DIR / "names_generated_flowagg.csv"  # (미사용: 호환 위해 남김)

# 튜닝 파라미터
MAX_WORKERS       = max(1, int(os.getenv("MAX_WORKERS", "4")))
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT", "60"))
GPT_RETRY         = max(1, int(os.getenv("GPT_RETRY", "4")))
BACKOFF_BASE      = float(os.getenv("BACKOFF_BASE", "1.5"))

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

def _call_gpt(user_prompt: str, sys_prompt: str, retry: int = GPT_RETRY) -> Dict:
    """OpenAI 호출: 재시도(지수 백오프+지터), v1 우선"""
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
                    timeout=REQUEST_TIMEOUT_S,
                )
                raw = resp.choices[0].message.content or "{}"
            else:
                resp = _OPENAI_LEGACY.ChatCompletion.create(
                    model=OPENAI_MODEL,
                    messages=[{"role":"system","content":sys_prompt},
                              {"role":"user","content":user_prompt}],
                    temperature=0.35,
                    request_timeout=REQUEST_TIMEOUT_S
                )
                raw = resp["choices"][0]["message"]["content"]
            try: data = json.loads(raw)
            except Exception: data = _extract_json(raw)
            data = _normalize_keys(data); data["_raw_text"] = raw
            return data
        except Exception as e:
            last_err = e
            # 429/5xx/네트워크 등에 대해 백오프
            sleep_s = (BACKOFF_BASE ** i) + random.uniform(0.0, 0.5)
            time.sleep(sleep_s)
    # 재시도 실패
    raise last_err

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
    return panel

def _panel_from_artifacts_edges(flow_edges_df: pd.DataFrame) -> pd.DataFrame:
    return _build_panel_from_edges(flow_edges_df)

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

SYS_FLOWAG = (
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

def _load_cluster_docs_artifacts(artifacts: Dict[str, Any], year:int, cluster_id:int) -> pd.DataFrame:
    label_col = artifacts["label_col"]
    df = artifacts["clustered_by_year"].get(int(year), pd.DataFrame())
    if df.empty or label_col not in df.columns:
        return pd.DataFrame()
    return df[df[label_col] == int(cluster_id)].copy()

def _load_top_terms_artifacts(artifacts: Dict[str, Any], year:int, cluster_id:int, topk:int=40) -> List[str]:
    df = artifacts["tfidf_by_year"].get(int(year), pd.DataFrame())
    if df.empty or not {"cluster_id","term","tfidf"} <= set(df.columns): return []
    sub = df[df["cluster_id"]==int(cluster_id)].sort_values("tfidf", ascending=False).head(topk)
    return sub["term"].astype(str).tolist()

# ---- 병렬 실행 유틸
def _as_completed_results(futures):
    for fut in as_completed(futures):
        try:
            yield fut.result()
        except Exception as e:
            # 호출 단에서 메시지로 변환해 반환하도록 함
            yield {"status": f"error: {type(e).__name__}: {e}"}

def _run_flowagg(panel: pd.DataFrame, target_flows: List[int], artifacts: Dict[str, Any]) -> pd.DataFrame:
    """flow-aggregated 네이밍을 실행해 DataFrame을 반환 (파일 저장 없음)."""
    rows_out: List[Dict[str, Any]] = []

    def _task(fid: int):
        sub = panel[panel["flow_id"]==fid].sort_values("year")
        term_score: Dict[str,float] = {}
        for y, g in sub.groupby("year"):
            df = artifacts["tfidf_by_year"].get(int(y), pd.DataFrame())
            if df.empty or not {"cluster_id","term","tfidf"} <= set(df.columns): continue
            for cid in g["cluster_id"].astype(int).tolist():
                top = df[df["cluster_id"]==cid].sort_values("tfidf",ascending=False).head(80)
                for _, s in top.iterrows():
                    t = str(s["term"]).lower()
                    term_score[t] = term_score.get(t,0.0) + float(s["tfidf"])
        if not term_score:
            return {"flow_id": fid, "status":"no_terms"}
        terms = sorted(term_score.items(), key=lambda x:x[1], reverse=True)[:60]
        prompt = f"[flow_id] {fid}\n[flow_keywords] {', '.join([t for t,_ in terms])}"
        try:
            out = _call_gpt(prompt, SYS_FLOWAG)
            return {
                "flow_id": fid,
                "tech_name_ko": out.get("tech_name_ko",""),
                "tech_name_en": out.get("tech_name_en",""),
                "purpose": out.get("purpose",""),
                "method": out.get("method",""),
                "novelty": out.get("novelty",""),
                "rationale": out.get("rationale",""),
                "raw_text": out.get("_raw_text",""),
                "model": OPENAI_MODEL, "status": "ok"
            }
        except Exception as e:
            return {"flow_id": fid, "status": f"error: {type(e).__name__}: {e}"}

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_task, int(fid)) for fid in target_flows]
        for res in _as_completed_results(futures):
            rows_out.append(res)

    # (참고) 이전 버전은 파일로 저장했으나 지금은 반환만 합니다.
    # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # pd.DataFrame(rows_out).to_csv(OUT_FLOWAG, index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows_out)

#################################################################################
################################# top_n 지정 부분 #################################
#################################################################################
def run_tech_naming(_prompt_ignored: str | None = None, *,
                    artifacts: Optional[Dict[str, Any]] = None,
                    top_n: int = 10) -> Dict:

    if artifacts is None or not isinstance(artifacts, dict):
        raise RuntimeError("artifacts 가 필요합니다. run_clustering(...)[1]['artifacts'] 를 전달하세요.")

    # 1) 패널 구성(artifacts의 flow_edges_df 사용)
    fedges = artifacts.get("flow_edges_df", pd.DataFrame())
    if fedges.empty:
        raise RuntimeError("artifacts.flow_edges_df 가 비었습니다.")
    panel = _panel_from_artifacts_edges(fedges)
    if panel.empty:
        raise RuntimeError("flow_edges_df 로부터 panel 생성에 실패했습니다(비어있음).")
    panel["year"] = panel["year"].astype(int)

    # 2) 문서수 합 기준 상위 N개 flow 선정
    def _select_top_flows(panel_df: pd.DataFrame, n: int) -> List[int]:
        if "docs" not in panel_df.columns:
            panel_df["docs"] = 1
        agg = (panel_df.groupby("flow_id", as_index=False)["docs"].sum()
               .sort_values("docs", ascending=False)
               .head(n))
        return agg["flow_id"].astype(int).tolist()

    top_flows = _select_top_flows(panel, n=top_n)

    # 3) 플로우집계 네이밍 실행
    df_out = _run_flowagg(panel, top_flows, artifacts)
    csv_text = df_out.to_csv(index=False, encoding="utf-8-sig")

    # (참고) 파일 저장은 비활성화 (주석 처리)
    # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # (OUTPUT_DIR / "names_generated_flowagg.csv").write_text(csv_text, encoding="utf-8-sig")

    return {
        "df": df_out,
        "csv_text": csv_text,
        "meta": {
            "top_n": top_n,
            "max_workers": MAX_WORKERS,
            "request_timeout_s": REQUEST_TIMEOUT_S,
            "retry": GPT_RETRY,
            "backoff_base": BACKOFF_BASE,
            "method": "F",
            "label_suffix": LABEL_SUFFIX,
            "model": OPENAI_MODEL,
        }
    }
