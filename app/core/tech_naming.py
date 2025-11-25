# ============================================================
#  tech_naming.py
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
LABEL_SUFFIX = os.getenv("LABEL_SUFFIX", "k100")
METHOD       = os.getenv("METHOD", "A").upper().strip()

MAX_WORKERS       = max(1, int(os.getenv("MAX_WORKERS", "4")))
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT", "60"))
GPT_RETRY         = max(1, int(os.getenv("GPT_RETRY", "4")))
BACKOFF_BASE      = float(os.getenv("BACKOFF_BASE", "1.5"))

OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_MODE = None
_CLIENT_V1 = None
_OPENAI_LEGACY = None


# ============================================================
#  OpenAI Module
# ============================================================

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
    global _MODE, _CLIENT_V1, _OPENAI_LEGACY
    if _MODE is None:
        _MODE, _CLIENT_V1, _OPENAI_LEGACY = _init_openai()


# ============================================================
#  GPT Helpers
# ============================================================

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
            except: out[k] = str(out[k])
    return out

def _extract_json(txt: str) -> Dict[str, Any]:
    t = re.sub(r"```(?:json)?\s*|\s*```", "", txt, flags=re.I)
    m = re.search(r"\{[\s\S]*?\}", t)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    kv = {}
    for line in t.splitlines():
        mm = re.match(r"\s*([A-Za-z0-9_]+)\s*:\s*(.+)", line)
        if mm:
            kv[mm.group(1)] = mm.group(2).strip()
    return kv if kv else {}

def _call_gpt(user_prompt: str, sys_prompt: str, retry: int = GPT_RETRY) -> Dict:
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
                    timeout=REQUEST_TIMEOUT_S
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
            except: data = _extract_json(raw)
            data = _normalize_keys(data); data["_raw_text"] = raw
            return data

        except Exception as e:
            last_err = e
            time.sleep((BACKOFF_BASE ** i) + random.uniform(0.0, 0.5))

    raise last_err


# ============================================================
#  Panel Reconstruction
# ============================================================

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

    sources = [(y,c) for (y,c) in forward.keys() if (y,c) not in reverse]

    for s in sorted(sources):
        follow(s)

    all_nodes = set(list(forward.keys()) + list(forward.values()))
    for node in sorted(all_nodes):
        if node in visited: continue
        cur = node
        while cur in reverse:
            cur = reverse[cur]
        follow(cur)

    panel = pd.DataFrame(rows)
    if panel.empty: return panel
    panel["docs"] = panel.apply(lambda r: docs_map.get((r["year"], r["cluster_id"]), np.nan), axis=1)
    return panel


def _panel_from_artifacts_edges(flow_edges_df: pd.DataFrame) -> pd.DataFrame:
    return _build_panel_from_edges(flow_edges_df)



# ============================================================
# TF-IDF / Titles Helpers
# ============================================================

TITLE_COLS_CAND = ["title","paper_title","doc_title"]

def _first_existing_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None

def _parse_embedding(x):
    if isinstance(x,(list,tuple,np.ndarray)):
        return np.asarray(x,dtype=np.float32)
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
    sims = (E @ centroid)/(np.linalg.norm(E,axis=1)*(np.linalg.norm(centroid)+1e-9)+1e-9)
    df2 = df_c.iloc[:len(sims)].copy()
    df2["__sim__"] = sims
    return df2.sort_values("__sim__",ascending=False)[title_col].head(n).astype(str).tolist()


# ============================================================
#  NEW ADDITION — CLUSTER NAMING
# ============================================================

def generate_cluster_names(artifacts: Dict[str, Any], topk:int=5) -> pd.DataFrame:
    rows = []
    tfidf_by_year = artifacts.get("tfidf_by_year", {})

    for year, tfdf in tfidf_by_year.items():
        if tfdf.empty:
            continue

        for cid, g in tfdf.groupby("cluster_id"):
            terms = (
                g.sort_values("tfidf", ascending=False)
                 .head(topk)["term"].astype(str).tolist()
            )
            if not terms:
                cname = f"Cluster-{year}-{cid}"
            else:
                cname = " ".join([t.capitalize() for t in terms])

            rows.append({
                "year": int(year),
                "cluster_id": int(cid),
                "cluster_name": cname
            })

    return pd.DataFrame(rows)


def attach_flow_names_to_cluster_names(cluster_df, flow_df, artifacts):
    fedges = artifacts.get("flow_edges_df")
    if fedges is None or fedges.empty:
        return cluster_df

    panel = _panel_from_artifacts_edges(fedges)
    flow_names = flow_df.set_index("flow_id")

    merged = cluster_df.merge(panel[["year","cluster_id","flow_id"]],
                              on=["year","cluster_id"], how="left")

    merged["flow_name_en"] = merged["flow_id"].map(
        lambda x: flow_names.loc[x]["tech_name_en"] if x in flow_names.index else None
    )
    merged["flow_name_ko"] = merged["flow_id"].map(
        lambda x: flow_names.loc[x]["tech_name_ko"] if x in flow_names.index else None
    )

    return merged



# ============================================================
#  Flow-Aggregated Naming (기존 FLOW 네이밍 유지)
# ============================================================

TOPK_KEYWORDS = 40
N_REP_TITLES  = 3

SYS_FLOWAG = (
    "당신은 기술 네이밍 비서입니다. 입력된 클러스터의 키워드와 대표 타이틀을 보고 "
    "① 기술의 '목적', ② '구현 방법', ③ '신규 기여'를 간결히 도출한 다음, "
    "이를 근거로 한국어/영문 기술명을 제안하세요.\n\n"
    "출력은 반드시 JSON만 반환:\n"
    "{"
    "\"tech_name_ko\":\"(18자 이내)\","
    "\"tech_name_en\":\"(3~5 words)\","
    "\"purpose\":\"~을 위한 ~\","
    "\"method\":\"핵심 접근/알고리즘\","
    "\"novelty\":\"차별점\","
    "\"rationale\":\"1문장 이유\""
    "}"
)

def _run_flowagg(panel: pd.DataFrame, target_flows: List[int], artifacts: Dict[str, Any]) -> pd.DataFrame:

    rows_out: List[Dict[str,Any]] = []

    def _task(fid: int):
        sub = panel[panel["flow_id"]==fid].sort_values("year")
        term_score = {}

        for y, g in sub.groupby("year"):
            df = artifacts["tfidf_by_year"].get(int(y), pd.DataFrame())
            if df.empty or not {"cluster_id","term","tfidf"} <= set(df.columns):
                continue
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
                "tech_name_ko": out.get("tech_name_ko"),
                "tech_name_en": out.get("tech_name_en"),
                "purpose": out.get("purpose"),
                "method": out.get("method"),
                "novelty": out.get("novelty"),
                "rationale": out.get("rationale"),
                "raw_text": out.get("_raw_text"),
                "model": OPENAI_MODEL,
                "status": "ok"
            }
        except Exception as e:
            return {"flow_id": fid, "status": f"error: {e}"}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_task, int(fid)) for fid in target_flows]
        for res in as_completed(futures):
            rows_out.append(res.result())

    return pd.DataFrame(rows_out)


# ============================================================
#  Main Entrypoint: run_tech_naming
# ============================================================

def run_tech_naming(_ignored=None, *, artifacts=None, top_n=10):
    if artifacts is None:
        raise RuntimeError("artifacts 가 필요합니다.")

    # ---------------------------
    # 1) PANEL 구성
    # ---------------------------
    fedges = artifacts.get("flow_edges_df", pd.DataFrame())

    if fedges is None or fedges.empty:
        years = artifacts.get("years", [])
        if len(years)==1:
            y = int(years[0])
            df_year = artifacts["clustered_by_year"][y]
            label_col = artifacts["label_col"]
            panel = df_year[["year", label_col]].drop_duplicates().rename(columns={label_col:"cluster_id"})
        else:
            raise RuntimeError("flow_edges_df 비었고 연도도 1개보다 많음")
    else:
        panel = _panel_from_artifacts_edges(fedges)

    panel["year"] = panel["year"].astype(int)

    # ---------------------------
    # 2) 상위 Flow 선정
    # ---------------------------
    def _select_top_flows(panel_df, n):
        if "docs" not in panel_df.columns:
            panel_df["docs"] = 1
        agg = (panel_df.groupby("flow_id", as_index=False)["docs"].sum()
                          .sort_values("docs", ascending=False)
                          .head(n))
        return agg["flow_id"].astype(int).tolist()

    top_flows = _select_top_flows(panel, top_n)

    # ---------------------------
    # 3) FLOW 이름 생성 (GPT)
    # ---------------------------
    flow_df = _run_flowagg(panel, top_flows, artifacts)
    flow_csv = flow_df.to_csv(index=False, encoding="utf-8-sig")

    # ---------------------------
    # 4) CLUSTER 이름 자동 생성 (비용 ZERO)
    # ---------------------------
    cluster_df = generate_cluster_names(artifacts, topk=5)

    # ---------------------------
    # 5) CLUSTER에 FLOW 이름 연동
    # ---------------------------
    cluster_final = attach_flow_names_to_cluster_names(cluster_df, flow_df, artifacts)
    cluster_csv = cluster_final.to_csv(index=False, encoding="utf-8-sig")

    # ---------------------------
    # 6) RESULT 반환
    # ---------------------------
    return {
        "flow_names": flow_df,
        "cluster_names": cluster_final,
        "flow_csv": flow_csv,
        "cluster_csv": cluster_csv,
        "meta": {
            "top_n": top_n,
            "model": OPENAI_MODEL,
            "cluster_topk": 5
        }
    }
