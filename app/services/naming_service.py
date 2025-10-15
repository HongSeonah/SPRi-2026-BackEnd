import os, re, io, json, time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from app.utils.io_utils import tmp_path  # temp 경로 유틸 재사용

# ===== OpenAI 클라이언트 (v1/legacy 모두 지원) =====
def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")
    try:
        from openai import OpenAI
        return "v1", OpenAI(api_key=api_key)
    except Exception:
        import openai
        openai.api_key = api_key
        return "legacy", openai

# ===== 공통 경로/파일 찾기 =====
def _compare_root(label_suffix: str) -> Path:
    return Path(f"./cluster_out/compare_methods_{label_suffix}")

def _yearly_dirs() -> List[Path]:
    return [Path("./cluster_out/yearly_v4"), Path("./cluster_out/yearly_v3"), Path("./cluster_out/yearly_v2")]

def _find_year_file(year: int, stem_prefix: str, label_suffix: str) -> Optional[Path]:
    # 우선 고정 접미 우선: stem_prefix_{label_suffix}.csv
    for base in _yearly_dirs():
        f = base/str(year)/f"{stem_prefix}_{label_suffix}.csv"
        if f.exists(): return f
    # 그 다음 가변 k 지원
    for base in _yearly_dirs():
        hits = sorted((base/str(year)).glob(f"{stem_prefix}_k*.csv"))
        if hits: return hits[0]
    return None

def _find_counts(year: int, label_suffix: str) -> Optional[Path]:
    return _find_year_file(year, "cluster_counts", label_suffix)

def _find_tfidf(year: int, label_suffix: str) -> Optional[Path]:
    return _find_year_file(year, "tfidf_top_terms", label_suffix)

def _find_clustered(year: int) -> Optional[Tuple[Path, str]]:
    # *_clustered_k*.csv 중 라벨 컬럼(cluster_k###)이 있는 파일 발견
    for base in _yearly_dirs():
        ydir = base / str(year)
        if not ydir.exists():
            continue
        for f in sorted(ydir.glob("*clustered_k*.csv")):
            try:
                df = pd.read_csv(f, nrows=2)
                lab = next((c for c in df.columns if re.fullmatch(r"cluster_k\d+", c)), None)
                if lab:
                    return f, lab
            except Exception:
                pass
    return None

# ===== 텍스트/임베딩 유틸 =====
TITLE_COLS_CAND = ["title", "paper_title", "doc_title"]
ABSTR_COLS_CAND = ["abstract", "summary", "doc_abstract"]

def _first_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    return next((c for c in cands if c in df.columns), None)

def _parse_embedding(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, str):
        s = x.strip()
        try:
            return np.asarray(json.loads(s), dtype=np.float32)
        except Exception:
            try:
                import ast
                return np.asarray(ast.literal_eval(s), dtype=np.float32)
            except Exception:
                return None
    return None

def _rep_titles_via_embeddings(df_c: pd.DataFrame, n: int) -> List[str]:
    emb_col = next((c for c in df_c.columns if c.lower()=="embedding"), None)
    ti_col  = _first_col(df_c, TITLE_COLS_CAND)
    if emb_col is None or ti_col is None: return []
    embs = []
    for _, r in df_c.iterrows():
        v = _parse_embedding(r[emb_col])
        if v is None: continue
        embs.append(v)
    if not embs: return []
    E = np.vstack(embs).astype(np.float32)
    c = E.mean(axis=0)
    sims = (E @ c) / (np.linalg.norm(E, axis=1)*np.linalg.norm(c)+1e-9)
    df2 = df_c.iloc[:len(sims)].copy(); df2["__sim__"] = sims
    return df2.sort_values("__sim__", ascending=False).head(n)[ti_col].astype(str).tolist()

def _rep_titles_via_tfidf(df_c: pd.DataFrame, keywords: List[str], n: int) -> List[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    ti_col = _first_col(df_c, TITLE_COLS_CAND)
    ab_col = _first_col(df_c, ABSTR_COLS_CAND)
    if ti_col is None and ab_col is None: return []
    texts = (df_c[ti_col].fillna("") + " " + (df_c[ab_col].fillna("") if ab_col else "")).astype(str)
    if texts.empty: return []
    vec = TfidfVectorizer(max_features=120000, ngram_range=(1,2), min_df=2, max_df=0.95)
    X = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()
    if not keywords:
        scores = np.asarray(X.mean(axis=1)).ravel()
    else:
        kw = set(k.lower() for k in keywords)
        w = np.zeros(len(vocab), dtype=np.float32)
        for i, t in enumerate(vocab):
            if t in kw: w[i] = 1.0
        scores = (X @ w).A.ravel()
    order = np.argsort(-scores)[:n]
    return df_c.iloc[order][ti_col].astype(str).tolist()

# ===== GPT 호출 =====
SYS_HYBRID = (
    "당신은 기술 네이밍 비서입니다. 입력된 클러스터의 키워드와 대표 타이틀을 보고 "
    "① 기술의 '목적', ② '구현 방법', ③ '신규 기여'를 간결히 도출한 다음, "
    "이를 근거로 한국어/영문 기술명을 제안하세요.\n\n"
    "출력은 반드시 JSON만 반환:\n"
    "{"
    "\"tech_name_ko\":\"(18자 이내)\","
    "\"tech_name_en\":\"(3~5 words)\","
    "\"purpose\":\"~을/를 위한 ~\","
    "\"method\":\"핵심 접근\","
    "\"novelty\":\"차별점(1~2문장)\","
    "\"rationale\":\"선정 이유(한국어 1문장)\""
    "}"
)
SYS_FLOWAGG = (
    "역할: 기술 네이밍 컨설턴트.\n"
    "입력된 '흐름 전체(여러 연도·클러스터)에서 집계한 상위 키워드'를 보고 "
    "① 목적, ② 구현방법, ③ 신규기여 를 간결히 요약하고, "
    "세부 도메인(질병/제품/업체/데이터셋/수치 등)은 피하며 상위 개념으로 KO/EN 기술명을 제안.\n"
    "JSON만 출력:\n"
    "{"
    "\"tech_name_ko\":\"(12~18자)\","
    "\"tech_name_en\":\"(3~5 words)\","
    "\"purpose\":\"~을 위한 ~\","
    "\"method\":\"핵심 접근\","
    "\"novelty\":\"차별점(1~2문장)\","
    "\"rationale\":\"한국어 1문장\""
    "}"
)

def _extract_json(txt: str) -> str:
    m = re.search(r"\{.*\}", txt, flags=re.S)
    return m.group(0) if m else "{}"

def _call_gpt(system_prompt: str, user_prompt: str, model: str) -> Dict:
    mode, cli = _get_openai_client()
    if mode == "v1":
        resp = cli.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}],
            temperature=0.35,
            response_format={"type":"json_object"},
        )
        txt = resp.choices[0].message.content or "{}"
    else:
        resp = cli.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}],
            temperature=0.35
        )
        txt = resp["choices"][0]["message"]["content"]
    return json.loads(_extract_json(txt))

# ===== 대상(flow_id) 선택: WEAK TOP20 + UPGRADE TOP20 =====
def _load_targets(root: Path, year_pred: int) -> pd.DataFrame:
    weak_f = root / "grading_v3/model_weak_upgrade/pred_WEAK_2024.csv"
    upg_f  = root / "grading_v3/model_weak_upgrade/pred_upgrade_from_WEAK_2024.csv"
    cols = ["flow_id","cluster_name"]
    weak = pd.read_csv(weak_f)[cols+["p_WEAK_y"]].sort_values("p_WEAK_y", ascending=False).head(20) if weak_f.exists() else pd.DataFrame(columns=cols)
    upg  = pd.read_csv(upg_f)[cols+["p_upgrade_2y"]].sort_values("p_upgrade_2y", ascending=False).head(20) if upg_f.exists() else pd.DataFrame(columns=cols)
    weak["list_type"] = "WEAK_TOP20"; upg["list_type"] = "UPGRADE_TOP20"
    tgt = pd.concat([weak[cols+["list_type"]], upg[cols+["list_type"]]]).drop_duplicates(subset=["flow_id","list_type"])
    return tgt

# ===== 최신 (연도, 클러스터) 추출 =====
def _latest_info(panel: pd.DataFrame, flow_id: str) -> Optional[Dict]:
    rows = panel[panel["flow_id"]==flow_id]
    if rows.empty: return None
    r = rows.sort_values("year").iloc[-1]
    year = int(r["year"])
    cid_col = next((c for c in rows.columns if re.fullmatch(r"cluster_id|cluster_k\d+", c)), None)
    cid = int(r[cid_col]) if cid_col and not pd.isna(r[cid_col]) else None
    name = r["cluster_name"] if "cluster_name" in rows.columns else None
    return dict(year=year, cluster_id=cid, cluster_name=name)

# ===== HYBRID 모드 =====
def _build_prompt_hybrid(flow_id: str, cname: Optional[str], year: int, keywords: List[str], rep_titles: List[str],
                         topk_keywords: int, n_rep_titles: int) -> str:
    kw = ", ".join(keywords[:topk_keywords]) if keywords else ""
    tl = "; ".join([str(t)[:90] + ("…" if len(str(t))>90 else "") for t in rep_titles[:n_rep_titles]]) if rep_titles else ""
    base = cname or flow_id
    return (f"[기술 흐름] {flow_id}\n[최신 연도] {year}\n[클러스터명] {base}\n"
            f"[핵심 키워드] {kw}\n[대표 타이틀] {tl}\n\n"
            "위 정보를 바탕으로 JSON만 출력하세요.")

def _load_top_terms(year: int, cluster_id: int, label_suffix: str, topk: int) -> List[str]:
    f = _find_tfidf(year, label_suffix)
    if not f: return []
    df = pd.read_csv(f)
    if not {"cluster_id","term","tfidf"}.issubset(df.columns): return []
    sub = df[df["cluster_id"]==int(cluster_id)].sort_values("tfidf", ascending=False).head(topk)
    return sub["term"].astype(str).tolist()

def _load_cluster_docs(year: int, cluster_id: int) -> pd.DataFrame:
    hit = _find_clustered(year)
    if not hit: return pd.DataFrame()
    f, lab = hit
    d = pd.read_csv(f)
    if lab not in d.columns: return pd.DataFrame()
    return d[d[lab]==int(cluster_id)].copy()

# ===== FLOWAGG 모드 =====
def _aggregate_flow_keywords(panel: pd.DataFrame, label_suffix: str, flow_id: str,
                             agg_window: Optional[int], topk_per_cluster: int,
                             doc_weight_norm_by_year: bool, topk_flow_keywords: int) -> Tuple[List[Tuple[str,float]], int, int]:
    rows = panel[panel["flow_id"]==flow_id].copy()
    if rows.empty: return [], 0, 0
    rows["year"] = rows["year"].astype(int)
    rows = rows.sort_values("year")

    if isinstance(agg_window, int) and agg_window>0:
        y_max = rows["year"].max()
        rows = rows[rows["year"] >= (y_max - agg_window + 1)]

    cid_col = next((c for c in rows.columns if re.fullmatch(r"cluster_id|cluster_k\d+", c)), None)
    if cid_col is None: return [], 0, 0

    term_score: Dict[str, float] = {}
    year_used, pairs_used = set(), 0

    for y, g in rows.groupby("year"):
        f_tf = _find_tfidf(int(y), label_suffix)
        if not f_tf: continue
        df_tf = pd.read_csv(f_tf)
        if not {"cluster_id","term","tfidf"}.issubset(df_tf.columns): continue

        # counts(문서수) 로드
        f_ct = _find_counts(int(y), label_suffix)
        doc_map = {}
        if f_ct and f_ct.exists():
            df_ct = pd.read_csv(f_ct)
            if {"cluster_id","count"}.issubset(df_ct.columns):
                doc_map = {int(r["cluster_id"]): int(r["count"]) for _, r in df_ct.iterrows()}
        maxd = max(doc_map.values()) if (doc_map and doc_weight_norm_by_year) else 1

        for _, r in g.iterrows():
            cid = r[cid_col]
            if pd.isna(cid): continue
            cid = int(cid)
            sub = df_tf[df_tf["cluster_id"]==cid].sort_values("tfidf", ascending=False).head(topk_per_cluster)
            if sub.empty: continue
            w_docs = (doc_map.get(cid, 1) / maxd) if (doc_map and doc_weight_norm_by_year) else (doc_map.get(cid, 1) or 1)
            for _, s in sub.iterrows():
                t = str(s["term"]).lower()
                sc = float(s["tfidf"]) * float(w_docs)
                term_score[t] = term_score.get(t, 0.0) + sc
            year_used.add(int(y)); pairs_used += 1

    if not term_score:
        return [], len(year_used), pairs_used
    items = sorted(term_score.items(), key=lambda x: x[1], reverse=True)
    return items[:topk_flow_keywords], len(year_used), pairs_used

def _build_prompt_flowagg(flow_id: str, terms: List[Tuple[str,float]], meta: str) -> str:
    kw = ", ".join([t for t,_ in terms])
    return (f"[기술 흐름] {flow_id}\n[메타] {meta}\n[흐름-집계 키워드] {kw}\n\n"
            "위 정보를 바탕으로 JSON만 출력하세요.")

# ===== 파이프라인 =====
async def run_naming_pipeline(
    mode: str,
    label_suffix: str,
    method: str,
    year_pred: int,
    topk_keywords: int,
    n_rep_titles: int,
    use_embeddings: bool,
    agg_window: Optional[int],
    topk_per_cluster: int,
    topk_flow_keywords: int,
    doc_weight_norm_by_year: bool,
    model: str,
) -> Dict:
    root = _compare_root(label_suffix)
    panel = pd.read_csv(root / f"flow_year_overall_{method}_{label_suffix}.csv")
    if "year" in panel.columns:
        panel["year"] = panel["year"].astype(int)

    targets = _load_targets(root, year_pred)
    if targets.empty:
        raise RuntimeError("대상 리스트가 비어 있습니다. (pred_WEAK_2024.csv / pred_upgrade_from_WEAK_2024.csv 확인)")

    out_dir = root / "grading_v3" / "model_weak_upgrade"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 결과 테이블들
    hy_rows, ag_rows = [], []

    # ---------- HYBRID ----------
    if mode in ("hybrid", "both"):
        for _, r in targets.iterrows():
            fid = r["flow_id"]; cname = r.get("cluster_name")
            info = _latest_info(panel, fid)
            if info is None or info["cluster_id"] is None:
                hy_rows.append({"flow_id": fid, "list_type": r["list_type"], "status": "no_latest"})
                continue
            year = int(info["year"]); cid = int(info["cluster_id"])
            # 키워드
            kws = _load_top_terms(year, cid, label_suffix, topk=topk_keywords)
            # 대표 타이틀
            dfc = _load_cluster_docs(year, cid)
            reps = []
            if not dfc.empty:
                if use_embeddings:
                    reps = _rep_titles_via_embeddings(dfc, n=n_rep_titles)
                if not reps:
                    reps = _rep_titles_via_tfidf(dfc, kws, n=n_rep_titles)
            prompt = _build_prompt_hybrid(fid, info.get("cluster_name") or cname, year, kws, reps,
                                          topk_keywords, n_rep_titles)
            try:
                out = _call_gpt(SYS_HYBRID, prompt, model=model)
                hy_rows.append({
                    "flow_id": fid, "list_type": r["list_type"], "year": year, "cluster_id": cid,
                    "cluster_name_src": info.get("cluster_name") or cname,
                    "tech_name_ko": out.get("tech_name_ko"),
                    "tech_name_en": out.get("tech_name_en"),
                    "purpose": out.get("purpose"), "method": out.get("method"),
                    "novelty": out.get("novelty"), "rationale": out.get("rationale"),
                    "keywords_used": ", ".join(kws),
                    "rep_titles_used": "; ".join(reps),
                    "model": model, "status": "ok"
                })
            except Exception as e:
                hy_rows.append({"flow_id": fid, "list_type": r["list_type"], "status": f"error: {type(e).__name__}: {e}"})

        hybrid_csv = out_dir / "names_generated_hybrid.csv"
        pd.DataFrame(hy_rows).to_csv(hybrid_csv, index=False, encoding="utf-8-sig")
    else:
        hybrid_csv = None

    # ---------- FLOWAGG ----------
    if mode in ("flowagg", "both"):
        for _, r in targets.iterrows():
            fid = r["flow_id"]
            terms, n_years, n_pairs = _aggregate_flow_keywords(
                panel, label_suffix, fid,
                agg_window=agg_window,
                topk_per_cluster=topk_per_cluster,
                doc_weight_norm_by_year=doc_weight_norm_by_year,
                topk_flow_keywords=topk_flow_keywords
            )
            if not terms:
                ag_rows.append({"flow_id": fid, "list_type": r["list_type"], "status":"no_terms"})
                continue
            meta = f"years_used={n_years}, pairs_used={n_pairs}, top_terms={len(terms)}"
            prompt = _build_prompt_flowagg(fid, terms, meta)
            try:
                out = _call_gpt(SYS_FLOWAGG, prompt, model=model)
                ag_rows.append({
                    "flow_id": fid, "list_type": r["list_type"],
                    "tech_name_ko": out.get("tech_name_ko"),
                    "tech_name_en": out.get("tech_name_en"),
                    "purpose": out.get("purpose"), "method": out.get("method"),
                    "novelty": out.get("novelty"), "rationale": out.get("rationale"),
                    "keywords_used": ", ".join([t for t,_ in terms]),
                    "n_years_used": n_years, "n_pairs_used": n_pairs,
                    "model": model, "status": "ok"
                })
            except Exception as e:
                ag_rows.append({"flow_id": fid, "list_type": r["list_type"], "status": f"error: {type(e).__name__}: {e}"})

        flowagg_csv = out_dir / "names_generated_flowagg.csv"
        pd.DataFrame(ag_rows).to_csv(flowagg_csv, index=False, encoding="utf-8-sig")
    else:
        flowagg_csv = None

    return {
        "mode": mode,
        "label_suffix": label_suffix,
        "method": method,
        "out_hybrid_csv": str(hybrid_csv) if hybrid_csv else None,
        "out_flowagg_csv": str(flowagg_csv) if flowagg_csv else None,
        "targets": len(targets),
    }