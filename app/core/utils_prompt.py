import re
from typing import List, Optional

def short(s: str, L=28) -> str:
    """문자열을 L자 이내로 잘라주는 유틸"""
    s = str(s) if s is not None else ""
    return s if len(s) <= L else s[:L] + "…"

def build_user_prompt(
    flow_id: str,
    cluster_name: Optional[str],
    year: int,
    keywords: List[str],
    rep_titles: List[str],
    TOPK_KEYWORDS: int = 40,
    N_REP_TITLES: int = 5
) -> str:
    """
    기술명명 GPT에 전달할 사용자 프롬프트를 구성.
    원본 코드의 build_user_prompt() 내용을 그대로 유지.
    """
    kw = ", ".join(keywords[:TOPK_KEYWORDS]) if keywords else ""
    tl = "; ".join([short(t, 90) for t in rep_titles[:N_REP_TITLES]]) if rep_titles else ""
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
