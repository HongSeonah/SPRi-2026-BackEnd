from fastapi import APIRouter
import os, re, json, time
from typing import Dict
from openai import OpenAI

router = APIRouter()

# ====== 설정 ======
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("환경변수 OPENAI_API_KEY 가 설정되어 있지 않습니다.")

# ====== OpenAI 클라이언트 초기화 ======
try:
    client = OpenAI(api_key=API_KEY)
    _MODE = "v1"
except Exception:
    import openai
    openai.api_key = API_KEY
    client = openai
    _MODE = "legacy"

# ====== 시스템 프롬프트 (원문 유지) ======
SYS_PROMPT = (
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

# ====== 내부 함수 ======
def _extract_json(txt: str) -> str:
    m = re.search(r"\{.*\}", txt, flags=re.S)
    return m.group(0) if m else txt

def call_gpt(user_prompt: str, retry=3, sleep=1.5) -> Dict:
    last_err = None
    for i in range(retry):
        try:
            if _MODE == "v1":
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYS_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.4,
                    response_format={"type": "json_object"},
                )
                txt = resp.choices[0].message.content or "{}"
            else:
                resp = client.ChatCompletion.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYS_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.4
                )
                txt = resp["choices"][0]["message"]["content"]
            return json.loads(_extract_json(txt))
        except Exception as e:
            last_err = e
            time.sleep(sleep * (i + 1))
    raise last_err

# ====== 외부 호출용 함수 ======
def run_tech_naming(user_prompt: str) -> Dict:
    """
    기존 기술명명 코드와 동일하게 작동하는 래퍼.
    user_prompt는 기존 build_user_prompt() 결과 문자열을 그대로 전달해야 함.
    """
    out = call_gpt(user_prompt)
    return {
        "tech_name_ko": out.get("tech_name_ko"),
        "tech_name_en": out.get("tech_name_en"),
        "purpose": out.get("purpose"),
        "method": out.get("method"),
        "novelty": out.get("novelty"),
        "rationale": out.get("rationale"),
        "model": MODEL,
        "status": "ok"
    }
