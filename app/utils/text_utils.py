import re

def clean_text(text: str, keep_signs: bool = True) -> str:
    """
    문자열을 전처리하는 간단한 함수
    - 알파벳/숫자만 남기고 소문자로 변환
    - keep_signs=True면 + # - / . 등의 특수문자는 유지
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    if keep_signs:
        text = re.sub(r"[^a-z0-9#\+\/\-\.\s]", " ", text)
    else:
        text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
