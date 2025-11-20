# app/core/pipeline_steps/eta_format.py

def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "예상 소요 시간: 계산 중"

    if seconds < 1:
        return "예상 소요 시간: 거의 완료"

    if seconds < 60:
        return f"예상 소요 시간: {int(seconds)}초"

    minutes = seconds / 60
    return f"예상 소요 시간: {minutes:.1f}분"
