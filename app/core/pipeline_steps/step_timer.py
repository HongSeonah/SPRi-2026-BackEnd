# app/core/pipeline_steps/step_timer.py

import time

class StepTimer:
    """
    각 Step의 진행률 대비 ETA 계산 담당
    """

    def __init__(self):
        self.start_ts = None

    def start(self):
        self.start_ts = time.time()

    def eta_seconds(self, processed: int, total: int) -> float | None:
        """ETA를 초 단위로 반환"""
        if processed <= 0 or total <= 0 or self.start_ts is None:
            return None

        elapsed = time.time() - self.start_ts
        if elapsed <= 0:
            return None

        speed = processed / elapsed  # items per sec
        if speed <= 0:
            return None

        remain = total - processed
        if remain <= 0:
            return 0

        return remain / speed
