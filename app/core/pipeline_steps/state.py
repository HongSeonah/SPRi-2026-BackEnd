# app/core/pipeline_steps/state.py

from .progress_tracker import ProgressTracker
from .step_timer import StepTimer

class PipelineState:
    """
    pipeline.py에서 사용하는 통합 상태 객체
    """

    def __init__(self):
        self.progress = ProgressTracker()
        self.timer = StepTimer()
        self.current_step = None
        self.phase = "element"

    def start_step(self, step_name: str, phase: str):
        """새 스텝 시작"""
        self.current_step = step_name
        self.phase = phase
        self.timer.start()
        self.progress.update(step_name, 0.0)

    def update_progress(self, frac: float):
        """프로그레스 갱신"""
        self.progress.update(self.current_step, frac)

    def snapshot(self, processed=None, total=None):
        """현 단계 상태+ETA 패키징"""
        eta = None
        if processed is not None and total is not None:
            eta = self.timer.eta_seconds(processed, total)

        return {
            "phase": self.phase,
            "step": self.current_step,
            "step_progress": int(self.progress._el_progress.get(self.current_step, 0.0) * 100)
                             if self.phase == "element"
                             else int(self.progress._co_progress.get(self.current_step, 0.0) * 100),
            "element_progress": int(self.progress.element_progress() * 100),
            "component_progress": int(self.progress.component_progress() * 100),
            "overall_progress": int(self.progress.overall_progress() * 100),
            "eta_seconds": eta,
        }
