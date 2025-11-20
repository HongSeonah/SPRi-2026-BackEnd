# app/core/pipeline_steps/progress_tracker.py

class ProgressTracker:
    """
    스텝별 가중치를 기반으로 전체 진행률 계산
    기존 pipeline.py에 있던 로직을 분리한 버전
    """

    ELEMENT_WEIGHT = 0.85
    COMPONENT_WEIGHT = 0.15

    ELEMENT_STEPS = [
        ("upload",       0.05),
        ("load",         0.05),
        ("year_filter",  0.10),
        ("preprocess",   0.20),
        ("embedding",    0.35),
        ("clustering",   0.15),
        ("tech_naming",  0.10),
    ]

    COMPONENT_STEPS = [
        ("component_grouping", 0.30),
        ("component_naming",   0.30),
        ("component_result",   0.10),
        ("visualize",          0.30),
    ]

    def __init__(self):
        self._el_progress = {name: 0.0 for name, _ in self.ELEMENT_STEPS}
        self._co_progress = {name: 0.0 for name, _ in self.COMPONENT_STEPS}

        self._el_weights = dict(self.ELEMENT_STEPS)
        self._co_weights = dict(self.COMPONENT_STEPS)

    def update(self, step_name: str, frac: float):
        frac = max(0.0, min(1.0, frac))
        if step_name in self._el_progress:
            self._el_progress[step_name] = frac
        elif step_name in self._co_progress:
            self._co_progress[step_name] = frac

    def _weighted_sum(self, progress_map, weight_map):
        return sum(progress_map[s] * weight_map[s] for s in progress_map)

    def element_progress(self) -> float:
        return self._weighted_sum(self._el_progress, self._el_weights)

    def component_progress(self) -> float:
        return self._weighted_sum(self._co_progress, self._co_weights)

    def overall_progress(self) -> float:
        return (
            self.ELEMENT_WEIGHT * self.element_progress()
            + self.COMPONENT_WEIGHT * self.component_progress()
        )
