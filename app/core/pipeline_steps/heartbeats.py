# app/core/pipeline_steps/heartbeats.py

from .eta_format import format_eta

def make_heartbeat(
    phase: str,
    step_name: str,
    progress: float,
    element_progress: float,
    component_progress: float,
    overall_progress: float,
    eta_seconds: float | None,
    meta: dict | None = None,
):
    return {
        "phase": phase,
        "step": step_name,
        "step_progress": int(progress * 100),
        "element_progress": int(element_progress * 100),
        "component_progress": int(component_progress * 100),
        "overall_progress": int(overall_progress * 100),
        "eta": format_eta(eta_seconds),
        "meta": meta or {},
    }
