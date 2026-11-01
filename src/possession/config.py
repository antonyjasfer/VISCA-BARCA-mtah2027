"""The model's switches, in one place. Defaults are what the submission uses; METHOD.md records how
they were chosen. Numeric constants stay beside the code they belong to.
"""

from dataclasses import dataclass

RESTART_RULES = ("v3", "first_control")


@dataclass(frozen=True)
class ModelConfig:
    # True: relabel a live run under 1.5 s between two runs of the other team (control.merge_touches)
    touch_merge: bool = False
    # "v3": tag team, then ball-out, nearest team, next tagged phase (dead_ball.resolve_restart_teams)
    # "first_control": team of the first control event within 4 s of the restart, then that chain
    restart_rule: str = "first_control"

    def __post_init__(self):
        if self.restart_rule not in RESTART_RULES:
            raise ValueError(f"restart_rule must be one of {RESTART_RULES}")
