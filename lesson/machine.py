from __future__ import annotations

from lesson.schema import Step, Verdict

MAX_TRIES = 3


class StepMachine:
    def __init__(self, steps: list[Step], max_tries: int = MAX_TRIES, state: dict | None = None):
        self.steps = steps
        self.max_tries = max_tries
        # status: active | complete
        self.state = state or {"current": 0, "tries": 0, "status": "active", "history": []}

    @classmethod
    def from_dict(cls, steps: list[Step], state: dict, max_tries: int = MAX_TRIES) -> "StepMachine":
        return cls(steps, max_tries, state)

    def to_dict(self) -> dict:
        return self.state

    @property
    def status(self) -> str:
        return self.state["status"]

    @property
    def current(self) -> Step | None:
        return None if self.status == "complete" else self.steps[self.state["current"]]

    def submit(self, verdict: Verdict, strike: bool = True) -> str:
        """Apply a verdict. Returns: advanced | complete | retry | struggling.
        `strike`: an ADJUST counts toward max_tries (False for a nudge the painter is already
        making progress on). "struggling" means max_tries strikes on this step; it stays active."""
        if self.status != "active":
            raise RuntimeError(f"cannot submit while status is {self.status!r}")
        step = self.current
        self.state["history"].append(
            {"step": step.index, "verdict": verdict.verdict, "category": verdict.category,
             "adjustment": verdict.adjustment}
        )
        if verdict.verdict == "READY":
            return self._advance("advanced")
        self.state["tries"] += int(strike)
        return "struggling" if self.state["tries"] >= self.max_tries else "retry"

    def skip(self) -> str:
        """Painter (or demo operator) moves on from the current step without it being done."""
        if self.status != "active":
            raise RuntimeError(f"cannot skip while status is {self.status!r}")
        self.state["history"].append({"step": self.current.index, "skipped": True})
        return self._advance("advanced")

    def _advance(self, label: str) -> str:
        self.state["tries"] = 0
        if self.state["current"] + 1 >= len(self.steps):
            self.state["status"] = "complete"
            return "complete"
        self.state["current"] += 1
        return label
