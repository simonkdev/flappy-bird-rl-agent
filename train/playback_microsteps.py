"""Isolated policy-step expansion for smooth checkpoint playback.

This module is deliberately not wired into the GUI or training.  It preserves
the C++ environment's ``ticks_per_step=N`` action semantics while exposing the
intermediate physics ticks needed by a renderer.
"""

from dataclasses import dataclass
from typing import Callable, Protocol


class StepEnvironment(Protocol):
    def step_result(self, action: int): ...


@dataclass(frozen=True)
class PolicyStep:
    """Aggregate result of one policy action expanded into physics ticks."""

    result: object
    reward: float
    passed_pipe: bool
    ticks_completed: int


def microstep_actions(action: int, ticks_per_policy_step: int) -> tuple[int, ...]:
    """Return the exact per-tick input sequence for one policy action."""
    if action not in (0, 1):
        raise ValueError("action must be 0 (no flap) or 1 (flap)")
    if ticks_per_policy_step < 1:
        raise ValueError("ticks_per_policy_step must be positive")
    return (action,) + (0,) * (ticks_per_policy_step - 1)


def step_policy_action(
    env: StepEnvironment,
    action: int,
    ticks_per_policy_step: int,
    on_tick: Callable[[object], None] | None = None,
) -> PolicyStep:
    """Advance one policy decision while optionally exposing every tick.

    ``FlappyEnv.step(action)`` applies a flap only on the first tick of its
    internal loop.  Splitting that loop into one-tick calls must therefore use
    ``(action, 0, ..., 0)`` rather than repeating ``action``.
    """
    total_reward = 0.0
    passed_pipe = False
    result = None
    ticks_completed = 0

    for tick_action in microstep_actions(action, ticks_per_policy_step):
        result = env.step_result(tick_action)
        ticks_completed += 1
        total_reward += result.reward
        passed_pipe = passed_pipe or result.passed_pipe
        if on_tick is not None:
            on_tick(result)
        if result.terminated:
            break

    if result is None:
        raise RuntimeError("policy step did not execute a physics tick")
    return PolicyStep(result, total_reward, passed_pipe, ticks_completed)
