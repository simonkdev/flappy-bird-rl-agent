from __future__ import annotations

import base64
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class StepResult:
    observation: bytes
    width: int
    height: int
    dtype: str
    reward: float
    terminated: bool
    alive: bool
    score: int
    passed_pipe: bool
    simulation_time: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 1)


class FlappyEnv:
    def __init__(
        self,
        executable: str | Path = "lib/flappy-bird-env/build/flappy_env_server",
        *,
        width: int = 42,
        height: int = 42,
        ticks_per_step: int = 4,
        dt: float = 1.0 / 60.0,
        seed: Optional[int] = None,
        debug_window: bool = False,
        show_game_window: bool = False,
    ) -> None:
        self.executable = Path(executable)
        args = [
            str(self.executable),
            "--width",
            str(width),
            "--height",
            str(height),
            "--ticks",
            str(ticks_per_step),
            "--dt",
            str(dt),
        ]
        if seed is not None:
            args.extend(["--seed", str(seed)])
        if debug_window:
            args.append("--debug-window")
        if show_game_window:
            args.append("--show-game-window")

        self._process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self._readline()
        parts = ready.split()
        if len(parts) != 5 or parts[0] != "READY":
            raise RuntimeError(f"Unexpected server startup response: {ready!r}")

        self.width = int(parts[1])
        self.height = int(parts[2])
        self.ticks_per_step = int(parts[3])
        self.dt = float(parts[4])

    def reset(self, seed: Optional[int] = None) -> bytes:
        command = "RESET" if seed is None else f"RESET {seed}"
        return self._request(command).observation

    def reset_result(self, seed: Optional[int] = None) -> StepResult:
        command = "RESET" if seed is None else f"RESET {seed}"
        return self._request(command)

    def step(self, action: int) -> tuple[bytes, float, bool, dict[str, object]]:
        result = self.step_result(action)
        info = {
            "alive": result.alive,
            "score": result.score,
            "passed_pipe": result.passed_pipe,
            "simulation_time": result.simulation_time,
            "shape": result.shape,
            "dtype": result.dtype,
        }
        return result.observation, result.reward, result.terminated, info

    def step_result(self, action: int) -> StepResult:
        return self._request(f"STEP {action}")

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return

        if process.poll() is None and process.stdin:
            try:
                process.stdin.write("CLOSE\n")
                process.stdin.flush()
                self._readline()
            except (BrokenPipeError, RuntimeError):
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def __enter__(self) -> "FlappyEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _request(self, command: str) -> StepResult:
        if self._process.poll() is not None:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise RuntimeError(f"flappy_env_server exited with code {self._process.returncode}: {stderr}")
        if not self._process.stdin:
            raise RuntimeError("flappy_env_server stdin is unavailable")

        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()
        line = self._readline()
        if line.startswith("ERR "):
            raise ValueError(line[4:])
        return self._parse_result(line)

    def _readline(self) -> str:
        if not self._process.stdout:
            raise RuntimeError("flappy_env_server stdout is unavailable")

        line = self._process.stdout.readline()
        if not line:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise RuntimeError(f"flappy_env_server closed stdout: {stderr}")
        return line.strip()

    @staticmethod
    def _parse_result(line: str) -> StepResult:
        parts = line.split(maxsplit=11)
        if len(parts) != 12 or parts[0] != "OK" or parts[1] not in {"RESET", "STEP"}:
            raise RuntimeError(f"Unexpected server response: {line!r}")

        width = int(parts[2])
        height = int(parts[3])
        observation = base64.b64decode(parts[11])
        expected_size = width * height
        if len(observation) != expected_size:
            raise RuntimeError(f"Observation has {len(observation)} bytes, expected {expected_size}")

        return StepResult(
            observation=observation,
            width=width,
            height=height,
            dtype=parts[4],
            reward=float(parts[5]),
            terminated=parts[6] == "1",
            alive=parts[7] == "1",
            score=int(parts[8]),
            passed_pipe=parts[9] == "1",
            simulation_time=float(parts[10]),
        )
