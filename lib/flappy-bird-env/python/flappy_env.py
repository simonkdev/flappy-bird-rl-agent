from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from _native_flappy_env import NativeFlappyEnv
except ImportError:
    NativeFlappyEnv = None


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
    has_next_pipe: bool
    bird_y: float
    next_pipe_x: float
    next_gap_center_y: float
    next_gap_half_height: float

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
        self._native = None
        self._native_pending_result = None
        self._process = None
        default_executable = Path("lib/flappy-bird-env/build/flappy_env_server")
        use_native = NativeFlappyEnv is not None and self.executable == default_executable and "tensorflow" not in sys.modules
        if use_native:
            self._native = NativeFlappyEnv(
                width=width,
                height=height,
                ticks_per_step=ticks_per_step,
                dt=dt,
                seed=seed,
                debug_window=debug_window,
                show_game_window=show_game_window,
            )
            self.width = width
            self.height = height
            self.ticks_per_step = ticks_per_step
            self.dt = dt
            return

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
            bufsize=0,
        )
        ready = self._readline()
        parts = ready.split()
        if len(parts) != 5 or parts[0] != "READY":
            raise RuntimeError(f"Unexpected server startup response: {ready!r}")

        self.width = int(parts[1])
        self.height = int(parts[2])
        self.ticks_per_step = int(parts[3])
        self.dt = float(parts[4])
        self._pending_command: str | None = None

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

    def start_step(self, action: int) -> None:
        if self._native is not None:
            if self._native_pending_result is not None:
                raise RuntimeError("A native FlappyEnv step is already pending")
            self._native_pending_result = self._step_native(action)
            return
        self._start_request(f"STEP {action}")

    def finish_step(self) -> StepResult:
        if self._native is not None:
            if self._native_pending_result is None:
                raise RuntimeError("No pending native FlappyEnv step")
            result = self._native_pending_result
            self._native_pending_result = None
            return result
        return self._finish_request()

    def close(self) -> None:
        process = getattr(self, "_process", None)
        native = getattr(self, "_native", None)
        if native is not None:
            native.close()
            self._native = None

        if process is None:
            return

        if process.poll() is None and process.stdin:
            try:
                process.stdin.write(b"CLOSE\n")
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
        if self._native is not None:
            if command == "RESET":
                return self._result_from_native_tuple(self._native.reset())
            if command.startswith("RESET "):
                return self._result_from_native_tuple(self._native.reset(seed=int(command.split()[1])))
            if command.startswith("STEP "):
                return self._step_native(int(command.split()[1]))
            raise RuntimeError(f"Unsupported native command: {command!r}")

        self._start_request(command)
        return self._finish_request()

    def _step_native(self, action: int) -> StepResult:
        try:
            return self._result_from_native_tuple(self._native.step(action))
        except RuntimeError as exc:
            message = str(exc)
            if message.startswith("Invalid action."):
                raise ValueError(message) from exc
            raise

    @staticmethod
    def _result_from_native_tuple(result) -> StepResult:
        return StepResult(
            observation=result[0],
            width=result[1],
            height=result[2],
            dtype=result[3],
            reward=result[4],
            terminated=result[5],
            alive=result[6],
            score=result[7],
            passed_pipe=result[8],
            simulation_time=result[9],
            has_next_pipe=result[10],
            bird_y=result[11],
            next_pipe_x=result[12],
            next_gap_center_y=result[13],
            next_gap_half_height=result[14],
        )

    def _start_request(self, command: str) -> None:
        if self._pending_command is not None:
            raise RuntimeError(f"Cannot start {command!r}; {self._pending_command!r} is still pending")
        if self._process.poll() is not None:
            stderr = self._read_stderr()
            raise RuntimeError(f"flappy_env_server exited with code {self._process.returncode}: {stderr}")
        if not self._process.stdin:
            raise RuntimeError("flappy_env_server stdin is unavailable")

        self._process.stdin.write((command + "\n").encode("ascii"))
        self._process.stdin.flush()
        self._pending_command = command

    def _finish_request(self) -> StepResult:
        if self._pending_command is None:
            raise RuntimeError("No pending flappy_env_server request")

        command = self._pending_command
        self._pending_command = None
        line = self._readline()
        if line.startswith("ERR "):
            raise ValueError(line[4:])
        return self._parse_result(line, self._process.stdout)

    def _readline(self) -> str:
        if not self._process.stdout:
            raise RuntimeError("flappy_env_server stdout is unavailable")

        line = self._process.stdout.readline()
        if not line:
            stderr = self._read_stderr()
            raise RuntimeError(f"flappy_env_server closed stdout: {stderr}")
        return line.decode("ascii").strip()

    def _read_stderr(self) -> str:
        if not self._process.stderr:
            return ""
        return self._process.stderr.read().decode("utf-8", errors="replace")

    @staticmethod
    def _parse_result(line: str, stdout) -> StepResult:
        parts = line.split()
        if len(parts) != 17 or parts[0] != "OK" or parts[1] not in {"RESET", "STEP"}:
            raise RuntimeError(f"Unexpected server response: {line!r}")

        width = int(parts[2])
        height = int(parts[3])
        expected_size = width * height
        payload_size = int(parts[16])
        if payload_size != expected_size:
            raise RuntimeError(f"Observation header announced {payload_size} bytes, expected {expected_size}")

        observation = stdout.read(payload_size)
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
            has_next_pipe=parts[11] == "1",
            bird_y=float(parts[12]),
            next_pipe_x=float(parts[13]),
            next_gap_center_y=float(parts[14]),
            next_gap_half_height=float(parts[15]),
        )
