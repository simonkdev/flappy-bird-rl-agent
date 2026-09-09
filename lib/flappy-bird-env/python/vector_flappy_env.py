from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from flappy_env import StepResult


class VectorFlappyEnv:
    def __init__(
        self,
        executable: str | Path = "lib/flappy-bird-env/build/flappy_env_vector_server",
        *,
        num_envs: int = 4,
        width: int = 42,
        height: int = 42,
        ticks_per_step: int = 4,
        dt: float = 1.0 / 60.0,
    ) -> None:
        self.executable = Path(executable)
        args = [
            str(self.executable),
            "--envs",
            str(num_envs),
            "--width",
            str(width),
            "--height",
            str(height),
            "--ticks",
            str(ticks_per_step),
            "--dt",
            str(dt),
        ]
        self._process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        ready = self._readline()
        parts = ready.split()
        if len(parts) != 6 or parts[0] != "READY":
            raise RuntimeError(f"Unexpected vector server startup response: {ready!r}")

        self.num_envs = int(parts[1])
        self.width = int(parts[2])
        self.height = int(parts[3])
        self.ticks_per_step = int(parts[4])
        self.dt = float(parts[5])

    def reset_all(self, seeds: Optional[list[int]] = None) -> list[StepResult]:
        if seeds is None:
            command = "RESET_ALL"
        else:
            command = "RESET_ALL " + " ".join(str(seed) for seed in seeds)
        return self._request(command, "RESET_ALL")

    def reset_one(self, env_index: int, seed: Optional[int] = None) -> StepResult:
        if env_index < 0 or env_index >= self.num_envs:
            raise IndexError(f"env_index {env_index} is out of range for {self.num_envs} envs")
        command = f"RESET_ONE {env_index}"
        if seed is not None:
            command += f" {seed}"
        return self._request(command, "RESET_ONE")[0]

    def step_batch(self, actions: list[int]) -> list[StepResult]:
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}")
        command = "STEP_BATCH " + " ".join(str(action) for action in actions)
        return self._request(command, "STEP_BATCH")

    def close(self) -> None:
        process = getattr(self, "_process", None)
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

    def __enter__(self) -> "VectorFlappyEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _request(self, command: str, expected_kind: str) -> list[StepResult]:
        if self._process.poll() is not None:
            stderr = self._read_stderr()
            raise RuntimeError(f"flappy_env_vector_server exited with code {self._process.returncode}: {stderr}")
        if not self._process.stdin:
            raise RuntimeError("flappy_env_vector_server stdin is unavailable")

        self._process.stdin.write((command + "\n").encode("ascii"))
        self._process.stdin.flush()
        line = self._readline()
        if line.startswith("ERR "):
            raise ValueError(line[4:])
        return self._parse_batch(line, expected_kind)

    def _readline(self) -> str:
        if not self._process.stdout:
            raise RuntimeError("flappy_env_vector_server stdout is unavailable")

        line = self._process.stdout.readline()
        if not line:
            stderr = self._read_stderr()
            raise RuntimeError(f"flappy_env_vector_server closed stdout: {stderr}")
        return line.decode("ascii").strip()

    def _read_stderr(self) -> str:
        if not self._process.stderr:
            return ""
        return self._process.stderr.read().decode("utf-8", errors="replace")

    def _parse_batch(self, line: str, expected_kind: str) -> list[StepResult]:
        parts = line.split()
        if len(parts) < 3 or parts[0] != "OK" or parts[1] != expected_kind:
            raise RuntimeError(f"Unexpected vector server response: {line!r}")

        count = int(parts[2])
        expected_header_values = 3 + (count * 15)
        if len(parts) != expected_header_values:
            raise RuntimeError(f"Unexpected vector server header length: {line!r}")

        headers = []
        payload_size = 0
        cursor = 3
        for _ in range(count):
            width = int(parts[cursor])
            height = int(parts[cursor + 1])
            dtype = parts[cursor + 2]
            reward = float(parts[cursor + 3])
            terminated = parts[cursor + 4] == "1"
            alive = parts[cursor + 5] == "1"
            score = int(parts[cursor + 6])
            passed_pipe = parts[cursor + 7] == "1"
            simulation_time = float(parts[cursor + 8])
            has_next_pipe = parts[cursor + 9] == "1"
            bird_y = float(parts[cursor + 10])
            next_pipe_x = float(parts[cursor + 11])
            next_gap_center_y = float(parts[cursor + 12])
            next_gap_half_height = float(parts[cursor + 13])
            size = int(parts[cursor + 14])
            headers.append((width, height, dtype, reward, terminated, alive, score, passed_pipe, simulation_time, has_next_pipe, bird_y, next_pipe_x, next_gap_center_y, next_gap_half_height, size))
            payload_size += size
            cursor += 15

        payload = self._process.stdout.read(payload_size)
        if len(payload) != payload_size:
            raise RuntimeError(f"Observation payload has {len(payload)} bytes, expected {payload_size}")

        results = []
        offset = 0
        for width, height, dtype, reward, terminated, alive, score, passed_pipe, simulation_time, has_next_pipe, bird_y, next_pipe_x, next_gap_center_y, next_gap_half_height, size in headers:
            observation = payload[offset:offset + size]
            offset += size
            if len(observation) != width * height:
                raise RuntimeError(f"Observation has {len(observation)} bytes, expected {width * height}")
            results.append(
                StepResult(
                    observation=observation,
                    width=width,
                    height=height,
                    dtype=dtype,
                    reward=reward,
                    terminated=terminated,
                    alive=alive,
                    score=score,
                    passed_pipe=passed_pipe,
                    simulation_time=simulation_time,
                    has_next_pipe=has_next_pipe,
                    bird_y=bird_y,
                    next_pipe_x=next_pipe_x,
                    next_gap_center_y=next_gap_center_y,
                    next_gap_half_height=next_gap_half_height,
                )
            )

        return results
