from __future__ import annotations

from typing import Optional

import numpy as np

from flappy_env import StepResult


class FastVectorFlappyEnv:
    def __init__(
        self,
        *,
        num_envs: int = 4,
        width: int = 42,
        height: int = 42,
        ticks_per_step: int = 4,
        dt: float = 1.0 / 60.0,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.width = width
        self.height = height
        self.ticks_per_step = ticks_per_step
        self.dt = dt

        self.virtual_width = 1920.0
        self.virtual_height = 1080.0
        self.bird_x = 250.0
        self.bird_w = 105.0
        self.bird_h = 70.0
        self.pipe_w = 140.0
        self.pipe_h = 800.0
        self.game_velocity_const = -400.0
        self.acceleration = 15.0
        self.pipe_spawn_rate = 1.8
        self.boost = 450.0
        self.start_boost = 100.0
        self.gravity = -2300.0

        self.max_pipes = 10
        self.rngs = [np.random.default_rng(i) for i in range(num_envs)]
        self.bird_y = np.zeros(num_envs, dtype=np.float32)
        self.bird_vy = np.zeros(num_envs, dtype=np.float32)
        self.bird_rotation = np.zeros(num_envs, dtype=np.float32)
        self.score = np.zeros(num_envs, dtype=np.int32)
        self.simulation_time = np.zeros(num_envs, dtype=np.float32)
        self.last_pipe_spawn_time = np.zeros(num_envs, dtype=np.float32)
        self.pipe_x = np.zeros((num_envs, self.max_pipes), dtype=np.float32)
        self.bottom_y = np.zeros((num_envs, self.max_pipes), dtype=np.float32)
        self.top_y = np.zeros((num_envs, self.max_pipes), dtype=np.float32)
        self.pipe_used = np.zeros((num_envs, self.max_pipes), dtype=bool)
        self.pipe_scored = np.zeros((num_envs, self.max_pipes), dtype=bool)
        self.alive = np.zeros(num_envs, dtype=bool)
        self._last_observations = [b"" for _ in range(num_envs)]

    def reset_all(self, seeds: Optional[list[int]] = None) -> list[StepResult]:
        if seeds is None:
            seeds = [None for _ in range(self.num_envs)]
        if len(seeds) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} seeds, got {len(seeds)}")
        return [self.reset_one(i, seed=seeds[i]) for i in range(self.num_envs)]

    def reset_one(self, env_index: int, seed: Optional[int] = None) -> StepResult:
        self._check_env_index(env_index)
        if seed is not None:
            self.rngs[env_index] = np.random.default_rng(seed)

        self.bird_y[env_index] = 540.0
        self.bird_vy[env_index] = self.start_boost
        self.bird_rotation[env_index] = 0.0
        self.score[env_index] = 0
        self.simulation_time[env_index] = 0.0
        self.last_pipe_spawn_time[env_index] = 0.0
        self.pipe_x[env_index].fill(2000.0)
        self.bottom_y[env_index].fill(-200.0)
        self.top_y[env_index].fill(-200.0)
        self.pipe_used[env_index].fill(False)
        self.pipe_scored[env_index].fill(False)
        self.alive[env_index] = True
        observation = self._render_env(env_index)
        self._last_observations[env_index] = observation
        return self._make_result(env_index, observation, reward=0.0, passed_pipe=False)

    def step_batch(self, actions: list[int]) -> list[StepResult]:
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}")
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        passed_pipes = np.zeros(self.num_envs, dtype=bool)
        was_alive = self.alive.copy()
        previous_scores = self.score.copy()

        for env_index, action in enumerate(actions):
            if action not in (0, 1):
                raise ValueError("Invalid action. Use 0 for no flap or 1 for flap.")
            if not self.alive[env_index]:
                continue
            for tick in range(self.ticks_per_step):
                if not self.alive[env_index]:
                    break
                self._simulate_tick(env_index, flap_action=action == 1 and tick == 0)

        results = []
        for env_index in range(self.num_envs):
            score_delta = int(self.score[env_index] - previous_scores[env_index])
            passed_pipes[env_index] = score_delta > 0
            rewards[env_index] = float(score_delta)
            if was_alive[env_index] and not self.alive[env_index]:
                rewards[env_index] -= 1.0
            observation = self._render_env(env_index)
            self._last_observations[env_index] = observation
            results.append(
                self._make_result(
                    env_index,
                    observation,
                    reward=float(rewards[env_index]),
                    passed_pipe=bool(passed_pipes[env_index]),
                )
            )
        return results

    def close(self) -> None:
        return None

    def _simulate_tick(self, env_index: int, flap_action: bool) -> None:
        self.simulation_time[env_index] += self.dt
        pipe_velocity = self.game_velocity_const - (float(self.score[env_index]) * self.acceleration)
        acceleration_factor = pipe_velocity / self.game_velocity_const

        if flap_action:
            self.bird_vy[env_index] = self.boost + (self.boost * acceleration_factor * 0.4)

        if self._collides_with_pipe(env_index):
            self.bird_vy[env_index] += self.boost / 2.0
            if self.bird_vy[env_index] > self.boost * 1.5:
                self.bird_vy[env_index] = self.boost
            self.alive[env_index] = False
            return

        self.bird_vy[env_index] += self.gravity * self.dt * acceleration_factor
        rotation = abs(float(self.bird_vy[env_index])) / self.boost * 10.0
        self.bird_rotation[env_index] = rotation if self.bird_vy[env_index] >= 0 else -rotation

        unscored = self.pipe_used[env_index] & ~self.pipe_scored[env_index] & (self.bird_x > self.pipe_x[env_index])
        if np.any(unscored):
            self.score[env_index] += int(np.count_nonzero(unscored))
            self.pipe_scored[env_index, unscored] = True

        if self.simulation_time[env_index] - self.last_pipe_spawn_time[env_index] > self.pipe_spawn_rate / acceleration_factor:
            self._spawn_pipe(env_index)

        if self.bird_y[env_index] < 0.0 or self.bird_y[env_index] > self.virtual_height:
            self.alive[env_index] = False
            return

        self.bird_y[env_index] += self.bird_vy[env_index] * self.dt
        self.pipe_x[env_index, self.pipe_used[env_index]] += pipe_velocity * self.dt
        offscreen = self.pipe_used[env_index] & (self.pipe_x[env_index] < -100.0)
        self.pipe_used[env_index, offscreen] = False
        self.pipe_scored[env_index, offscreen] = False
        self.pipe_x[env_index, offscreen] = 2000.0
        self.bottom_y[env_index, offscreen] = -200.0
        self.top_y[env_index, offscreen] = -200.0

    def _spawn_pipe(self, env_index: int) -> None:
        free_indices = np.flatnonzero(~self.pipe_used[env_index])
        if free_indices.size == 0:
            return
        pipe_index = int(free_indices[0])
        gap_size = float(self.rngs[env_index].integers(1030, 1180))
        y_offset = float(self.rngs[env_index].integers(0, int(1700 - gap_size))) - 310.0
        self.pipe_used[env_index, pipe_index] = True
        self.pipe_scored[env_index, pipe_index] = False
        self.pipe_x[env_index, pipe_index] = 2000.0
        self.bottom_y[env_index, pipe_index] = y_offset
        self.top_y[env_index, pipe_index] = y_offset + gap_size
        self.last_pipe_spawn_time[env_index] = self.simulation_time[env_index]

    def _collides_with_pipe(self, env_index: int) -> bool:
        if not np.any(self.pipe_used[env_index]):
            return False
        bird_min_x = self.bird_x - self.bird_w / 2.0
        bird_max_x = self.bird_x + self.bird_w / 2.0
        bird_min_y = float(self.bird_y[env_index]) - self.bird_h / 2.0
        bird_max_y = float(self.bird_y[env_index]) + self.bird_h / 2.0
        pipe_min_x = self.pipe_x[env_index] - self.pipe_w / 2.0
        pipe_max_x = self.pipe_x[env_index] + self.pipe_w / 2.0

        x_overlap = (bird_max_x > pipe_min_x) & (bird_min_x < pipe_max_x)
        bottom_overlap = (
            x_overlap
            & (bird_max_y > (self.bottom_y[env_index] - self.pipe_h / 2.0))
            & (bird_min_y < (self.bottom_y[env_index] + self.pipe_h / 2.0))
        )
        top_overlap = (
            x_overlap
            & (bird_max_y > (self.top_y[env_index] - self.pipe_h / 2.0))
            & (bird_min_y < (self.top_y[env_index] + self.pipe_h / 2.0))
        )
        return bool(np.any(self.pipe_used[env_index] & (bottom_overlap | top_overlap)))

    def _render_env(self, env_index: int) -> bytes:
        frame = np.zeros((self.height, self.width), dtype=np.uint8)
        for pipe_index in np.flatnonzero(self.pipe_used[env_index]):
            self._draw_rect(frame, self.pipe_x[env_index, pipe_index], self.bottom_y[env_index, pipe_index], self.pipe_w, self.pipe_h, 255)
            self._draw_rect(frame, self.pipe_x[env_index, pipe_index], self.top_y[env_index, pipe_index], self.pipe_w, self.pipe_h, 255)
        self._draw_rect(frame, self.bird_x, float(self.bird_y[env_index]), self.bird_w, self.bird_h, 125)
        return frame.tobytes()

    def _draw_rect(self, frame: np.ndarray, center_x: float, center_y: float, width: float, height: float, gray: int) -> None:
        min_x = max(0, int(np.floor((center_x - width / 2.0) / self.virtual_width * self.width)))
        max_x = min(self.width - 1, int(np.ceil((center_x + width / 2.0) / self.virtual_width * self.width)))
        min_y = max(0, int(np.floor((1.0 - (center_y + height / 2.0) / self.virtual_height) * self.height)))
        max_y = min(self.height - 1, int(np.ceil((1.0 - (center_y - height / 2.0) / self.virtual_height) * self.height)))
        if min_x <= max_x and min_y <= max_y:
            frame[min_y:max_y + 1, min_x:max_x + 1] = gray

    def _make_result(self, env_index: int, observation: bytes, reward: float, passed_pipe: bool) -> StepResult:
        return StepResult(
            observation=observation,
            width=self.width,
            height=self.height,
            dtype="uint8",
            reward=reward,
            terminated=not bool(self.alive[env_index]),
            alive=bool(self.alive[env_index]),
            score=int(self.score[env_index]),
            passed_pipe=passed_pipe,
            simulation_time=float(self.simulation_time[env_index]),
        )

    def _check_env_index(self, env_index: int) -> None:
        if env_index < 0 or env_index >= self.num_envs:
            raise IndexError(f"env_index {env_index} is out of range for {self.num_envs} envs")
