from __future__ import annotations

from flappy_env import FlappyEnv


def assert_frame(result) -> None:
    assert result.shape == (42, 42, 1)
    assert result.dtype == "uint8"
    assert len(result.observation) == 42 * 42
    assert 0 <= min(result.observation) <= 255
    assert 0 <= max(result.observation) <= 255


def main() -> None:
    with FlappyEnv("build_check/flappy_env_server", seed=99) as env:
        first_reset = env.reset_result(seed=99)
        second_reset = env.reset_result(seed=99)
        assert_frame(first_reset)
        assert_frame(second_reset)
        assert first_reset.observation == second_reset.observation
        assert set(first_reset.observation) == {0, 125}

        no_flap = env.step_result(0)
        assert_frame(no_flap)
        flap = env.step_result(1)
        assert_frame(flap)
        assert flap.simulation_time > no_flap.simulation_time

        env.reset_result(seed=99)
        saw_pipe_pixels = False
        for step in range(80):
            action = 1 if step % 8 == 0 else 0
            high_contrast = env.step_result(action)
            if max(high_contrast.observation) == 255:
                saw_pipe_pixels = True
                break
            if high_contrast.terminated:
                break
        assert saw_pipe_pixels

        try:
            env.step_result(2)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid action did not raise ValueError")

        last = env.reset_result(seed=5)
        terminated_at = None
        for step in range(200):
            last = env.step_result(0)
            if last.terminated:
                terminated_at = step
                break
        assert terminated_at is not None
        assert not last.alive
        assert last.reward <= -1.0

        after_done = env.step_result(0)
        assert after_done.terminated
        assert after_done.simulation_time == last.simulation_time
        assert after_done.observation == last.observation

        reset_after_done = env.reset_result(seed=5)
        assert_frame(reset_after_done)
        assert reset_after_done.alive
        assert not reset_after_done.terminated
        assert reset_after_done.score == 0
        assert reset_after_done.simulation_time == 0.0

    print("flappy_env interface tests passed")


if __name__ == "__main__":
    main()
