import argparse

import numpy as np

from src import config
from src.environment import FastVectorFlappyEnv, VectorFlappyEnv
from src.ppo import PPO
from train.training import create_checkpoint_managers, restore_checkpoint


def evaluate_seed_batch(ppo, seed_start, episodes, max_steps, target_score, backend):
    environment_type = (
        VectorFlappyEnv if backend == "cpp_vector" else FastVectorFlappyEnv
    )
    env = environment_type(
        num_envs=episodes,
        width=config.OBS_WIDTH,
        height=config.OBS_HEIGHT,
    )
    try:
        buffers = [[] for _ in range(episodes)]
        results = env.reset_all([seed_start + index for index in range(episodes)])
        finished = [False for _ in range(episodes)]
        steps = [0 for _ in range(episodes)]
        scores = [0 for _ in range(episodes)]

        while not all(finished):
            active_indices = [index for index, done in enumerate(finished) if not done]
            states = [
                ppo.stack_frame_with_buffer(results[index].observation, steps[index], buffers[index])
                for index in active_indices
            ]
            actions = ppo.decide_batch_deterministic(states)
            batch_actions = [0 for _ in range(episodes)]
            for index, action in zip(active_indices, actions):
                batch_actions[index] = action

            next_results = env.step_batch(batch_actions)
            for index in active_indices:
                result = next_results[index]
                results[index] = result
                steps[index] += 1
                scores[index] = result.score
                if result.terminated or steps[index] >= max_steps or result.score >= target_score:
                    finished[index] = True
        return scores, steps
    finally:
        env.close()


def evaluate_checkpoint(checkpoint_path, episodes, batch_size, max_steps, target_score, backend):
    ppo = PPO(env_backend=backend)
    try:
        checkpoint, _ = create_checkpoint_managers(ppo, "/tmp/flappyrl-evaluation", 1)
        epoch = restore_checkpoint(ppo, checkpoint, checkpoint_path)
        ppo.vector_env.close()
        ppo.vector_env = None

        scores = []
        steps = []
        for seed_start in range(0, episodes, batch_size):
            batch_episodes = min(batch_size, episodes - seed_start)
            batch_scores, batch_steps = evaluate_seed_batch(
                ppo,
                seed_start,
                batch_episodes,
                max_steps,
                target_score,
                backend,
            )
            scores.extend(batch_scores)
            steps.extend(batch_steps)
            print(f"epoch {epoch}: evaluated seeds {seed_start}-{seed_start + batch_episodes - 1}")
        stats = ppo._summarize_evaluation(scores, steps, target_score)
        return epoch, stats
    finally:
        ppo.close()


def format_stats(label, epoch, stats):
    scores = np.asarray(stats["scores"], dtype=np.float64)
    return (
        f"{label} (epoch {epoch}): "
        f"mean={scores.mean():.2f} "
        f"median={np.median(scores):.2f} "
        f"std={scores.std(ddof=1):.2f} "
        f"min={scores.min():.0f} "
        f"max={scores.max():.0f} "
        f"target_hits={stats['target_hits']}/{stats['episodes']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare two PPO checkpoints on the same deterministic Flappy Bird seeds."
    )
    parser.add_argument(
        "--baseline",
        default="checkpoints/validate-training/best/ckpt-180",
    )
    parser.add_argument(
        "--candidate",
        default="checkpoints/validate-training/latest/ckpt-200",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--target-score", type=int, default=500)
    parser.add_argument(
        "--env-backend",
        choices=["cpp_vector", "fast"],
        default="cpp_vector",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    baseline_epoch, baseline_stats = evaluate_checkpoint(
        args.baseline,
        args.episodes,
        args.batch_size,
        args.max_steps,
        args.target_score,
        args.env_backend,
    )
    candidate_epoch, candidate_stats = evaluate_checkpoint(
        args.candidate,
        args.episodes,
        args.batch_size,
        args.max_steps,
        args.target_score,
        args.env_backend,
    )

    baseline_scores = np.asarray(baseline_stats["scores"], dtype=np.float64)
    candidate_scores = np.asarray(candidate_stats["scores"], dtype=np.float64)
    differences = candidate_scores - baseline_scores
    standard_error = differences.std(ddof=1) / np.sqrt(len(differences))
    confidence_interval = 1.96 * standard_error

    print(format_stats("baseline", baseline_epoch, baseline_stats))
    print(format_stats("candidate", candidate_epoch, candidate_stats))
    print(
        "paired difference (candidate - baseline): "
        f"mean={differences.mean():+.2f} "
        f"median={np.median(differences):+.2f} "
        f"95%_normal_ci=[{differences.mean() - confidence_interval:+.2f}, "
        f"{differences.mean() + confidence_interval:+.2f}] "
        f"wins={np.count_nonzero(differences > 0)} "
        f"losses={np.count_nonzero(differences < 0)} "
        f"ties={np.count_nonzero(differences == 0)}"
    )


if __name__ == "__main__":
    main()
