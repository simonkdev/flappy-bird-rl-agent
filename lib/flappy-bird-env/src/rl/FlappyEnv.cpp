#include <rl/FlappyEnv.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include <stdexcept>

FlappyEnv::FlappyEnv(const FlappyEnvConfig& config)
	: config(config),
	  game(FlappyBirdGameOptions{
		  .visibleWindow = config.showGameWindow,
		  .startFullscreen = false,
			  .persistHighScore = false,
			  .logResets = false,
			  .useNullPlatform = !config.debugWindow && !config.showGameWindow,
			  .resourceRoot = config.resourceRoot,
		  }) {
	if (config.ticksPerStep <= 0) {
		throw std::invalid_argument("ticksPerStep must be positive");
	}
	if (config.fixedDeltaTime <= 0.0f) {
		throw std::invalid_argument("fixedDeltaTime must be positive");
	}

	observationRenderer = std::make_unique<RlObservationRenderer>(
		config.observationWidth,
		config.observationHeight,
		config.debugWindow,
		game.window->getNativeWindow());
}

FlappyEnvStepResult FlappyEnv::reset(const std::optional<unsigned int> seed) {
	const unsigned int effectiveSeed = seed.value_or(config.seed.value_or(episodeCounter));
	episodeCounter++;
	game.resetGame(effectiveSeed);
	terminated = false;
	lastObservation = observationRenderer->render(game);
	renderGameWindowIfEnabled();
	return makeResult(0.0f, false);
}

FlappyEnvStepResult FlappyEnv::step(const int action) {
	if (action != 0 && action != 1) {
		throw std::invalid_argument("Invalid action. Use 0 for no flap or 1 for flap.");
	}

	if (lastObservation.empty()) {
		return reset(config.seed);
	}

	if (terminated) {
		return makeResult(0.0f, false);
	}

	const int previousScore = game.score;
	const bool wasAlive = game.isAlive();

	for (int i = 0; i < config.ticksPerStep && game.isAlive(); ++i) {
		game.simulateTick(config.fixedDeltaTime, action == 1 && i == 0);
	}

	lastObservation = observationRenderer->render(game);
	renderGameWindowIfEnabled();
	terminated = !game.isAlive();

	const int scoreDelta = game.score - previousScore;
	const bool passedPipe = scoreDelta > 0;
	float reward = static_cast<float>(scoreDelta);
	if (wasAlive && terminated) {
		reward -= 1.0f;
	}

	return makeResult(reward, passedPipe);
}

int FlappyEnv::observationWidth() const {
	return config.observationWidth;
}

int FlappyEnv::observationHeight() const {
	return config.observationHeight;
}

int FlappyEnv::ticksPerStep() const {
	return config.ticksPerStep;
}

float FlappyEnv::fixedDeltaTime() const {
	return config.fixedDeltaTime;
}

bool FlappyEnv::debugWindowShouldClose() const {
	return observationRenderer->debugWindowShouldClose();
}

FlappyEnvStepResult FlappyEnv::makeResult(const float reward, const bool passedPipe) {
	const auto& birdTransform = game.world.getComponent<CaffeineTransformComponent>(game.bird);
	PipePair* nextPipe = nullptr;
	float nextPipeX = std::numeric_limits<float>::infinity();
	for (PipePair* pipePair : game.pipePairs) {
		if (!pipePair || !pipePair->used || pipePair->scored) {
			continue;
		}
		const float pipeX = game.world.getComponent<CaffeineTransformComponent>(pipePair->bottomPipe).position.x;
		if (pipeX < nextPipeX) {
			nextPipe = pipePair;
			nextPipeX = pipeX;
		}
	}

	float nextGapCenterY = 0.0f;
	float nextGapHalfHeight = 0.0f;
	if (nextPipe) {
		const auto& bottomTransform = game.world.getComponent<CaffeineTransformComponent>(nextPipe->bottomPipe);
		const auto& topTransform = game.world.getComponent<CaffeineTransformComponent>(nextPipe->topPipe);
		const float bottomPipeTop = bottomTransform.position.y + std::abs(bottomTransform.size.y) / 2.0f;
		const float topPipeBottom = topTransform.position.y - std::abs(topTransform.size.y) / 2.0f;
		nextGapCenterY = (bottomPipeTop + topPipeBottom) / 2.0f;
		nextGapHalfHeight = std::max(
			0.0f,
			(topPipeBottom - bottomPipeTop) / 2.0f - std::abs(birdTransform.size.y) / 2.0f);
	}

	return FlappyEnvStepResult{
		.observation = &lastObservation,
		.reward = reward,
		.terminated = terminated,
		.alive = game.isAlive(),
		.score = game.score,
		.passedPipe = passedPipe,
		.simulationTime = game.getSimulationTime(),
		.hasNextPipe = nextPipe != nullptr,
		.birdY = birdTransform.position.y,
		.nextPipeX = nextPipe ? nextPipeX : 0.0f,
		.nextGapCenterY = nextGapCenterY,
		.nextGapHalfHeight = nextGapHalfHeight,
	};
}

void FlappyEnv::renderGameWindowIfEnabled() {
	if (config.showGameWindow) {
		game.render(0.0f);
	}
}
