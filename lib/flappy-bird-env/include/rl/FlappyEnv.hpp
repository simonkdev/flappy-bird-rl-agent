#ifndef FLAPPYENV_HPP
#define FLAPPYENV_HPP

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include <game/FlappyBirdGame.hpp>
#include <rl/RlObservationRenderer.hpp>

struct FlappyEnvConfig {
	int observationWidth = 42;
	int observationHeight = 42;
	int ticksPerStep = 4;
	float fixedDeltaTime = 1.0f / 60.0f;
	bool debugWindow = false;
	bool showGameWindow = false;
	std::optional<unsigned int> seed;
	std::optional<std::filesystem::path> resourceRoot;
};

struct FlappyEnvStepResult {
	const std::vector<std::uint8_t>* observation = nullptr;
	float reward = 0.0f;
	bool terminated = false;
	bool alive = false;
	int score = 0;
	bool passedPipe = false;
	float simulationTime = 0.0f;
	bool hasNextPipe = false;
	float birdY = 0.0f;
	float nextPipeX = 0.0f;
	float nextGapCenterY = 0.0f;
	float nextGapHalfHeight = 0.0f;
};

class FlappyEnv {
public:
	explicit FlappyEnv(const FlappyEnvConfig& config = {});

	FlappyEnvStepResult reset(std::optional<unsigned int> seed = std::nullopt);
	FlappyEnvStepResult step(int action);

	int observationWidth() const;
	int observationHeight() const;
	int ticksPerStep() const;
	float fixedDeltaTime() const;
	bool debugWindowShouldClose() const;

private:
	FlappyEnvConfig config;
	FlappyBirdGame game;
	std::unique_ptr<RlObservationRenderer> observationRenderer;
	std::vector<std::uint8_t> lastObservation;
	bool terminated = false;
	unsigned int episodeCounter = 0;

	FlappyEnvStepResult makeResult(float reward, bool passedPipe);
	void renderGameWindowIfEnabled();
};

#endif //FLAPPYENV_HPP
