#ifndef FLAPPYBIRDGAME_HPP
#define FLAPPYBIRDGAME_HPP

#include <filesystem>
#include <optional>

#include <caffeine-gl/base.hpp>
#include <game/PipePair.hpp>
#include <systems/PlayerMovementSystem.hpp>
#include <components/PlayerMovementComponent.hpp>

struct FlappyBirdGameOptions {
	bool visibleWindow = true;
	bool startFullscreen = true;
	bool persistHighScore = true;
	bool logResets = true;
	bool useNullPlatform = false;
	std::optional<std::filesystem::path> resourceRoot;
};

class FlappyBirdGame {
public:
	const float virtualWidth = 1920.0f;
	const float virtualHeight = 1080.0f;

	glm::vec2 birdSpawn;

	glm::vec2 birdVelocity;
	glm::vec2 cloudVelocity;
	glm::vec2 buildingVelocity;
	glm::vec2 bushVelocity;
	glm::vec2 pipeVelocity;

	const float gameVelocityConst = -400.0f;
	const float acceleration = 15.0f;
	float currentAccelerationFactor;

	const float pipeSpawnRate = 1.8f;
	float lastPipeSpawnTime;
	float simulationTime;

	const float boost = 450.0f;
	const float startBoost = 100.0f;
	const float gravity = -2300.0f;

	int score;
	int highScore;

	bool gamePaused;
	bool birdIsDying;
	bool gameShouldEnd;
	

	CaffeineWindow* window;
	CaffeineWorld world;
	CaffeineEntity bird;
	CaffeineEntity backgroundColor, backgroundClouds, backgroundClouds2, backgroundBuildings, backgroundBuildings2, backgroundBushes, backgroundBushes2;
	PipePair *pipePairs[10];
	CaffeineEntity scoreText;
	CaffeineEntity gameOverBackground, gameOverText1, gameOverText2, gameOverText3, gameOverText4;

	std::vector<CaffeineEntity> backgroundEntities;

	explicit FlappyBirdGame(const FlappyBirdGameOptions& options = {});
	~FlappyBirdGame();

	void init();
    void update(float deltaTime);
	void simulateTick(float deltaTime, bool flapAction);
    void updateBackgroundVelocities();
    void render(float deltaTime);
	void renderWorldOnly();

	void processInput();

	void checkGameOver();
	void birdCollisionCallback(CaffeineEntity thisEntity, CaffeineEntity otherEntity);
    void gameOver();
	void birdDyingAnimation(float deltaTime);
	void resetGame();
	void resetGame(unsigned int seed);
	bool isAlive() const;
	float getSimulationTime() const;

	void spawnPipe();
	void despawnPipe();
	void updateScore();
	void respawnBackground();

private:
	FlappyBirdGameOptions options;
	void applyFlapAction();
};

#endif //FLAPPYBIRDGAME_HPP
