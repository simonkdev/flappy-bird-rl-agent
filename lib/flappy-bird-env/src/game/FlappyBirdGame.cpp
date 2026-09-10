#include <game/FlappyBirdGame.hpp>
#include <iostream>
#include <fstream>

FlappyBirdGame::FlappyBirdGame(const FlappyBirdGameOptions& options) : options(options) {
	birdSpawn = glm::vec2(250.0f, 540.0f);

	birdVelocity = glm::vec2(0.0f);
	cloudVelocity = glm::vec2(0.0f);
	buildingVelocity = glm::vec2(0.0f);
	bushVelocity = glm::vec2(0.0f);
	pipeVelocity = glm::vec2(0.0f);

	lastPipeSpawnTime = 0.0f;
	simulationTime = 0.0f;
	gamePaused = true;
	birdIsDying = false;
	gameShouldEnd = false;

	highScore = 0;
	score = 0;
	init();
}	


FlappyBirdGame::~FlappyBirdGame() {
	for(int i = 0; i < sizeof(pipePairs) / sizeof(pipePairs[0]); i++) {
		delete pipePairs[i];
	}
	delete window;
	world.clear();
	CaffeineResourceManager::clear();
}

void FlappyBirdGame::init() {
	window = new CaffeineWindow("Flappy Bird", options.visibleWindow, 0, 0, nullptr, options.useNullPlatform);
	window->createViewport();
	if (options.startFullscreen) {
		window->toggleFullscreen();
	}

	
	
		CaffeineResourceManager::setResourceRoot(
			options.resourceRoot.value_or(CaffeineResourceManager::getExecutablePath() / "resources"));

	CaffeineResourceManager::createDefaultMeshes();

	CaffeineResourceManager::loadShader("shaders/default.vert", "shaders/default.frag", nullptr, "default");
	CaffeineResourceManager::loadShader("shaders/default_text.vert", "shaders/default_text.frag", nullptr, "default_text");
	
	CaffeineResourceManager::loadTexture("textures/bird.png", "bird");
	CaffeineResourceManager::getTexture("bird").setTextureParameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	CaffeineResourceManager::getTexture("bird").setTextureParameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	CaffeineResourceManager::loadTexture("textures/pipe.png", "pipe");
	CaffeineResourceManager::getTexture("pipe").setTextureParameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	CaffeineResourceManager::getTexture("pipe").setTextureParameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	CaffeineResourceManager::loadTexture("textures/color.png", "backgroundColor");
	CaffeineResourceManager::getTexture("backgroundColor").setTextureParameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	CaffeineResourceManager::getTexture("backgroundColor").setTextureParameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	CaffeineResourceManager::loadTexture("textures/clouds.png", "backgroundClouds");
	CaffeineResourceManager::getTexture("backgroundClouds").setTextureParameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	CaffeineResourceManager::getTexture("backgroundClouds").setTextureParameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	CaffeineResourceManager::loadTexture("textures/buildings.png", "backgroundBuildings");
	CaffeineResourceManager::getTexture("backgroundBuildings").setTextureParameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	CaffeineResourceManager::getTexture("backgroundBuildings").setTextureParameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	CaffeineResourceManager::loadTexture("textures/bushes.png", "backgroundBushes");
	CaffeineResourceManager::getTexture("backgroundBushes").setTextureParameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	CaffeineResourceManager::getTexture("backgroundBushes").setTextureParameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	CaffeineResourceManager::loadFont("fonts/retro.ttf", "retro");


	bird = world.createEntity();
	world.addComponent<CaffeineTransformComponent>(bird, {birdSpawn,  0.0f, glm::vec2(105.0f, 70.0f)});
	world.addComponent<CaffeineRenderComponent>(bird, {1000, true});
	world.addComponent<CaffeineMeshComponent>(bird, CaffeineMeshComponent(&CaffeineResourceManager::getMesh("quad")));
	world.addComponent<CaffeineMaterialComponent>(bird, {&CaffeineResourceManager::getShader("default"),
			&CaffeineResourceManager::getTexture("bird")});
	world.addComponent<CaffeineColliderComponent>(bird, {ColliderType::DYNAMIC, ColliderShape::QUAD, true,
		glm::vec2(0.0f), glm::vec2(world.getComponent<CaffeineTransformComponent>(bird).size.x, world.getComponent<CaffeineTransformComponent>(bird).size.y), 
		[this] (CaffeineEntity thisEntity, CaffeineEntity otherEntity) {birdCollisionCallback(thisEntity, otherEntity);}});
	world.addComponent<CaffeineVelocityComponent>(bird, CaffeineVelocityComponent(&birdVelocity));
	world.addComponent<PlayerMovementComponent>(bird, PlayerMovementComponent(gravity, boost, GLFW_KEY_UP));

	backgroundColor = world.createEntity();
	world.addComponent<CaffeineTransformComponent>(backgroundColor, {glm::vec2(virtualWidth / 2, virtualHeight / 2),  0.0f, glm::vec2(virtualWidth, virtualHeight)}); 
	world.addComponent<CaffeineRenderComponent>(backgroundColor, {-1000, true});
	world.addComponent<CaffeineMeshComponent>(backgroundColor, CaffeineMeshComponent(&CaffeineResourceManager::getMesh("quad")));
	world.addComponent<CaffeineMaterialComponent>(backgroundColor, {&CaffeineResourceManager::getShader("default"),
			&CaffeineResourceManager::getTexture("backgroundColor")});

	backgroundClouds = world.createEntity();
	world.addComponent<CaffeineTransformComponent>(backgroundClouds, {glm::vec2(virtualWidth / 2, virtualHeight / 2),  0.0f, glm::vec2(virtualWidth, virtualHeight)});
	world.addComponent<CaffeineRenderComponent>(backgroundClouds, {-999, true});
	world.addComponent<CaffeineMeshComponent>(backgroundClouds, CaffeineMeshComponent(&CaffeineResourceManager::getMesh("quad")));
	world.addComponent<CaffeineMaterialComponent>(backgroundClouds, {&CaffeineResourceManager::getShader("default"),
			&CaffeineResourceManager::getTexture("backgroundClouds")});
	world.addComponent<CaffeineVelocityComponent>(backgroundClouds, CaffeineVelocityComponent(&cloudVelocity));
	backgroundEntities.push_back(backgroundClouds);

	backgroundClouds2 = world.createEntity();
	world.addComponent<CaffeineTransformComponent>(backgroundClouds2, {glm::vec2(virtualWidth / 2 + virtualWidth - 10.0f, virtualHeight / 2),  0.0f, glm::vec2(virtualWidth, virtualHeight)});
	world.addComponent<CaffeineRenderComponent>(backgroundClouds2, {-999, true});
	world.addComponent<CaffeineMeshComponent>(backgroundClouds2, CaffeineMeshComponent(&CaffeineResourceManager::getMesh("quad")));
	world.addComponent<CaffeineMaterialComponent>(backgroundClouds2, {&CaffeineResourceManager::getShader("default"),
			&CaffeineResourceManager::getTexture("backgroundClouds")});
	world.addComponent<CaffeineVelocityComponent>(backgroundClouds2, CaffeineVelocityComponent(&cloudVelocity));
	backgroundEntities.push_back(backgroundClouds2);

	backgroundBuildings = world.createEntity();
	world.addComponent<CaffeineTransformComponent>(backgroundBuildings, {glm::vec2(virtualWidth / 2, virtualHeight / 2),  0.0f, glm::vec2(virtualWidth, virtualHeight)}); 
	world.addComponent<CaffeineRenderComponent>(backgroundBuildings, {-998, true});
	world.addComponent<CaffeineMeshComponent>(backgroundBuildings, CaffeineMeshComponent(&CaffeineResourceManager::getMesh("quad")));
	world.addComponent<CaffeineMaterialComponent>(backgroundBuildings, {&CaffeineResourceManager::getShader("default"),
			&CaffeineResourceManager::getTexture("backgroundBuildings")});
	world.addComponent<CaffeineVelocityComponent>(backgroundBuildings, CaffeineVelocityComponent(&buildingVelocity));
	backgroundEntities.push_back(backgroundBuildings);

	backgroundBuildings2 = world.createEntity();
	world.addComponent<CaffeineTransformComponent>(backgroundBuildings2, {glm::vec2(virtualWidth / 2 + virtualWidth - 10.0f, virtualHeight / 2),  0.0f, glm::vec2(virtualWidth, virtualHeight)});
	world.addComponent<CaffeineRenderComponent>(backgroundBuildings2, {-998, true});
	world.addComponent<CaffeineMeshComponent>(backgroundBuildings2, CaffeineMeshComponent(&CaffeineResourceManager::getMesh("quad")));
	world.addComponent<CaffeineMaterialComponent>(backgroundBuildings2, {&CaffeineResourceManager::getShader("default"),
			&CaffeineResourceManager::getTexture("backgroundBuildings")});
	world.addComponent<CaffeineVelocityComponent>(backgroundBuildings2, CaffeineVelocityComponent(&buildingVelocity));
	backgroundEntities.push_back(backgroundBuildings2);

	backgroundBushes = world.createEntity();
	world.addComponent<CaffeineTransformComponent>(backgroundBushes, {glm::vec2(virtualWidth / 2, virtualHeight / 2),  0.0f, glm::vec2(virtualWidth, virtualHeight)});
	world.addComponent<CaffeineRenderComponent>(backgroundBushes, {-997, true});
	world.addComponent<CaffeineMeshComponent>(backgroundBushes, CaffeineMeshComponent(&CaffeineResourceManager::getMesh("quad")));
	world.addComponent<CaffeineMaterialComponent>(backgroundBushes, {&CaffeineResourceManager::getShader("default"),
			&CaffeineResourceManager::getTexture("backgroundBushes")});
	world.addComponent<CaffeineVelocityComponent>(backgroundBushes, CaffeineVelocityComponent(&bushVelocity));
	backgroundEntities.push_back(backgroundBushes);

	backgroundBushes2 = world.createEntity();
	world.addComponent<CaffeineTransformComponent>(backgroundBushes2, {glm::vec2(virtualWidth / 2 + virtualWidth - 10.0f, virtualHeight / 2),  0.0f, glm::vec2(virtualWidth, virtualHeight)});
	world.addComponent<CaffeineRenderComponent>(backgroundBushes2, {-997, true});
	world.addComponent<CaffeineMeshComponent>(backgroundBushes2, CaffeineMeshComponent(&CaffeineResourceManager::getMesh("quad")));
	world.addComponent<CaffeineMaterialComponent>(backgroundBushes2, {&CaffeineResourceManager::getShader("default"),
			&CaffeineResourceManager::getTexture("backgroundBushes")});
	world.addComponent<CaffeineVelocityComponent>(backgroundBushes2, CaffeineVelocityComponent(&bushVelocity));
	backgroundEntities.push_back(backgroundBushes2);

	scoreText = world.createEntity();
	world.addComponent<CaffeineTransformComponent>(scoreText, {glm::vec2(30.0f, virtualHeight - 60.0f),  0.0f, glm::vec2(0.9f)});
	world.addComponent<CaffeineRenderComponent>(scoreText, {1001, true});
	world.addComponent<CaffeineMeshComponent>(scoreText, CaffeineMeshComponent(&CaffeineResourceManager::getMesh("quad")));
	world.addComponent<CaffeineTextComponent>(scoreText, {"score:0", &CaffeineResourceManager::getFont("retro"), &CaffeineResourceManager::getShader("default_text"), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)});


	gameOverBackground = world.createEntity();
	world.addComponent<CaffeineTransformComponent>(gameOverBackground, {glm::vec2(0.0f, -2500.0f),  0.0f, glm::vec2(400.0f)});
	world.addComponent<CaffeineRenderComponent>(gameOverBackground, {1001, true});
	world.addComponent<CaffeineMeshComponent>(gameOverBackground, CaffeineMeshComponent(&CaffeineResourceManager::getMesh("quad")));	
	world.addComponent<CaffeineTextComponent>(gameOverBackground, {"m", &CaffeineResourceManager::getFont("retro"), &CaffeineResourceManager::getShader("default_text"), glm::vec4(0.0f, 0.0f, 0.0f, 0.7f)});

	gameOverText1 = world.createEntity();
	world.addComponent<CaffeineTransformComponent>(gameOverText1, {glm::vec2(470.0f, 580.0f),  0.0f, glm::vec2(2.0f)});
	world.addComponent<CaffeineRenderComponent>(gameOverText1, {1002, true});
	world.addComponent<CaffeineMeshComponent>(gameOverText1, CaffeineMeshComponent(&CaffeineResourceManager::getMesh("quad")));
	world.addComponent<CaffeineTextComponent>(gameOverText1, {"Flappy Bird", &CaffeineResourceManager::getFont("retro"), &CaffeineResourceManager::getShader("default_text"), glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)});
	
	gameOverText2 = world.createEntity();
	world.addComponent<CaffeineTransformComponent>(gameOverText2, {glm::vec2(350.0f, 495.0f),  0.0f, glm::vec2(0.8f)});
	world.addComponent<CaffeineRenderComponent>(gameOverText2, {1002, false});
	world.addComponent<CaffeineMeshComponent>(gameOverText2, CaffeineMeshComponent(&CaffeineResourceManager::getMesh("quad")));
	world.addComponent<CaffeineTextComponent>(gameOverText2, {"Final Score: 0 -- High Score: 0", &CaffeineResourceManager::getFont("retro"), &CaffeineResourceManager::getShader("default_text"), glm::vec4(4.0f, 4.0f, 4.0f, 1.0f)});

	gameOverText3 = world.createEntity();
	world.addComponent<CaffeineTransformComponent>(gameOverText3, {glm::vec2(500.0f, 430.0f),  0.0f, glm::vec2(0.8f)});
	world.addComponent<CaffeineRenderComponent>(gameOverText3, {1002, false});
	world.addComponent<CaffeineMeshComponent>(gameOverText3, CaffeineMeshComponent(&CaffeineResourceManager::getMesh("quad")));
	world.addComponent<CaffeineTextComponent>(gameOverText3, {"All Time High Score: -", &CaffeineResourceManager::getFont("retro"), &CaffeineResourceManager::getShader("default_text"), glm::vec4(4.0f, 4.0f, 4.0f, 1.0f)});	

	gameOverText4 = world.createEntity();
	world.addComponent<CaffeineTransformComponent>(gameOverText4, {glm::vec2(600.0f, 230.0f),  0.0f, glm::vec2(0.7f)});
	world.addComponent<CaffeineRenderComponent>(gameOverText4, {1002, true});
	world.addComponent<CaffeineMeshComponent>(gameOverText4, CaffeineMeshComponent(&CaffeineResourceManager::getMesh("quad")));
	world.addComponent<CaffeineTextComponent>(gameOverText4, {"Press SPACE to start", &CaffeineResourceManager::getFont("retro"), &CaffeineResourceManager::getShader("default_text"), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)});
	

	for(int i = 0; i < std::size(pipePairs); i++) {
		pipePairs[i] = new PipePair(world, pipeVelocity, randomGenerator);
	}
}

void FlappyBirdGame::update(const float  deltaTime) {
	processInput();
	simulateTick(deltaTime, false);
	render(deltaTime);
}

void FlappyBirdGame::simulateTick(const float deltaTime, const bool flapAction) {
	simulationTime += deltaTime;
    updateBackgroundVelocities();
    if (!gamePaused) {
		pipeVelocity.x = gameVelocityConst - (score * acceleration);
		currentAccelerationFactor = pipeVelocity.x / gameVelocityConst;
		if (flapAction) {
			applyFlapAction();
		}
		CaffeineCollisionSystem::update(world);
		PlayerMovementSystem::update(world, *window, deltaTime, currentAccelerationFactor);
		
		world.getComponent<CaffeineTextComponent>(scoreText).text = "score:" + std::to_string(score);
		updateScore();
		spawnPipe();
		despawnPipe();
		respawnBackground();
		
		checkGameOver();
	}
	if (gamePaused && birdIsDying) {birdDyingAnimation(deltaTime);}
	CaffeineVelocitySystem::update(world, deltaTime);
}

void FlappyBirdGame::processInput() {
	if (window->keys[GLFW_KEY_SPACE] && !window->processedKeys[GLFW_KEY_SPACE]) {
		window->processedKeys[GLFW_KEY_SPACE] = true;
		if (gamePaused) {
			resetGame();
		}
	}
	if (window->keys[GLFW_KEY_V] && !window->processedKeys[GLFW_KEY_V]) {
		window->processedKeys[GLFW_KEY_V] = true;
		window->toggleFullscreen();
	}
	if (window->keys[GLFW_KEY_ESCAPE]) {
		gameShouldEnd = true;
	}
}

void FlappyBirdGame::updateBackgroundVelocities() {
    cloudVelocity.x = pipeVelocity.x * 0.3f;
    buildingVelocity.x = pipeVelocity.x * 0.4f;
    bushVelocity.x = pipeVelocity.x * 0.5f;
}

void FlappyBirdGame::spawnPipe() {
	if (simulationTime - lastPipeSpawnTime > pipeSpawnRate / currentAccelerationFactor) {
		for (PipePair*& pipePair : pipePairs) {
			if (!pipePair->used) {
				pipePair->spawn();
				lastPipeSpawnTime = simulationTime;
				break;
			}
		}
	}
}

void FlappyBirdGame::despawnPipe() {
	for (PipePair*& pipePair : pipePairs) {
		if (pipePair->used && pipePair->world.getComponent<CaffeineTransformComponent>(pipePair->bottomPipe).position.x < -100.0f) {
			pipePair->despawn();
		}
	}
}

void FlappyBirdGame::updateScore() {
	for (PipePair*& pipePair : pipePairs) {
		if (world.getComponent<CaffeineTransformComponent>(bird).position.x > pipePair->world.getComponent<CaffeineTransformComponent>(pipePair->bottomPipe).position.x) {
			if (pipePair->used && !pipePair->scored) {
				score ++;
				pipePair->scored = true;
			}
		}
	}
}

void FlappyBirdGame::respawnBackground() {
	for (CaffeineEntity& backgroundEntity : backgroundEntities) {
		if (world.getComponent<CaffeineTransformComponent>(backgroundEntity).position.x < -virtualWidth / 2) {
			world.getComponent<CaffeineTransformComponent>(backgroundEntity).position = glm::vec2(virtualWidth / 2 + virtualWidth - 30.0f, virtualHeight / 2);
		}
	}
}


void FlappyBirdGame::checkGameOver() {
	if (world.getComponent<CaffeineTransformComponent>(bird).position.y < 0.0f || world.getComponent<CaffeineTransformComponent>(bird).position.y > virtualHeight) {
		gameOver();
	}
}

void FlappyBirdGame::birdCollisionCallback(CaffeineEntity thisEntity, CaffeineEntity otherEntity) {
    birdVelocity.y += boost / 2;
	if (birdVelocity.y > boost * 1.5f) {
		birdVelocity.y = boost;
	}
	gameOver();
}

void FlappyBirdGame::gameOver() {
    gamePaused = true;
    birdIsDying = true;
	highScore = std::max(score, highScore);

	int allTimeHighScore = 0;
	if (options.persistHighScore) {
		std::ifstream in("save.txt");
		if (in.is_open()) {
			in >> allTimeHighScore;
			in.close();
		}

		if (highScore > allTimeHighScore) {
			std::ofstream out("save.txt");
			out << highScore;
			out.close();
			world.getComponent<CaffeineTextComponent>(gameOverText3).text = "New All Time High Score " + std::to_string(highScore) + "!";
		}
		else {
			world.getComponent<CaffeineTextComponent>(gameOverText3).text = "All Time High Score: " + std::to_string(allTimeHighScore);
		}
	}
	
    pipeVelocity = glm::vec2(0.0f);
	world.getComponent<CaffeineRenderComponent>(gameOverBackground).visible = true;
	world.getComponent<CaffeineRenderComponent>(gameOverText1).visible = true;
	world.getComponent<CaffeineRenderComponent>(gameOverText2).visible = true;
	world.getComponent<CaffeineRenderComponent>(gameOverText3).visible = true;
	world.getComponent<CaffeineRenderComponent>(gameOverText4).visible = true;
	world.getComponent<CaffeineTextComponent>(gameOverText2).text = "Your Score: " + std::to_string(score) + " -- High Score: " + std::to_string(highScore);
	world.getComponent<CaffeineTextComponent>(gameOverText4).text = "Press SPACE to restart"; 
	world.getComponent<CaffeineTextComponent>(gameOverText1).text = "GAME OVER!";
}

void FlappyBirdGame::birdDyingAnimation(float deltaTime) {
	if (world.getComponent<CaffeineTransformComponent>(bird).rotation < 90.0f) {
		world.getComponent<CaffeineTransformComponent>(bird).rotation += 200.0f * deltaTime;
	}
	else {
		world.getComponent<CaffeineTransformComponent>(bird).rotation += 100.0f * deltaTime;
	}
	if (world.getComponent<CaffeineTransformComponent>(bird).position.y < 0.0f) {
		birdIsDying = false;
	}
	birdVelocity.y += gravity * deltaTime;

}

void FlappyBirdGame::resetGame() {
	score = 0;
	gamePaused = false;
	birdIsDying = false;
	simulationTime = 0.0f;
	lastPipeSpawnTime = 0.0f;
	world.getComponent<CaffeineRenderComponent>(gameOverBackground).visible = false;
	world.getComponent<CaffeineRenderComponent>(gameOverText1).visible = false;
	world.getComponent<CaffeineRenderComponent>(gameOverText2).visible = false;
	world.getComponent<CaffeineRenderComponent>(gameOverText3).visible = false;
	world.getComponent<CaffeineRenderComponent>(gameOverText4).visible = false;

	world.getComponent<CaffeineTransformComponent>(bird).position = birdSpawn;
	world.getComponent<CaffeineTransformComponent>(bird).rotation = 0.0f;
	
	world.getComponent<CaffeineTransformComponent>(backgroundColor).position = glm::vec2(virtualWidth / 2, virtualHeight / 2);
	world.getComponent<CaffeineTransformComponent>(backgroundClouds2).position = glm::vec2(virtualWidth / 2 + virtualWidth - 10.0f, virtualHeight / 2);
	world.getComponent<CaffeineTransformComponent>(backgroundClouds).position = glm::vec2(virtualWidth / 2, virtualHeight / 2);
	world.getComponent<CaffeineTransformComponent>(backgroundBuildings2).position = glm::vec2(virtualWidth / 2 + virtualWidth - 10.0f, virtualHeight / 2);
	world.getComponent<CaffeineTransformComponent>(backgroundBuildings).position = glm::vec2(virtualWidth / 2, virtualHeight / 2);
	world.getComponent<CaffeineTransformComponent>(backgroundBushes2).position = glm::vec2(virtualWidth / 2 + virtualWidth - 10.0f, virtualHeight / 2);
	world.getComponent<CaffeineTransformComponent>(backgroundBushes).position = glm::vec2(virtualWidth / 2, virtualHeight / 2);
	
	birdVelocity = glm::vec2(0.0f, startBoost);
	
	for (PipePair*& pipePair : pipePairs) {
		pipePair->despawn();
	}
	pipeVelocity = glm::vec2(gameVelocityConst, 0.0f);
	if (options.logResets) {
		std::cout << "Game reset!" << std::endl;
	}
}

void FlappyBirdGame::resetGame(const unsigned int seed) {
	randomGenerator.seed(seed);
	resetGame();
}

bool FlappyBirdGame::isAlive() const {
	return !gamePaused && !birdIsDying;
}

float FlappyBirdGame::getSimulationTime() const {
	return simulationTime;
}

void FlappyBirdGame::applyFlapAction() {
	auto& movementComponent = world.getComponent<PlayerMovementComponent>(bird);
	birdVelocity.y = movementComponent.jumpBoost + (movementComponent.jumpBoost * currentAccelerationFactor * 0.4f);
}

void FlappyBirdGame::renderWorldOnly() {
	CaffeineRenderingSystem::update(world);
	CaffeineTextRenderingSystem::update(world);
}

void FlappyBirdGame::render(float deltaTime) {
	renderWorldOnly();
	window->update();
}
