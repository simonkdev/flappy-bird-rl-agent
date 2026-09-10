
#include <caffeine-gl/base.hpp>

#include <random>

struct PipePair {
	CaffeineWorld& world;
	CaffeineEntity bottomPipe = world.createEntity();
	CaffeineEntity topPipe = world.createEntity();
	bool used;
	bool scored;

	float gapSize;
	float yOffset;
	glm::vec2 &gameVelocity;
	std::mt19937 &randomGenerator;

	PipePair(CaffeineWorld& world, glm::vec2 &gameVelocity, std::mt19937 &randomGenerator) : world(world), gameVelocity(gameVelocity), randomGenerator(randomGenerator) {
		used = false;
		scored = false;
		
		world.addComponent<CaffeineTransformComponent>(bottomPipe, {glm::vec2(2000, -200.0f),  0.0f, glm::vec2(-140.0f, 800.0f)});
		world.addComponent<CaffeineRenderComponent>(bottomPipe, {0, false});
		world.addComponent<CaffeineMeshComponent>(bottomPipe, CaffeineMeshComponent(&CaffeineResourceManager::getMesh("quad")));
		world.addComponent<CaffeineMaterialComponent>(bottomPipe, {&CaffeineResourceManager::getShader("default"),
			&CaffeineResourceManager::getTexture("pipe")});
		world.addComponent<CaffeineColliderComponent>(bottomPipe, {ColliderType::STATIC, ColliderShape::QUAD, true,
			glm::vec2(0.0f), glm::vec2(std::abs(world.getComponent<CaffeineTransformComponent>(bottomPipe).size.x), world.getComponent<CaffeineTransformComponent>(bottomPipe).size.y), 
			nullptr});
		world.addComponent<CaffeineVelocityComponent>(bottomPipe, CaffeineVelocityComponent(&gameVelocity));


		world.addComponent<CaffeineTransformComponent>(topPipe, {glm::vec2(2000, -200.0f),  180.0f, glm::vec2(140.0f, 800.0f)});
		world.addComponent<CaffeineRenderComponent>(topPipe, {0, false});
		world.addComponent<CaffeineMeshComponent>(topPipe, CaffeineMeshComponent(&CaffeineResourceManager::getMesh("quad")));
		world.addComponent<CaffeineMaterialComponent>(topPipe, {&CaffeineResourceManager::getShader("default"),
			&CaffeineResourceManager::getTexture("pipe")});
		world.addComponent<CaffeineColliderComponent>(topPipe, {ColliderType::STATIC, ColliderShape::QUAD, true,
			glm::vec2(0.0f), glm::vec2(std::abs(world.getComponent<CaffeineTransformComponent>(topPipe).size.x), world.getComponent<CaffeineTransformComponent>(topPipe).size.y), 
			nullptr});
		world.addComponent<CaffeineVelocityComponent>(topPipe, CaffeineVelocityComponent(&gameVelocity));
	}

	void spawn() {
		used = true;
		world.getComponent<CaffeineRenderComponent>(bottomPipe).visible = true;
		world.getComponent<CaffeineRenderComponent>(topPipe).visible = true;
		
		
		std::uniform_int_distribution<int> gapSizeDistribution(1030, 1179);
		gapSize = static_cast<float>(gapSizeDistribution(randomGenerator));

		std::uniform_int_distribution<int> yOffsetDistribution(
			-310,
			1389 - static_cast<int>(gapSize));
		yOffset = static_cast<float>(yOffsetDistribution(randomGenerator));
		
		world.getComponent<CaffeineTransformComponent>(bottomPipe).position = glm::vec2(2000, yOffset);
		world.getComponent<CaffeineTransformComponent>(topPipe).position = glm::vec2(2000, yOffset + gapSize);
	}

	void despawn() {
		used = false;
		scored = false;
		world.getComponent<CaffeineRenderComponent>(bottomPipe).visible = false;
		world.getComponent<CaffeineRenderComponent>(topPipe).visible = false;
		world.getComponent<CaffeineTransformComponent>(bottomPipe).position = glm::vec2(2000, -200.0f);
		world.getComponent<CaffeineTransformComponent>(topPipe).position = glm::vec2(2000, -200.0f);
	}
};
