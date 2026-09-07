#include <iostream>

#include <caffeine-gl/base.hpp>

#include <game/FlappyBirdGame.hpp>

float lastFrame = 0.0;
float currentFrame = 0.0;
float deltaTime = 0.0;

int main() {
	FlappyBirdGame flappyBird;

	while(!flappyBird.gameShouldEnd) {
		currentFrame = static_cast<float>(glfwGetTime());
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		flappyBird.update(deltaTime);
	}
	return 0;
}
