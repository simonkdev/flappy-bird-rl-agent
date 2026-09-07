#ifndef PLAYERMOVEMENTCOMPONENT_HPP
#define PLAYERMOVEMENTCOMPONENT_HPP

#include <caffeine-gl/base.hpp>

struct PlayerMovementComponent : CaffeineComponent {
	float gravity;
	float jumpBoost;
	int jumpKey;

	explicit PlayerMovementComponent(float gravity, float jumpBoost, int jumpKey):
		gravity(gravity), jumpBoost(jumpBoost), jumpKey(jumpKey) {} 
};

#endif //PLAYERMOVEMENTCOMPONENT_HPP