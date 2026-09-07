#ifndef PLAYERMOVEMENTSYSTEM_HPP 
#define PLAYERMOVEMENTSYSTEM_HPP

#include <caffeine-gl/base.hpp>
#include <caffeine-gl/components/CaffeineVelocityComponent.hpp>
#include <components/PlayerMovementComponent.hpp>

class PlayerMovementSystem {
public:
    static void update(CaffeineWorld& world, CaffeineWindow& window, float deltaTime, float currentAccelerationFactor);
};

inline void PlayerMovementSystem::update(CaffeineWorld &world, CaffeineWindow& window, float deltaTime, float currentAccelerationFactor) {
    const auto& movementPool = world.getPool<PlayerMovementComponent>();

    for (const CaffeineEntity entity : movementPool.entities) {
        if (!world.hasComponent<CaffeineVelocityComponent>(entity)) continue;

        auto& velocityComponent = world.getComponent<CaffeineVelocityComponent>(entity);
        auto& movementComponent = world.getComponent<PlayerMovementComponent>(entity);
        auto& transformComponent = world.getComponent<CaffeineTransformComponent>(entity);

        if (window.keys[movementComponent.jumpKey] && !window.processedKeys[movementComponent.jumpKey]) {
            window.processedKeys[movementComponent.jumpKey] = true;
            velocityComponent.velocity->y = movementComponent.jumpBoost + (movementComponent.jumpBoost * currentAccelerationFactor * 0.4f);
        }

        velocityComponent.velocity->y += movementComponent.gravity * deltaTime * currentAccelerationFactor;

        float rotation = std::abs(velocityComponent.velocity->y) / movementComponent.jumpBoost * 10.0f;
        (velocityComponent.velocity->y >= 0) ? transformComponent.rotation = rotation : transformComponent.rotation = -rotation;
    }
}

#endif //PLAYERMOVEMENTSYSTEM_HPP