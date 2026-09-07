cmake_minimum_required(VERSION 3.31)
project(flappy_bird)

add_subdirectory(caffeine-gl)

set(GLFW_BUILD_X11 OFF CACHE BOOL "" FORCE)

set(FLAPPY_BIRD_GAME_SOURCES
    src/game/FlappyBirdGame.cpp
    include/game/FlappyBirdGame.hpp
    include/game/PipePair.hpp
    include/components/PlayerMovementComponent.hpp
    include/systems/PlayerMovementSystem.hpp
)

add_library(flappy_bird_game STATIC
    ${FLAPPY_BIRD_GAME_SOURCES}
)

target_compile_features(flappy_bird_game PUBLIC cxx_std_20)

target_include_directories(flappy_bird_game PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(flappy_bird_game PUBLIC
    caffeine_gl
)

add_executable(flappy_bird
    src/main.cpp
)

target_compile_features(flappy_bird PRIVATE cxx_std_20)

target_link_libraries(flappy_bird PRIVATE
        flappy_bird_game
)

add_executable(flappy_env_server
    src/rl_server_main.cpp
    src/rl/FlappyEnv.cpp
    include/rl/FlappyEnv.hpp
    src/rl/RlObservationRenderer.cpp
    include/rl/RlObservationRenderer.hpp
)

target_compile_features(flappy_env_server PRIVATE cxx_std_20)

target_link_libraries(flappy_env_server PRIVATE
        flappy_bird_game
)

add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/resources
    $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>/resources
)

add_custom_command(TARGET flappy_env_server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/resources
    $<TARGET_FILE_DIR:flappy_env_server>/resources
)
