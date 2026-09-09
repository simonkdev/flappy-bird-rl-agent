#include <rl/FlappyEnv.hpp>

#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
void printResult(const char* kind, const FlappyEnvStepResult& result, const int width, const int height) {
	std::cout << "OK " << kind << ' '
	          << width << ' '
	          << height << ' '
	          << "uint8" << ' '
	          << std::fixed << std::setprecision(6) << result.reward << ' '
	          << (result.terminated ? 1 : 0) << ' '
	          << (result.alive ? 1 : 0) << ' '
	          << result.score << ' '
	          << (result.passedPipe ? 1 : 0) << ' '
	          << std::fixed << std::setprecision(6) << result.simulationTime << ' '
	          << (result.hasNextPipe ? 1 : 0) << ' '
	          << std::fixed << std::setprecision(6) << result.birdY << ' '
	          << std::fixed << std::setprecision(6) << result.nextPipeX << ' '
	          << std::fixed << std::setprecision(6) << result.nextGapCenterY << ' '
	          << std::fixed << std::setprecision(6) << result.nextGapHalfHeight << ' '
	          << result.observation->size()
	          << std::endl;
	std::cout.write(
		reinterpret_cast<const char*>(result.observation->data()),
		static_cast<std::streamsize>(result.observation->size()));
	std::cout.flush();
}

void printUsage() {
	std::cerr
		<< "Usage: flappy_env_server [--width N] [--height N] [--ticks N] [--dt SECONDS]\n"
		<< "                         [--seed N] [--debug-window] [--show-game-window]\n";
}

FlappyEnvConfig parseArgs(const int argc, char** argv) {
	FlappyEnvConfig config;

	for (int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];
		auto requireValue = [&](const char* name) -> std::string {
			if (i + 1 >= argc) {
				throw std::invalid_argument(std::string("Missing value for ") + name);
			}
			return argv[++i];
		};

		if (arg == "--width") {
			config.observationWidth = std::stoi(requireValue("--width"));
		} else if (arg == "--height") {
			config.observationHeight = std::stoi(requireValue("--height"));
		} else if (arg == "--ticks") {
			config.ticksPerStep = std::stoi(requireValue("--ticks"));
		} else if (arg == "--dt") {
			config.fixedDeltaTime = std::stof(requireValue("--dt"));
		} else if (arg == "--seed") {
			config.seed = static_cast<unsigned int>(std::stoul(requireValue("--seed")));
		} else if (arg == "--debug-window") {
			config.debugWindow = true;
		} else if (arg == "--show-game-window") {
			config.showGameWindow = true;
		} else if (arg == "--help" || arg == "-h") {
			printUsage();
			std::exit(EXIT_SUCCESS);
		} else {
			throw std::invalid_argument("Unknown argument: " + arg);
		}
	}

	return config;
}

std::optional<unsigned int> parseOptionalSeed(std::istringstream& stream) {
	unsigned int seed = 0;
	if (stream >> seed) {
		return seed;
	}
	return std::nullopt;
}
}

int main(const int argc, char** argv) {
	try {
		FlappyEnvConfig config = parseArgs(argc, argv);
		FlappyEnv env(config);

		std::cout << "READY "
		          << env.observationWidth() << ' '
		          << env.observationHeight() << ' '
		          << env.ticksPerStep() << ' '
		          << std::fixed << std::setprecision(6) << env.fixedDeltaTime()
		          << std::endl;

		std::string line;
		while (std::getline(std::cin, line)) {
			std::istringstream stream(line);
			std::string command;
			stream >> command;

			try {
				if (command == "RESET") {
					printResult("RESET", env.reset(parseOptionalSeed(stream)), env.observationWidth(), env.observationHeight());
				} else if (command == "STEP") {
					int action = -1;
					if (!(stream >> action)) {
						throw std::invalid_argument("STEP requires an action value");
					}
					printResult("STEP", env.step(action), env.observationWidth(), env.observationHeight());
				} else if (command == "INFO") {
					std::cout << "OK INFO "
					          << env.observationWidth() << ' '
					          << env.observationHeight() << ' '
					          << env.ticksPerStep() << ' '
					          << std::fixed << std::setprecision(6) << env.fixedDeltaTime()
					          << std::endl;
				} else if (command == "CLOSE" || command == "QUIT") {
					std::cout << "OK CLOSE" << std::endl;
					break;
				} else if (command.empty()) {
					continue;
				} else {
					throw std::invalid_argument("Unknown command: " + command);
				}
			} catch (const std::exception& e) {
				std::cout << "ERR " << e.what() << std::endl;
			}

			if (env.debugWindowShouldClose()) {
				std::cerr << "RL observation debug window was closed" << std::endl;
			}
		}
	} catch (const std::exception& e) {
		std::cerr << "flappy_env_server failed: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
