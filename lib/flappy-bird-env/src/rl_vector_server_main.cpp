#include <rl/FlappyEnv.hpp>

#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct VectorConfig {
	int numEnvs = 4;
	FlappyEnvConfig envConfig;
};

VectorConfig parseArgs(const int argc, char** argv) {
	VectorConfig config;
	for (int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];
		auto requireValue = [&](const char* name) -> std::string {
			if (i + 1 >= argc) {
				throw std::invalid_argument(std::string("Missing value for ") + name);
			}
			return argv[++i];
		};

		if (arg == "--envs") {
			config.numEnvs = std::stoi(requireValue("--envs"));
		} else if (arg == "--width") {
			config.envConfig.observationWidth = std::stoi(requireValue("--width"));
		} else if (arg == "--height") {
			config.envConfig.observationHeight = std::stoi(requireValue("--height"));
		} else if (arg == "--ticks") {
			config.envConfig.ticksPerStep = std::stoi(requireValue("--ticks"));
		} else if (arg == "--dt") {
			config.envConfig.fixedDeltaTime = std::stof(requireValue("--dt"));
		} else {
			throw std::invalid_argument("Unknown argument: " + arg);
		}
	}

	if (config.numEnvs <= 0) {
		throw std::invalid_argument("--envs must be positive");
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

void printResultHeader(const FlappyEnvStepResult& result, const int width, const int height) {
	std::cout << width << ' '
	          << height << ' '
	          << "uint8" << ' '
	          << std::fixed << std::setprecision(6) << result.reward << ' '
	          << (result.terminated ? 1 : 0) << ' '
	          << (result.alive ? 1 : 0) << ' '
	          << result.score << ' '
	          << (result.passedPipe ? 1 : 0) << ' '
	          << std::fixed << std::setprecision(6) << result.simulationTime << ' '
	          << result.observation->size();
}

void printBatch(const char* kind, const std::vector<FlappyEnvStepResult>& results, const int width, const int height) {
	std::cout << "OK " << kind << ' ' << results.size();
	for (const auto& result : results) {
		std::cout << ' ';
		printResultHeader(result, width, height);
	}
	std::cout << std::endl;

	for (const auto& result : results) {
		std::cout.write(
			reinterpret_cast<const char*>(result.observation->data()),
			static_cast<std::streamsize>(result.observation->size()));
	}
	std::cout.flush();
}
}

int main(const int argc, char** argv) {
	try {
		const VectorConfig config = parseArgs(argc, argv);
		std::vector<std::unique_ptr<FlappyEnv>> envs;
		envs.reserve(static_cast<std::size_t>(config.numEnvs));
		for (int i = 0; i < config.numEnvs; ++i) {
			envs.push_back(std::make_unique<FlappyEnv>(config.envConfig));
		}

		std::cout << "READY "
		          << config.numEnvs << ' '
		          << config.envConfig.observationWidth << ' '
		          << config.envConfig.observationHeight << ' '
		          << config.envConfig.ticksPerStep << ' '
		          << std::fixed << std::setprecision(6) << config.envConfig.fixedDeltaTime
		          << std::endl;

		std::string line;
		while (std::getline(std::cin, line)) {
			std::istringstream stream(line);
			std::string command;
			stream >> command;

			try {
					if (command == "RESET_ALL") {
						std::vector<FlappyEnvStepResult> results;
						results.reserve(envs.size());
						for (auto& env : envs) {
							results.push_back(env->reset(parseOptionalSeed(stream)));
						}
						printBatch("RESET_ALL", results, config.envConfig.observationWidth, config.envConfig.observationHeight);
					} else if (command == "RESET_ONE") {
						int envIndex = -1;
						if (!(stream >> envIndex)) {
							throw std::invalid_argument("RESET_ONE requires an env index");
						}
						if (envIndex < 0 || envIndex >= static_cast<int>(envs.size())) {
							throw std::out_of_range("RESET_ONE env index is out of range");
						}
						std::vector<FlappyEnvStepResult> results;
						results.reserve(1);
						results.push_back(envs[static_cast<std::size_t>(envIndex)]->reset(parseOptionalSeed(stream)));
						printBatch("RESET_ONE", results, config.envConfig.observationWidth, config.envConfig.observationHeight);
					} else if (command == "STEP_BATCH") {
						std::vector<FlappyEnvStepResult> results;
					results.reserve(envs.size());
					for (auto& env : envs) {
						int action = -1;
						if (!(stream >> action)) {
							throw std::invalid_argument("STEP_BATCH requires one action per env");
						}
						results.push_back(env->step(action));
					}
					printBatch("STEP_BATCH", results, config.envConfig.observationWidth, config.envConfig.observationHeight);
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
		}
	} catch (const std::exception& e) {
		std::cerr << "flappy_env_vector_server failed: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
