#include <rl/FlappyEnv.hpp>

#include <cctype>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::string base64Encode(const std::vector<std::uint8_t>& data) {
	static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	std::string output;
	output.reserve(((data.size() + 2) / 3) * 4);

	for (std::size_t i = 0; i < data.size(); i += 3) {
		const unsigned int a = data[i];
		const unsigned int b = i + 1 < data.size() ? data[i + 1] : 0;
		const unsigned int c = i + 2 < data.size() ? data[i + 2] : 0;
		const unsigned int triple = (a << 16) | (b << 8) | c;

		output.push_back(alphabet[(triple >> 18) & 0x3F]);
		output.push_back(alphabet[(triple >> 12) & 0x3F]);
		output.push_back(i + 1 < data.size() ? alphabet[(triple >> 6) & 0x3F] : '=');
		output.push_back(i + 2 < data.size() ? alphabet[triple & 0x3F] : '=');
	}

	return output;
}

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
	          << base64Encode(*result.observation)
	          << std::endl;
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
