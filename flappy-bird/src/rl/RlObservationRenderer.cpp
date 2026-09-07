#include <rl/RlObservationRenderer.hpp>

#include <array>
#include <iostream>
#include <stdexcept>

#include <glm/gtc/type_ptr.hpp>

#include <game/FlappyBirdGame.hpp>
#include <caffeine-gl/components/CaffeineRenderComponent.hpp>
#include <caffeine-gl/components/CaffeineTransformComponent.hpp>
#include <caffeine-gl/gfx/CaffeineResourceManager.hpp>
#include <caffeine-gl/systems/CaffeineRenderingSystem.hpp>

RlObservationRenderer::RlObservationRenderer(const int width, const int height, const bool debugWindow, GLFWwindow* sharedContext)
	: width(width), height(height), debugEnabled(debugWindow), rgbPixels(static_cast<std::size_t>(width * height * 3)), grayscalePixels(static_cast<std::size_t>(width * height)) {
	if (width <= 0 || height <= 0) {
		throw std::invalid_argument("RL observation dimensions must be positive");
	}

	mainContext = sharedContext;
	createFramebuffer();
	createSolidRenderResources();
	if (debugEnabled) {
		createDebugResources(sharedContext);
	}
}

RlObservationRenderer::~RlObservationRenderer() {
	if (mainContext) {
		glfwMakeContextCurrent(mainContext);
	}
	glDeleteFramebuffers(1, &framebuffer);
	glDeleteTextures(1, &colorTexture);
	glDeleteProgram(solidProgram);

	if (debugWindow) {
		glfwMakeContextCurrent(debugWindow);
		glDeleteTextures(1, &debugTexture);
		glDeleteProgram(debugProgram);
		glDeleteVertexArrays(1, &debugVao);
		glDeleteBuffers(1, &debugVbo);
		glDeleteBuffers(1, &debugEbo);
		glfwDestroyWindow(debugWindow);
		debugWindow = nullptr;
	}
}

const std::vector<std::uint8_t>& RlObservationRenderer::render(FlappyBirdGame& game) {
	game.window->makeContextCurrent();

	GLint previousFramebuffer = 0;
	GLint previousViewport[4]{};
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, &previousFramebuffer);
	glGetIntegerv(GL_VIEWPORT, previousViewport);

	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glViewport(0, 0, width, height);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	drawHighContrastScene(game);

	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, rgbPixels.data());

	for (int y = 0; y < height; ++y) {
		const int sourceY = height - 1 - y;
		for (int x = 0; x < width; ++x) {
			const std::size_t rgbIndex = static_cast<std::size_t>((sourceY * width + x) * 3);
			const std::uint8_t r = rgbPixels[rgbIndex];
			const std::uint8_t g = rgbPixels[rgbIndex + 1];
			const std::uint8_t b = rgbPixels[rgbIndex + 2];
			grayscalePixels[static_cast<std::size_t>(y * width + x)] = static_cast<std::uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
		}
	}

	glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(previousFramebuffer));
	glViewport(previousViewport[0], previousViewport[1], previousViewport[2], previousViewport[3]);

	if (debugEnabled) {
		updateDebugWindow();
		game.window->makeContextCurrent();
	}

	return grayscalePixels;
}

int RlObservationRenderer::getWidth() const {
	return width;
}

int RlObservationRenderer::getHeight() const {
	return height;
}

bool RlObservationRenderer::debugWindowShouldClose() const {
	return debugWindow && glfwWindowShouldClose(debugWindow);
}

void RlObservationRenderer::createFramebuffer() {
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

	glGenTextures(1, &colorTexture);
	glBindTexture(GL_TEXTURE_2D, colorTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		throw std::runtime_error("Failed to create RL observation framebuffer");
	}

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RlObservationRenderer::createSolidRenderResources() {
	constexpr const char* vertexSource = R"(
		#version 330 core
		layout (location = 0) in vec2 coordinates;
		uniform mat4 modelMatrix;
		uniform mat4 projectionMatrix;
		void main() {
			gl_Position = projectionMatrix * modelMatrix * vec4(coordinates, 0.0, 1.0);
		}
	)";

	constexpr const char* fragmentSource = R"(
		#version 330 core
		out vec4 fragColor;
		uniform float gray;
		void main() {
			fragColor = vec4(gray, gray, gray, 1.0);
		}
	)";

	solidProgram = createProgram(vertexSource, fragmentSource);
}

void RlObservationRenderer::drawHighContrastScene(FlappyBirdGame& game) const {
	glUseProgram(solidProgram);
	glUniformMatrix4fv(
		glGetUniformLocation(solidProgram, "projectionMatrix"),
		1,
		GL_FALSE,
		glm::value_ptr(CaffeineRenderingSystem::projectionMatrix));

	for (PipePair* pipePair : game.pipePairs) {
		if (!pipePair || !pipePair->used) {
			continue;
		}

		if (game.world.getComponent<CaffeineRenderComponent>(pipePair->bottomPipe).visible) {
			drawSolidQuad(game.world.getComponent<CaffeineTransformComponent>(pipePair->bottomPipe), 1.0f);
		}
		if (game.world.getComponent<CaffeineRenderComponent>(pipePair->topPipe).visible) {
			drawSolidQuad(game.world.getComponent<CaffeineTransformComponent>(pipePair->topPipe), 1.0f);
		}
	}

	drawSolidQuad(game.world.getComponent<CaffeineTransformComponent>(game.bird), 125.0f / 255.0f);
}

void RlObservationRenderer::drawSolidQuad(const CaffeineTransformComponent& transform, const float gray) const {
	glUniformMatrix4fv(
		glGetUniformLocation(solidProgram, "modelMatrix"),
		1,
		GL_FALSE,
		glm::value_ptr(transform.getModelMatrix()));
	glUniform1f(glGetUniformLocation(solidProgram, "gray"), gray);
	CaffeineResourceManager::getMesh("quad").draw();
}

void RlObservationRenderer::createDebugResources(GLFWwindow* sharedContext) {
	glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	debugWindow = glfwCreateWindow(width * 10, height * 10, "RL Observation", nullptr, sharedContext);
	if (!debugWindow) {
		throw std::runtime_error("Failed to create RL observation debug window");
	}

	glfwMakeContextCurrent(debugWindow);

	constexpr const char* vertexSource = R"(
		#version 330 core
		layout (location = 0) in vec2 aPos;
		layout (location = 1) in vec2 aTexCoord;
		out vec2 TexCoord;
		void main() {
			gl_Position = vec4(aPos, 0.0, 1.0);
			TexCoord = aTexCoord;
		}
	)";

	constexpr const char* fragmentSource = R"(
		#version 330 core
		in vec2 TexCoord;
		out vec4 FragColor;
		uniform sampler2D image;
		void main() {
			float gray = texture(image, TexCoord).r;
			FragColor = vec4(gray, gray, gray, 1.0);
		}
	)";

	debugProgram = createProgram(vertexSource, fragmentSource);

	const std::array<float, 16> vertices = {
		-1.0f, -1.0f, 0.0f, 0.0f,
		 1.0f, -1.0f, 1.0f, 0.0f,
		-1.0f,  1.0f, 0.0f, 1.0f,
		 1.0f,  1.0f, 1.0f, 1.0f,
	};
	const std::array<unsigned int, 6> indices = {0, 1, 2, 1, 2, 3};

	glGenVertexArrays(1, &debugVao);
	glGenBuffers(1, &debugVbo);
	glGenBuffers(1, &debugEbo);

	glBindVertexArray(debugVao);
	glBindBuffer(GL_ARRAY_BUFFER, debugVbo);
	glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(float)), vertices.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, debugEbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(indices.size() * sizeof(unsigned int)), indices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	glEnableVertexAttribArray(1);
	glBindVertexArray(0);

	glGenTextures(1, &debugTexture);
	glBindTexture(GL_TEXTURE_2D, debugTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, grayscalePixels.data());
	glBindTexture(GL_TEXTURE_2D, 0);
}

void RlObservationRenderer::updateDebugWindow() {
	if (!debugWindow) {
		return;
	}

	glfwMakeContextCurrent(debugWindow);
	int framebufferWidth = 0;
	int framebufferHeight = 0;
	glfwGetFramebufferSize(debugWindow, &framebufferWidth, &framebufferHeight);
	glViewport(0, 0, framebufferWidth, framebufferHeight);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(debugProgram);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, debugTexture);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_UNSIGNED_BYTE, grayscalePixels.data());
	glUniform1i(glGetUniformLocation(debugProgram, "image"), 0);
	glBindVertexArray(debugVao);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glfwSwapBuffers(debugWindow);
	glfwPollEvents();
}

GLuint RlObservationRenderer::compileShader(const GLenum type, const char* source) {
	const GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &source, nullptr);
	glCompileShader(shader);

	GLint success = GL_FALSE;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		char infoLog[1024];
		glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
		glDeleteShader(shader);
		throw std::runtime_error(std::string("Failed to compile debug shader: ") + infoLog);
	}

	return shader;
}

GLuint RlObservationRenderer::createProgram(const char* vertexSource, const char* fragmentSource) {
	const GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
	const GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);

	const GLuint program = glCreateProgram();
	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);
	glLinkProgram(program);

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	GLint success = GL_FALSE;
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if (!success) {
		char infoLog[1024];
		glGetProgramInfoLog(program, 1024, nullptr, infoLog);
		glDeleteProgram(program);
		throw std::runtime_error(std::string("Failed to link debug shader program: ") + infoLog);
	}

	return program;
}
