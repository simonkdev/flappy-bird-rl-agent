#ifndef RLOBSERVATIONRENDERER_HPP
#define RLOBSERVATIONRENDERER_HPP

#include <cstdint>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

class FlappyBirdGame;

class RlObservationRenderer {
public:
	RlObservationRenderer(int width, int height, bool debugWindow, GLFWwindow* sharedContext);
	~RlObservationRenderer();

	RlObservationRenderer(const RlObservationRenderer&) = delete;
	RlObservationRenderer& operator=(const RlObservationRenderer&) = delete;

	const std::vector<std::uint8_t>& render(FlappyBirdGame& game);
	int getWidth() const;
	int getHeight() const;
	bool debugWindowShouldClose() const;

private:
	int width;
	int height;
	bool debugEnabled;

	GLuint framebuffer = 0;
		GLuint colorTexture = 0;
		GLFWwindow* mainContext = nullptr;
		GLuint solidProgram = 0;
		std::vector<std::uint8_t> grayscalePixels;

	GLFWwindow* debugWindow = nullptr;
	GLuint debugTexture = 0;
	GLuint debugProgram = 0;
	GLuint debugVao = 0;
	GLuint debugVbo = 0;
	GLuint debugEbo = 0;

		void createFramebuffer();
		void createSolidRenderResources();
		void drawHighContrastScene(FlappyBirdGame& game) const;
		void drawSolidQuad(const class CaffeineTransformComponent& transform, float gray) const;
		void renderCpuObservation(FlappyBirdGame& game);
		void drawCpuQuad(const class CaffeineTransformComponent& transform, std::uint8_t gray);
		void createDebugResources(GLFWwindow* sharedContext);
	void updateDebugWindow();
	static GLuint compileShader(GLenum type, const char* source);
	static GLuint createProgram(const char* vertexSource, const char* fragmentSource);
};

#endif //RLOBSERVATIONRENDERER_HPP
