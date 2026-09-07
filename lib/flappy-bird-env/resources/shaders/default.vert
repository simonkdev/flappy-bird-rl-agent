#version 330 core

layout(location = 0) in vec2 coordinates;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;

out vec2 textureCoords;
out vec4 vertexColor;

uniform mat4 modelMatrix;
uniform mat4 projectionMatrix;

void main() {
    textureCoords = uv;
    vertexColor = color;
    gl_Position = projectionMatrix * modelMatrix * vec4(coordinates, 0.0, 1.0);
}