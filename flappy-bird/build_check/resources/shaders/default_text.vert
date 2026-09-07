#version 330 core

layout (location = 0) in vec2 coordinates;
layout (location = 1) in vec2 uv;

out vec2 textureCoordinates;

uniform mat4 projectionMatrix;
uniform mat4 modelMatrix;

void main() {
    gl_Position = projectionMatrix * modelMatrix * vec4(coordinates, 0.0, 1.0);
    textureCoordinates = vec2(uv.x, 1.0f - uv.y); // V coordinate needs to be flipped since FreeType loads textures upside down
}