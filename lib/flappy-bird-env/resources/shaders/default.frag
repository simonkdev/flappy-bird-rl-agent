#version 330 core

in vec2 textureCoords;
in vec4 vertexColor;

out vec4 fragColor;

uniform sampler2D image;

void main() {
    fragColor = vec4(vertexColor) * texture(image, textureCoords);
}