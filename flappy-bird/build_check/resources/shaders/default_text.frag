#version 330 core

in vec2 textureCoordinates;
out vec4 fragColor;

uniform sampler2D text;
uniform vec4 textColor;

void main() {
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, textureCoordinates).r);
    fragColor = vec4(textColor) * sampled;
}