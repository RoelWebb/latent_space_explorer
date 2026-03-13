#version 330

uniform vec3 Light;
//uniform sampler2D Texture;

in vec3 v_vert;
in vec3 v_norm;
//in vec2 v_text;

out vec4 f_color;

void main() {
    float lum = clamp(dot(normalize(Light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.8 + 0.2;
    //f_color = vec4(texture(Texture, v_text).rgb * lum, 1.0);
    f_color = vec4(vec3(1.0, 1.0, 1.0) * lum, 1.0);
}