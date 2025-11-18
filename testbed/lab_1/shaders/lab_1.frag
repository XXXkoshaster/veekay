#version 450

layout (location = 0) in vec3 v_color;
layout (location = 1) in vec3 v_normal;

layout (location = 0) out vec4 out_color;

layout (push_constant) uniform PushConstants {
	mat4 model;
	vec3 animated_color;
	float padding;
} push_constants;

void main() {
	vec3 normal = normalize(v_normal);
	vec3 light_direction = normalize(vec3(0.35, 0.5, 0.8));
	float diffuse = max(dot(normal, light_direction), 0.0);
	float ambient = 0.25;
	vec3 modulation = push_constants.animated_color;

	vec3 color = v_color * modulation * (ambient + diffuse);
	out_color = vec4(color, 1.0);
}
