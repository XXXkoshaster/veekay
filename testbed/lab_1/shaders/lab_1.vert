#version 450

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;
layout (location = 3) in vec3 in_color;

layout (location = 0) out vec3 v_color;
layout (location = 1) out vec3 v_normal;

layout (binding = 0) uniform SceneUniforms {
	mat4 view_projection;
};

layout (push_constant) uniform PushConstants {
	mat4 model;
	vec3 animated_color;
	float padding;
} push_constants;

void main() {
	vec4 world_position = push_constants.model * vec4(in_position, 1.0);
	gl_Position = view_projection * world_position;

	v_normal = mat3(push_constants.model) * in_normal;
	v_color = in_color;
}
