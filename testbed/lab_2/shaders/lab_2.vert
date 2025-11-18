#version 450

// Входные атрибуты вершин
layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

// Выходные данные в фрагментный шейдер
layout (location = 0) out vec3 f_position_world;
layout (location = 1) out vec3 f_normal_world;
layout (location = 2) out vec2 f_uv;

// Uniform буфер сцены
layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
};

// Push constants с матрицей модели
layout (push_constant) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
    float shininess;
    vec3 specular_color;
    float use_texture;
    int texture_index;
    float _pad[2];
};

void main() {
	// Преобразование в мировые координаты
	vec4 world_position = model * vec4(v_position, 1.0f);
	vec3 world_normal = normalize(mat3(model) * v_normal);
	gl_Position = view_projection * world_position;
	f_position_world = world_position.xyz;
	f_normal_world = world_normal;
	f_uv = v_uv;
}