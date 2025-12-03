// Стандартные библиотеки C++
#include <cstdint>      // Целочисленные типы фиксированного размера
#include <climits>      // Константы пределов типов
#include <cstring>      // Функции работы со строками
#include <vector>       // Динамические массивы
#include <iostream>     // Ввод-вывод в консоль
#include <fstream>      // Работа с файлами
#include <cmath>        // Математические функции

// Библиотека Veekay для работы с Vulkan
#include <veekay/veekay.hpp>

// API Vulkan для работы с GPU
#include <vulkan/vulkan_core.h>
// Библиотека ImGui для создания UI
#include <imgui.h>

namespace {

// Максимальное количество моделей в сцене
constexpr uint32_t max_models = 1024;
// Максимальное количество точечных источников света
constexpr uint32_t max_point_lights = 4;

// Вершина меша: позиция, нормаль, UV-координаты
struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv;
};

// Направленный источник света
struct DirectionalLight {
	veekay::vec3 direction;
	float _pad0;
	veekay::vec3 color;
	float intensity;
};

// Точечный/прожекторный источник света
// type: 0.0 = точечный, 1.0 = прожектор
struct PointLight {
	veekay::vec3 position;
	float type;
	veekay::vec3 direction;    // для прожектора
	float cutOff;              // внутренний угол конуса
	veekay::vec3 color;
	float intensity;
	float constant;            // затухание: константа
	float linear;              // затухание: линейное
	float quadratic;           // затухание: квадратичное
	float outerCutOff;         // внешний угол конуса
};

// Uniform-буфер сцены
struct SceneUniforms {
	veekay::mat4 view_projection;
	DirectionalLight directional_light;
	veekay::vec3 camera_position;
	float _pad0;
};

// Push constants для каждой модели
struct ModelUniforms {
	veekay::mat4 model;
	veekay::vec3 albedo_color;     // базовый цвет
	float shininess;               // блеск (для Blinn-Phong)
	veekay::vec3 specular_color;   // цвет бликов
	float use_texture;
	int texture_index;
	float _pad[2];
};

// Storage buffer для массива источников света
struct PointLightsSSBO {
	PointLight lights[max_point_lights];
};

// Меш - геометрия объекта (вершины и индексы)
struct Mesh {
	veekay::graphics::Buffer* vertex_buffer;  // Буфер вершин на GPU
	veekay::graphics::Buffer* index_buffer;   // Буфер индексов (порядок соединения вершин)
	uint32_t indices;                         // Количество индексов для отрисовки
};

// Трансформация объекта в 3D пространстве
struct Transform {
	veekay::vec3 position = {};                    // Позиция в мире
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f};      // Масштаб по осям X, Y, Z
	veekay::vec3 rotation = {};                    // Углы поворота (в градусах) по осям X, Y, Z

	// Вычисляет итоговую матрицу трансформации
	veekay::mat4 matrix() const;
};

// Модель - полное описание 3D объекта в сцене
struct Model {
	Mesh mesh;                    // Геометрия объекта
	Transform transform;          // Трансформация (позиция, поворот, масштаб)
	veekay::vec3 albedo_color;    // Базовый цвет
	veekay::vec3 specular_color;  // Цвет бликов
	float shininess;              // Блеск поверхности
	bool use_texture = false;     // Использовать ли текстуру
	int texture_index = 0;        // Индекс текстуры
};

// Камера - точка наблюдения в 3D мире
struct Camera {
	// Константы по умолчанию для параметров камеры
	constexpr static float default_fov = 60.0f;           // Угол обзора (field of view)
	constexpr static float default_near_plane = 0.01f;    // Ближняя плоскость отсечения
	constexpr static float default_far_plane = 100.0f;    // Дальняя плоскость отсечения

	veekay::vec3 position = {};  // Позиция камеры в мире
	float yaw = 90.0f;           // Угол поворота по горизонтали (влево-вправо)
	float pitch = 0.0f;          // Угол наклона по вертикали (вверх-вниз)

	float fov = default_fov;                // Угол обзора
	float near_plane = default_near_plane;  // Ближняя плоскость
	float far_plane = default_far_plane;    // Дальняя плоскость

	// Методы для вычисления матриц и векторов камеры
	veekay::mat4 view() const;                              // Матрица вида (view matrix)
	veekay::mat4 view_projection(float aspect_ratio) const; // Комбинированная матрица view*projection
	veekay::vec3 get_front_vector() const;                  // Вектор направления взгляда
	veekay::vec3 get_right_vector() const;                  // Вектор вправо от камеры
	veekay::vec3 get_up_vector() const;                     // Вектор вверх от камеры
};

// Глобальные переменные сцены
inline namespace {
	// Камера с начальной позицией
	Camera camera{
		.position = {0.0f, 0.5f, 5.0f}  // Начальная позиция: немного выше земли, сзади
	};

	std::vector<Model> models;  // Все модели в сцене
	
	// Направленный свет (солнце) с начальными параметрами
	DirectionalLight directional_light {
		.direction = {-0.5f, -1.0f, -0.5f},  // Светит сверху и сбоку
		.color = {1.0f, 1.0f, 1.0f},         // Белый свет
		.intensity = 0.5f                     // Средняя яркость
	};

	PointLightsSSBO point_lights_data;  // Данные всех точечных источников света
}

// Глобальные ресурсы Vulkan
inline namespace {
	// Шейдерные модули 
	VkShaderModule vertex_shader_module;    // Вершинный шейдер
	VkShaderModule fragment_shader_module;  // Фрагментный шейдер

	// Дескрипторы (механизм связывания ресурсов с шейдерами)
	VkDescriptorPool descriptor_pool;           // Пул для выделения дескрипторов
	VkDescriptorSetLayout descriptor_set_layout; // Схема расположения ресурсов
	VkDescriptorSet descriptor_set;             // Набор дескрипторов (ссылки на буферы)

	// Pipeline - конфигурация графического конвейера
	VkPipelineLayout pipeline_layout;  // Схема данных pipeline
	VkPipeline pipeline;               // Сам графический pipeline

	// Буферы данных на GPU
	veekay::graphics::Buffer* scene_uniforms_buffer;  // Буфер данных сцены
	veekay::graphics::Buffer* model_uniforms_buffer;  // Буфер данных моделей
	veekay::graphics::Buffer* point_lights_ssbo;      // Буфер источников света

	// Геометрия примитивов
	Mesh plane_mesh;  // Плоскость (пол)
	Mesh cube_mesh;   // Куб

	// Текстура-заглушка (отображается когда текстура не загружена)
	veekay::graphics::Texture* missing_texture;
	VkSampler missing_texture_sampler;  // Сэмплер для текстуры
}

// Конвертация градусов в радианы
float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

// Нормализация вектора (приведение длины к 1)
veekay::vec3 normalize(const veekay::vec3& v) {
	float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	return len > 0.0f ? veekay::vec3{v.x / len, v.y / len, v.z / len} : v;
}

// Векторное произведение (cross product) - вектор перпендикулярный обоим
veekay::vec3 cross(const veekay::vec3& a, const veekay::vec3& b) {
	return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// Скалярное произведение (dot product) - проекция одного вектора на другой
float dot(const veekay::vec3& a, const veekay::vec3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Вычисление матрицы трансформации: Translation * Rotation * Scale
veekay::mat4 Transform::matrix() const {
	// Матрица переноса (позиция)
	auto t = veekay::mat4::translation(position);

	// Конвертация углов поворота в радианы
	float rad_x = toRadians(rotation.x);
	float rad_y = toRadians(rotation.y);
	float rad_z = toRadians(rotation.z);

	// Вычисление синусов и косинусов для матриц поворота
	float cos_x = cos(rad_x), sin_x = sin(rad_x);
	float cos_y = cos(rad_y), sin_y = sin(rad_y);
	float cos_z = cos(rad_z), sin_z = sin(rad_z);

	// Матрица поворота вокруг оси X
	veekay::mat4 rot_x = {
		1.0f,  0.0f,   0.0f,  0.0f,
		0.0f, cos_x, -sin_x,  0.0f,
		0.0f, sin_x,  cos_x,  0.0f,
		0.0f,  0.0f,   0.0f,  1.0f
	};

	// Матрица поворота вокруг оси Y
	veekay::mat4 rot_y = {
		cos_y, 0.0f, sin_y, 0.0f,
		0.0f, 1.0f,  0.0f, 0.0f,
		-sin_y, 0.0f, cos_y, 0.0f,
		0.0f, 0.0f,  0.0f, 1.0f
	};

	// Матрица поворота вокруг оси Z
	veekay::mat4 rot_z = {
		cos_z, -sin_z, 0.0f, 0.0f,
		sin_z,  cos_z, 0.0f, 0.0f,
		0.0f,   0.0f, 1.0f, 0.0f,
		0.0f,   0.0f, 0.0f, 1.0f
	};

	// Комбинированная матрица поворота (порядок: Z * Y * X)
	auto r = rot_z * rot_y * rot_x;
	// Матрица масштабирования
	auto s = veekay::mat4::scaling(scale);

	// Итоговая матрица: сначала масштаб, потом поворот, потом перенос
	return t * r * s;
}

// Вычисление вектора направления взгляда камеры
veekay::vec3 Camera::get_front_vector() const {
	float rad_yaw = toRadians(yaw), rad_pitch = toRadians(pitch), cos_pitch = cos(rad_pitch);
	// Сферические координаты преобразуются в декартовы
	return normalize({cos(rad_yaw) * cos_pitch, sin(rad_pitch), sin(rad_yaw) * cos_pitch});
}

// Вычисление вектора "вправо" от камеры
veekay::vec3 Camera::get_right_vector() const {
	// Векторное произведение направления взгляда и мирового вектора "вверх"
	return normalize(cross(get_front_vector(), {0.0f, 1.0f, 0.0f}));
}

// Вычисление вектора "вверх" от камеры
veekay::vec3 Camera::get_up_vector() const {
	// Векторное произведение векторов "вправо" и "вперед"
	return cross(get_right_vector(), get_front_vector());
}

// Вычисление матрицы вида (view matrix) - преобразование из мировых координат в координаты камеры
veekay::mat4 Camera::view() const {
	veekay::vec3 f = get_front_vector();  // Направление взгляда
	veekay::vec3 r = normalize(cross(f, {0.0f, 1.0f, 0.0f}));  // Вектор вправо
	veekay::vec3 u = cross(r, f);  // Вектор вверх

	// Построение матрицы вида (look-at matrix)
	return {
		r.x, u.x, -f.x, 0.0f,
		r.y, u.y, -f.y, 0.0f,
		r.z, u.z, -f.z, 0.0f,
		-dot(r, position), -dot(u, position), dot(f, position), 1.0f
	};
}

// Вычисление комбинированной матрицы view * projection
veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	// Матрица проекции (перспективная)
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
	projection[1][1] *= -1;  // Инверсия Y 
	return view() * projection;
}

// Загрузка скомпилированного шейдера из файла SPIR-V
VkShaderModule loadShaderModule(const char* path) {
	// Открытие файла в бинарном режиме, курсор в конце для определения размера
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	size_t size = file.tellg();  // Получение размера файла
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));  // Буфер для SPIR-V кода
	file.seekg(0);  // Возврат в начало файла
	file.read(reinterpret_cast<char*>(buffer.data()), size);  // Чтение данных
	file.close();

	// Создание шейдерного модуля Vulkan
	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;  // Ошибка создания
	}

	return result;
}

// Инициализация: создание pipeline, буферов, мешей, источников света
void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;

	{ // ===== СОЗДАНИЕ ГРАФИЧЕСКОГО PIPELINE =====
		
		// Загрузка вершинного шейдера
		vertex_shader_module = loadShaderModule("testbed/lab_2/shaders/lab_2.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		// Загрузка фрагментного шейдера
		fragment_shader_module = loadShaderModule("testbed/lab_2/shaders/lab_2.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		// Описание стадий шейдеров в pipeline
		VkPipelineShaderStageCreateInfo stage_infos[2];

		// Вершинный шейдер (обрабатывает каждую вершину)
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",  // Имя функции входа в шейдере
		};

		// Фрагментный шейдер (обрабатывает каждый пиксель)
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// Описание формата вершинного буфера
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,                        // Номер binding point
			.stride = sizeof(Vertex),            // Размер одной вершины в байтах
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,  // Данные для каждой вершины
		};

		// Описание атрибутов вершины (position, normal, uv)
		VkVertexInputAttributeDescription attributes[] = {
			{  // Атрибут 0: позиция (vec3)
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,  // 3 float компонента
				.offset = offsetof(Vertex, position),
			},
			{  // Атрибут 1: нормаль (vec3)
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, normal),
			},
			{  // Атрибут 2: UV координаты (vec2)
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,  // 2 float компонента
				.offset = offsetof(Vertex, uv),
			},
		};

		// Конфигурация входных данных вершин
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		// Конфигурация сборки примитивов (как соединять вершины)
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,  // Рисуем треугольники
		};

		// Конфигурация растеризации (как превращать треугольники в пиксели)
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,      // Заполнять треугольники
			.cullMode = VK_CULL_MODE_NONE,            // Не отсекать задние грани
			.frontFace = VK_FRONT_FACE_CLOCKWISE,     // Передние грани по часовой стрелке
			.lineWidth = 1.0f,
		};

		// Конфигурация мультисэмплинга (сглаживание)
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,  // Без мультисэмплинга
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		// Настройка viewport (область рендеринга)
		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,  // Ближняя граница глубины
			.maxDepth = 1.0f,  // Дальняя граница глубины
		};

		// Настройка scissor (прямоугольник отсечения)
		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		// Конфигурация viewport и scissor
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.pViewports = &viewport,
			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// Конфигурация теста глубины (depth test)
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,                          // Включить тест глубины
			.depthWriteEnable = true,                         // Записывать глубину
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,   // Пиксель ближе или равен
		};

		// Конфигурация смешивания цветов (color blending)
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |  // Записывать все каналы RGBA
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,
			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		// ===== СОЗДАНИЕ ДЕСКРИПТОРОВ (связывание ресурсов с шейдерами) =====
		
		{
			// Размеры пула дескрипторов (сколько дескрипторов каждого типа можно выделить)
			VkDescriptorPoolSize pools[] = {
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,          // Uniform буферы
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,  // Динамические uniform буферы
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,          // Storage буферы (SSBO)
					.descriptorCount = 8,
				},
			};

			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 1,  // Максимум 1 descriptor set
				.poolSizeCount = sizeof(pools) / sizeof(pools[0]),
				.pPoolSizes = pools,
			};

			if (vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor pool\n";
				veekay::app.running = false;
				return;
			}
		}

		{
			// Описание binding'ов в descriptor set layout
			VkDescriptorSetLayoutBinding bindings[] = {
				{  // Binding 0: SceneUniforms (данные сцены)
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{  // Binding 1: ModelUniforms (данные моделей, динамический offset)
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{  // Binding 2: PointLightsSSBO (массив точечных источников света)
					.binding = 2,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,  // Только для фрагментного шейдера
				},
			};

			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
				.pBindings = bindings,
			};

			if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				veekay::app.running = false;
				return;
			}
		}

		{
			// Выделение descriptor set из пула
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &descriptor_set_layout,
			};

			if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set\n";
				veekay::app.running = false;
				return;
			}
		}

		// Настройка push constants (быстрая передача небольших данных в шейдеры)
		VkPushConstantRange push_constant_range{
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			.offset = 0,
			.size = sizeof(ModelUniforms),  // Размер данных модели
		};

		// Создание layout для pipeline (описывает все ресурсы)
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &push_constant_range,
		};

		if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}

		// Финальное создание графического pipeline
		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};

		if (vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	// ===== СОЗДАНИЕ БУФЕРОВ НА GPU =====

	// Буфер для данных сцены (камера, направленный свет)
	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,  // Данные будут записаны позже
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	// Буфер для данных всех моделей (с выравниванием для динамического offset)
	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	// Буфер для массива точечных источников света
	point_lights_ssbo = new veekay::graphics::Buffer(
		sizeof(PointLightsSSBO),
		nullptr,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	{
		// Создание сэмплера (определяет как читать текстуру)
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};

		if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}

		// Создание текстуры 2x2 с шахматным узором (черный-розовый)
		uint32_t pixels[] = {
			0xff000000, 0xffff00ff,  // Черный, Розовый
			0xffff00ff, 0xff000000,  // Розовый, Черный
		};

		missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
		                                                VK_FORMAT_B8G8R8A8_UNORM,
		                                                pixels);
	}

	{
		// Описание буферов для дескрипторов
		VkDescriptorBufferInfo buffer_infos[] = {
			{  // SceneUniforms
				.buffer = scene_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(SceneUniforms),
			},
			{  // ModelUniforms (динамический)
				.buffer = model_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(ModelUniforms),
			},
			{  // PointLightsSSBO
				.buffer = point_lights_ssbo->buffer,
				.offset = 0,
				.range = sizeof(PointLightsSSBO),
			}
		};

		// Запись информации о буферах в descriptor set
		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 0,  // Binding 0: SceneUniforms
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 1,  // Binding 1: ModelUniforms (динамический)
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &buffer_infos[1],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 2,  // Binding 2: PointLightsSSBO
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[2],
			},
		};

		// Обновление дескрипторов
		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
		                       write_infos, 0, nullptr);
	}

	{
		// 4 вершины плоскости 20x20 на уровне Y=0
		std::vector<Vertex> vertices = {
			{{-10.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},  // Левый верхний
			{{10.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},   // Правый верхний
			{{10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},  // Правый нижний
			{{-10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}, // Левый нижний
		};

		// Индексы для двух треугольников (образуют квад)
		std::vector<uint32_t> indices = {
			0, 2, 1,  // Первый треугольник
			2, 0, 3   // Второй треугольник
		};

		// Создание буферов на GPU
		plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		plane_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		plane_mesh.indices = uint32_t(indices.size());
	}

	{
		// 24 вершины (по 4 на каждую из 6 граней)
		// Каждая грань имеет свои нормали для правильного освещения
		std::vector<Vertex> vertices = {
			// Передняя грань (Z-)
			{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

			// Правая грань (X+)
			{{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			// Задняя грань (Z+)
			{{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},

			// Левая грань (X-)
			{{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			// Нижняя грань (Y-)
			{{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

			// Верхняя грань (Y+)
			{{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		// Индексы для 12 треугольников (6 граней * 2 треугольника)
		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0,       // Передняя грань
			4, 5, 6, 6, 7, 4,       // Правая грань
			8, 9, 10, 10, 11, 8,    // Задняя грань
			12, 13, 14, 14, 15, 12, // Левая грань
			16, 17, 18, 18, 19, 16, // Нижняя грань
			20, 21, 22, 22, 23, 20, // Верхняя грань
		};

		cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		cube_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		cube_mesh.indices = uint32_t(indices.size());
	}

	// Модель 0: Плоскость (пол) - серая с низким блеском
	models.emplace_back(Model{
		.mesh = plane_mesh,
		.transform = Transform{.position = {0.0f, 0.0f, 0.0f}},
		.albedo_color = {0.8f, 0.8f, 0.8f},      // Светло-серый
		.specular_color = {0.1f, 0.1f, 0.1f},    // Слабые блики
		.shininess = 32.0f
	});

	// Модель 1: Красный куб слева
	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{ .position = {-2.0f, 0.5f, -1.5f}, },
		.albedo_color = {1.0f, 0.0f, 0.0f},      // Красный
		.specular_color = {1.0f, 1.0f, 1.0f},    // Белые блики
		.shininess = 64.0f                        // Высокий блеск
	});

	// Модель 2: Зеленый куб справа
	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{ .position = {1.5f, 0.5f, -0.5f}, },
		.albedo_color = {0.0f, 1.0f, 0.0f},      // Зеленый
		.specular_color = {1.0f, 1.0f, 1.0f},
		.shininess = 64.0f
	});

	// Модель 3: Синий куб в центре
	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{ .position = {0.0f, 0.5f, 1.0f}, },
		.albedo_color = {0.0f, 0.0f, 1.0f},      // Синий
		.specular_color = {1.0f, 1.0f, 1.0f},
		.shininess = 64.0f
	});

	// Свет 0: Оранжевый прожектор (spotlight)
	point_lights_data.lights[0] = {
		.position = {1.5f, 1.0f, 4.5f},           // Позиция в пространстве
		.type = 1.0f,                              // 1.0 = прожектор
		.direction = {-1.5f, -0.5f, -3.0f},       // Направление луча
		.cutOff = cos(toRadians(10.0f)),          // Внутренний угол конуса
		.color = {1.0f, 0.5f, 0.0f},              // Оранжевый цвет
		.intensity = 2.5f,                         // Яркость
		.constant = 1.0f, .linear = 0.09f, .quadratic = 0.032f,  // Параметры затухания
		.outerCutOff = cos(toRadians(17.5f))      // Внешний угол (плавный переход)
	};

	// Свет 1: Синий прожектор
	point_lights_data.lights[1] = {
		.position = {-4.0f, 0.5f, -1.5f},
		.type = 1.0f,
		.direction = {2.0f, 0.0f, 0.0f},          // Светит вправо
		.cutOff = cos(toRadians(12.5f)),
		.color = {0.0f, 0.5f, 1.0f},              // Голубой
		.intensity = 2.0f,
		.constant = 1.0f, .linear = 0.09f, .quadratic = 0.032f,
		.outerCutOff = cos(toRadians(17.5f))
	};

	// Свет 2: Фиолетовый точечный свет (point light)
	point_lights_data.lights[2] = {
		.position = {1.5f, 4.0f, -3.0f},          // Высоко над сценой
		.type = 0.0f,                              // 0.0 = точечный (светит во все стороны)
		.direction = {-1.5f, -1.0f, -7.0f},       // Не используется для точечного света
		.cutOff = cos(toRadians(15.0f)),
		.color = {1.0f, 0.0f, 1.0f},              // Фиолетовый (пурпурный)
		.intensity = 7.0f,                         // Высокая яркость
		.constant = 1.0f, .linear = 0.07f, .quadratic = 0.017f,  // Медленное затухание
		.outerCutOff = cos(toRadians(20.0f))
	};

	// Свет 3: Белый прожектор сверху (имитация потолочного света)
	point_lights_data.lights[3] = {
		.position = {0.0f, 4.0f, 0.0f},           // Прямо над центром
		.type = 1.0f,
		.direction = {0.0f, -1.0f, 0.0f},         // Светит строго вниз
		.cutOff = cos(toRadians(30.0f)),          // Широкий конус
		.color = {1.0f, 1.0f, 1.0f},              // Белый свет
		.intensity = 1.0f,                         // Умеренная яркость
		.constant = 1.0f, .linear = 0.09f, .quadratic = 0.032f,
		.outerCutOff = cos(toRadians(35.0f))
	};
}

// Освобождение всех ресурсов Vulkan
void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	// Удаление текстуры и сэмплера
	vkDestroySampler(device, missing_texture_sampler, nullptr);
	delete missing_texture;

	// Удаление геометрии куба
	delete cube_mesh.index_buffer;
	delete cube_mesh.vertex_buffer;

	// Удаление геометрии плоскости
	delete plane_mesh.index_buffer;
	delete plane_mesh.vertex_buffer;

	// Удаление буферов
	delete model_uniforms_buffer;
	delete scene_uniforms_buffer;
	delete point_lights_ssbo;

	// Удаление дескрипторов
	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

	// Удаление pipeline
	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	
	// Удаление шейдеров
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

// Обновление логики: UI, управление камерой, обновление буферов
void update(double time) {
	// Вычисление времени между кадрами (delta time)
	static double last_time = time;
	double delta_time = time - last_time;
	last_time = time;

	// ===== СОЗДАНИЕ UI С ПОМОЩЬЮ IMGUI =====
	
	ImGui::Begin("Controls:");
	
	// Настройки направленного света
	ImGui::Text("Directional Light");
	ImGui::DragFloat3("Direction", &directional_light.direction.x, 0.01f);
	ImGui::ColorEdit3("Color##dir", &directional_light.color.x);
	ImGui::SliderFloat("Intensity##dir", &directional_light.intensity, 0.0f, 5.0f);
	ImGui::Separator();
	
	// Настройки точечных источников света
	ImGui::Text("Point Lights");
	for(int i = 0; i < max_point_lights; ++i) {
		std::string label = "Point Light " + std::to_string(i);
		if (ImGui::TreeNode(label.c_str())) {
			// Выбор типа света (точечный или прожектор)
			const char* light_types[] = { "Point", "Spot" };
			int current_type = (int)point_lights_data.lights[i].type;
			if (ImGui::Combo("Type", &current_type, light_types, 2)) {
				point_lights_data.lights[i].type = (float)current_type;
			}

			ImGui::DragFloat3("Position", &point_lights_data.lights[i].position.x, 0.1f);

			// Направление показывается только для прожекторов
			if (current_type == 1) {
				ImGui::DragFloat3("Direction", &point_lights_data.lights[i].direction.x, 0.01f);
			}

			ImGui::ColorEdit3("Color", &point_lights_data.lights[i].color.x);
			ImGui::SliderFloat("Intensity", &point_lights_data.lights[i].intensity, 0.0f, 10.0f);
			ImGui::TreePop();
		}
	}
	ImGui::End();
	
	// Управление работает только если UI не в фокусе
	if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow)) {
		using namespace veekay::input;

		// Вращение камеры при зажатой левой кнопке мыши
		if (mouse::isButtonDown(mouse::Button::left)) {
			auto move_delta = mouse::cursorDelta();  // Смещение курсора
			float sensitivity = 0.1f;                 // Чувствительность мыши
			camera.yaw   -= move_delta.x * sensitivity;  // Поворот по горизонтали
			camera.pitch += move_delta.y * sensitivity;  // Наклон по вертикали

			// Ограничение угла наклона (чтобы не перевернуться)
			if(camera.pitch > 89.0f) camera.pitch = 89.0f;
			if(camera.pitch < -89.0f) camera.pitch = -89.0f;
		}

		// Вычисление векторов движения
		veekay::vec3 move_front = camera.get_front_vector();  // Вперед
		veekay::vec3 move_right = camera.get_right_vector();  // Вправо
		veekay::vec3 up = veekay::vec3{0.0f, 1.0f, 0.0f};    // Вверх (мировая ось Y)
		float speed = 2.5f * float(delta_time);               // Скорость с учетом времени кадра

		// Движение камеры по клавишам WASD + Space + Shift
		if (keyboard::isKeyDown(keyboard::Key::w))
			camera.position -= move_front * speed;  // Вперед (в Vulkan Z инвертирован)
		if (keyboard::isKeyDown(keyboard::Key::s))
			camera.position += move_front * speed;  // Назад
		if (keyboard::isKeyDown(keyboard::Key::d))
			camera.position += move_right * speed;  // Вправо
		if (keyboard::isKeyDown(keyboard::Key::a))
			camera.position -= move_right * speed;  // Влево
		if (keyboard::isKeyDown(keyboard::Key::space))
			camera.position -= up * speed;          // Вверх
		if (keyboard::isKeyDown(keyboard::Key::left_shift))
			camera.position += up * speed;          // Вниз
	}

	// Вычисление соотношения сторон экрана
	float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
	
	// Подготовка данных сцены
	SceneUniforms scene_uniforms{
		.view_projection = camera.view_projection(aspect_ratio),
		.directional_light = directional_light,
		.camera_position = camera.position
	};

	// Запись данных в GPU буферы (mapped_region - область памяти, доступная CPU)
	*(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;
	*(PointLightsSSBO*)point_lights_ssbo->mapped_region = point_lights_data;

	// Обновление данных всех моделей с учетом выравнивания памяти
	const size_t alignment = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));
	for (size_t i = 0; i < models.size(); ++i) {
		// Вычисление указателя на данные модели с учетом выравнивания
		auto* ptr = reinterpret_cast<ModelUniforms*>(
			static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment);
		const Model& m = models[i];
		// Запись данных модели
		*ptr = {m.transform.matrix(), m.albedo_color, m.shininess, 
		        m.specular_color, m.use_texture ? 1.0f : 0.0f, m.texture_index};
	}
}

// Отрисовка кадра: запись Vulkan команд для рендеринга
void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	// Сброс командного буфера
	vkResetCommandBuffer(cmd, 0);

	{
		// Начало записи команд
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,  // Одноразовое использование
		};
		vkBeginCommandBuffer(cmd, &info);
	}

	{
		// Значения для очистки экрана
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};  // Темно-серый фон
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};           // Очистка буфера глубины
		VkClearValue clear_values[] = {clear_color, clear_depth};

		// Начало render pass
		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.extent = {
					veekay::app.window_width,
					veekay::app.window_height
				},
			},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};
		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	// Привязка графического pipeline
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero = 0;
	VkBuffer prev_vb = VK_NULL_HANDLE, prev_ib = VK_NULL_HANDLE;  // Кэширование буферов
	const size_t align = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	// Цикл по всем моделям
	for (size_t i = 0; i < models.size(); ++i) {
		const Model& m = models[i];
		const Mesh& mesh = m.mesh;

		// Привязка вершинного буфера (только если изменился)
		if (prev_vb != mesh.vertex_buffer->buffer) {
			prev_vb = mesh.vertex_buffer->buffer;
			vkCmdBindVertexBuffers(cmd, 0, 1, &prev_vb, &zero);
		}

		// Привязка индексного буфера (только если изменился)
		if (prev_ib != mesh.index_buffer->buffer) {
			prev_ib = mesh.index_buffer->buffer;
			vkCmdBindIndexBuffer(cmd, prev_ib, zero, VK_INDEX_TYPE_UINT32);
		}

		// Привязка дескрипторов с динамическим offset'ом для текущей модели
		uint32_t offset = i * align;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                        0, 1, &descriptor_set, 1, &offset);

		// Передача push constants с данными модели
		ModelUniforms push = {m.transform.matrix(), m.albedo_color, m.shininess,
		                      m.specular_color, m.use_texture ? 1.0f : 0.0f, m.texture_index};
		vkCmdPushConstants(cmd, pipeline_layout,
		                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		                   0, sizeof(ModelUniforms), &push);

		// Команда отрисовки индексированной геометрии
		vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
	}

	// Завершение render pass
	vkCmdEndRenderPass(cmd);
	// Завершение записи команд
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,    // Инициализация
		.shutdown = shutdown,  // Завершение работы
		.update = update,      // Обновление каждый кадр
		.render = render,      // Отрисовка каждый кадр
	});
}