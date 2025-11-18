#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <algorithm>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>

namespace {

constexpr uint32_t major_segments = 20;
constexpr uint32_t minor_segments = 10;
constexpr float tau = 2.0f * static_cast<float>(M_PI);

struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv;
	veekay::vec3 color;
};

struct SceneUniforms {
	veekay::mat4 view_projection;
};

struct PushConstants {
	veekay::mat4 model;
	veekay::vec3 animated_color;
	float padding;
};

struct Mesh {
	veekay::graphics::Buffer* vertex_buffer = nullptr;
	veekay::graphics::Buffer* index_buffer = nullptr;
	uint32_t indices = 0;
};

struct Transform {
	veekay::vec3 position = {};
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
	veekay::vec3 rotation = {};

	veekay::mat4 matrix() const;
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;

	veekay::mat4 view() const;
	veekay::mat4 view_projection(float aspect_ratio) const;
};

inline namespace {
	Camera camera{
		.position = {0.0f, 0.0f, -4.0f}
	};

	Mesh torus_mesh;
	Transform torus_transform{};

	veekay::graphics::Buffer* scene_uniforms_buffer = nullptr;

	VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
	VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

	VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
	VkPipeline pipeline = VK_NULL_HANDLE;

	VkShaderModule vertex_shader_module = VK_NULL_HANDLE;
	VkShaderModule fragment_shader_module = VK_NULL_HANDLE;

	PushConstants push_constants_data{};

	float rotation_speed = 1.0f;
	float rotation_angle = 0.0f;

	double previous_time = 0.0;
}

veekay::mat4 Transform::matrix() const {
	auto translation = veekay::mat4::translation(position);
	auto scaling = veekay::mat4::scaling(scale);

	auto rotation_x = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x);
	auto rotation_y = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y);
	auto rotation_z = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z);

	auto rotation_matrix = rotation_z * rotation_y * rotation_x;

	return translation * rotation_matrix * scaling;
}

veekay::mat4 Camera::view() const {
	auto translation = veekay::mat4::translation(-position);

	auto rotation_x = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, -rotation.x);
	auto rotation_y = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, -rotation.y);
	auto rotation_z = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, -rotation.z);

	auto rotation_matrix = rotation_z * rotation_y * rotation_x;

	return rotation_matrix * translation;
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
	return view() * projection;
}

VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		return VK_NULL_HANDLE;
	}

	size_t size = static_cast<size_t>(file.tellg());
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &module) != VK_SUCCESS) {
		return VK_NULL_HANDLE;
	}

	return module;
}

void createDescriptorResources() {
	VkDevice device = veekay::app.vk_device;

	VkDescriptorPoolSize pool_sizes[] = {
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
		},
	};

	VkDescriptorPoolCreateInfo pool_info{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1,
		.poolSizeCount = static_cast<uint32_t>(std::size(pool_sizes)),
		.pPoolSizes = pool_sizes,
	};

	if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		veekay::app.running = false;
		return;
	}

	VkDescriptorSetLayoutBinding binding{
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};

	VkDescriptorSetLayoutCreateInfo layout_info{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};

	if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		veekay::app.running = false;
		return;
	}

	VkDescriptorSetAllocateInfo allocate_info{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};

	if (vkAllocateDescriptorSets(device, &allocate_info, &descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor set\n";
		veekay::app.running = false;
		return;
	}
}

void updateDescriptorSet() {
	VkDescriptorBufferInfo buffer_info{
		.buffer = scene_uniforms_buffer->buffer,
		.offset = 0,
		.range = sizeof(SceneUniforms),
	};

	VkWriteDescriptorSet write{
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptor_set,
		.dstBinding = 0,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &buffer_info,
	};

	vkUpdateDescriptorSets(veekay::app.vk_device, 1, &write, 0, nullptr);
}

void createPipeline() {
	VkDevice device = veekay::app.vk_device;

	vertex_shader_module = loadShaderModule("testbed/lab_1/shaders/lab_1.vert.spv");
	fragment_shader_module = loadShaderModule("testbed/lab_1/shaders/lab_1.frag.spv");

	if (vertex_shader_module == VK_NULL_HANDLE || fragment_shader_module == VK_NULL_HANDLE) {
		std::cerr << "Failed to load shader modules\n";
		veekay::app.running = false;
		return;
	}

	VkPipelineShaderStageCreateInfo shader_stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		},
	};

	VkVertexInputBindingDescription binding_description{
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	VkVertexInputAttributeDescription attribute_descriptions[] = {
		{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
		{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
		{2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
		{3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)},
	};

	VkPipelineVertexInputStateCreateInfo vertex_input_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding_description,
		.vertexAttributeDescriptionCount = static_cast<uint32_t>(std::size(attribute_descriptions)),
		.pVertexAttributeDescriptions = attribute_descriptions,
	};

	VkPipelineInputAssemblyStateCreateInfo input_assembly_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE,
	};

	VkViewport viewport{
		.x = 0.0f,
		.y = 0.0f,
		.width = static_cast<float>(veekay::app.window_width),
		.height = static_cast<float>(veekay::app.window_height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};

	VkRect2D scissor{
		.offset = {0, 0},
		.extent = {veekay::app.window_width, veekay::app.window_height},
	};

	VkPipelineViewportStateCreateInfo viewport_state_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.pViewports = &viewport,
		.scissorCount = 1,
		.pScissors = &scissor,
	};

	VkPipelineRasterizationStateCreateInfo rasterization_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f,
	};

	VkPipelineMultisampleStateCreateInfo multisample_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	VkPipelineDepthStencilStateCreateInfo depth_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
	};

	VkPipelineColorBlendAttachmentState color_attachment{
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
		                  VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT |
		                  VK_COLOR_COMPONENT_A_BIT,
	};

	VkPipelineColorBlendStateCreateInfo color_blend_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.logicOpEnable = VK_FALSE,
		.attachmentCount = 1,
		.pAttachments = &color_attachment,
	};

	VkPushConstantRange push_constant_range{
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(PushConstants),
	};

	VkDescriptorSetLayout layouts[] = {descriptor_set_layout};

	VkPipelineLayoutCreateInfo pipeline_layout_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = layouts,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_constant_range,
	};

	if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		veekay::app.running = false;
		return;
	}

	VkGraphicsPipelineCreateInfo pipeline_info{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = static_cast<uint32_t>(std::size(shader_stages)),
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input_info,
		.pInputAssemblyState = &input_assembly_info,
		.pViewportState = &viewport_state_info,
		.pRasterizationState = &rasterization_info,
		.pMultisampleState = &multisample_info,
		.pDepthStencilState = &depth_info,
		.pColorBlendState = &color_blend_info,
		.layout = pipeline_layout,
		.renderPass = veekay::app.vk_render_pass,
		.subpass = 0,
	};

	if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		veekay::app.running = false;
	}
}

void createTorusMesh() {
	std::vector<Vertex> vertices;
	vertices.reserve(major_segments * minor_segments);

	auto index_of = [](uint32_t major, uint32_t minor) {
		return major * minor_segments + minor;
	};

	for (uint32_t i = 0; i < major_segments; ++i) {
		float major_angle = tau * static_cast<float>(i) / static_cast<float>(major_segments);
		float cos_major = std::cos(major_angle);
		float sin_major = std::sin(major_angle);

		for (uint32_t j = 0; j < minor_segments; ++j) {
			float minor_angle = tau * static_cast<float>(j) / static_cast<float>(minor_segments);
			float cos_minor = std::cos(minor_angle);
			float sin_minor = std::sin(minor_angle);

			float outer_radius = 1.1f;
			float inner_radius = 0.45f;

			float ring = outer_radius + inner_radius * cos_minor;

			veekay::vec3 position{
				ring * cos_major,
				ring * sin_major,
				inner_radius * sin_minor
			};

			veekay::vec3 normal{
				cos_major * cos_minor,
				sin_major * cos_minor,
				sin_minor
			};

			veekay::vec2 uv{
				static_cast<float>(i) / static_cast<float>(major_segments),
				static_cast<float>(j) / static_cast<float>(minor_segments)
			};

			veekay::vec3 color{
				uv.x,
				uv.y,
				1.0f - uv.x
			};

			vertices.push_back(Vertex{position, normal, uv, color});
		}
	}

	std::vector<uint32_t> indices;
	indices.reserve(major_segments * minor_segments * 6);

	for (uint32_t i = 0; i < major_segments; ++i) {
		uint32_t next_i = (i + 1) % major_segments;

		for (uint32_t j = 0; j < minor_segments; ++j) {
			uint32_t next_j = (j + 1) % minor_segments;

			indices.push_back(index_of(i, j));
			indices.push_back(index_of(next_i, j));
			indices.push_back(index_of(next_i, next_j));

			indices.push_back(index_of(i, j));
			indices.push_back(index_of(next_i, next_j));
			indices.push_back(index_of(i, next_j));
		}
	}

	torus_mesh.vertex_buffer = new veekay::graphics::Buffer(
		vertices.size() * sizeof(Vertex), vertices.data(),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

	torus_mesh.index_buffer = new veekay::graphics::Buffer(
		indices.size() * sizeof(uint32_t), indices.data(),
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

	torus_mesh.indices = static_cast<uint32_t>(indices.size());
}

void initialize(VkCommandBuffer /*cmd*/) {
	createDescriptorResources();
	if (!veekay::app.running) {
		return;
	}

	createPipeline();
	if (!veekay::app.running) {
		return;
	}

	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms), nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	updateDescriptorSet();

	createTorusMesh();

	torus_transform.position = {0.0f, 0.0f, 0.0f};
	torus_transform.scale = {1.0f, 1.0f, 1.0f};
}

void shutdown() {
	VkDevice device = veekay::app.vk_device;

	delete torus_mesh.index_buffer;
	delete torus_mesh.vertex_buffer;
	delete scene_uniforms_buffer;

	if (pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(device, pipeline, nullptr);
	}

	if (pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	}

	if (descriptor_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	}

	if (descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
	}

	if (fragment_shader_module != VK_NULL_HANDLE) {
		vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	}

	if (vertex_shader_module != VK_NULL_HANDLE) {
		vkDestroyShaderModule(device, vertex_shader_module, nullptr);
	}
}

void update(double time) {
	float delta_time = previous_time > 0.0 ? static_cast<float>(time - previous_time) : 0.0f;
	previous_time = time;

	ImGui::Begin("Torus Controls");
	ImGui::SliderFloat("Field of View", &camera.fov, 30.0f, 110.0f);
	ImGui::SliderFloat("Rotation Speed", &rotation_speed, -6.0f, 6.0f, "%.2f rad/s");
	ImGui::End();

	// Camera rotation with mouse
	using namespace veekay::input;
	if (mouse::isButtonDown(mouse::Button::left)) {
		auto delta = mouse::cursorDelta();
		camera.rotation.y += delta.x * 0.005f;
		camera.rotation.x += delta.y * 0.005f;
		camera.rotation.x = std::clamp(camera.rotation.x, -1.5f, 1.5f);
	}

	rotation_angle += rotation_speed * delta_time;
	rotation_angle = std::fmod(rotation_angle, tau);
	if (rotation_angle < 0.0f) {
		rotation_angle += tau;
	}

	torus_transform.rotation = {0.35f, rotation_angle, 0.0f};

	float phase = static_cast<float>(time);
	push_constants_data.animated_color = {
		0.5f + 0.5f * std::sin(phase),
		0.5f + 0.5f * std::sin(phase + 2.0943951f),
		0.5f + 0.5f * std::sin(phase + 4.1887902f)
	};

	push_constants_data.model = torus_transform.matrix();
	push_constants_data.padding = 0.0f;

	float aspect_ratio = static_cast<float>(veekay::app.window_width) /
	                     static_cast<float>(veekay::app.window_height);

	SceneUniforms uniforms{
		.view_projection = camera.view_projection(aspect_ratio),
	};

	*reinterpret_cast<SceneUniforms*>(scene_uniforms_buffer->mapped_region) = uniforms;
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	VkCommandBufferBeginInfo begin_info{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(cmd, &begin_info);

	VkClearValue clear_values[2];
	clear_values[0].color = {{0.05f, 0.07f, 0.11f, 1.0f}};
	clear_values[1].depthStencil = {1.0f, 0};

	VkRenderPassBeginInfo render_pass_info{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = veekay::app.vk_render_pass,
		.framebuffer = framebuffer,
		.renderArea = {
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		},
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	VkDeviceSize offsets[] = {0};
	VkBuffer vertex_buffers[] = {torus_mesh.vertex_buffer->buffer};
	vkCmdBindVertexBuffers(cmd, 0, 1, vertex_buffers, offsets);
	vkCmdBindIndexBuffer(cmd, torus_mesh.index_buffer->buffer, 0, VK_INDEX_TYPE_UINT32);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
	                        0, 1, &descriptor_set, 0, nullptr);

	vkCmdPushConstants(cmd, pipeline_layout,
	                  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	                  0, sizeof(PushConstants), &push_constants_data);

	vkCmdDrawIndexed(cmd, torus_mesh.indices, 1, 0, 0, 0);

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}
