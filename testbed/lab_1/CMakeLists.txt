add_executable(lab_1 main.cpp)

set_target_properties(lab_1 PROPERTIES CXX_STANDARD_REQUIRED TRUE CXX_STANDARD 20)

if(MSVC)
    target_compile_options(lab_1 PRIVATE /wd4201)
    target_compile_definitions(lab_1 PRIVATE -D_USE_MATH_DEFINES)
endif()

target_link_libraries(lab_1 PRIVATE veekay Vulkan::Headers)

target_sources(lab_1 PRIVATE
    ${lodepng_SOURCE_DIR}/lodepng.cpp
)

target_include_directories(lab_1 PRIVATE
    ${lodepng_SOURCE_DIR}
)

find_program(GLSLC_FOUND glslc)
if(GLSLC_FOUND)
    set(LAB1_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
    set(LAB1_SHADER_SOURCES
        ${LAB1_SHADER_DIR}/lab_1.vert
        ${LAB1_SHADER_DIR}/lab_1.frag
    )

    set(LAB1_SHADER_BINARIES)
    foreach(SHADER ${LAB1_SHADER_SOURCES})
        get_filename_component(FILE_NAME ${SHADER} NAME)
        set(OUTPUT_SPV ${LAB1_SHADER_DIR}/${FILE_NAME}.spv)
        add_custom_command(
            OUTPUT ${OUTPUT_SPV}
            COMMAND glslc ${SHADER} -o ${OUTPUT_SPV}
            DEPENDS ${SHADER}
            COMMENT "Compiling ${FILE_NAME}"
        )
        list(APPEND LAB1_SHADER_BINARIES ${OUTPUT_SPV})
    endforeach()

    add_custom_target(lab_1_shaders DEPENDS ${LAB1_SHADER_BINARIES})
    add_dependencies(lab_1 lab_1_shaders)
endif()
