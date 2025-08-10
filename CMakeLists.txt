cmake_minimum_required(VERSION 3.10)

project(YupTexture)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages via vcpkg.
find_package(zstd CONFIG REQUIRED)

# Compressonator libraries
set(COMPRESSONATOR_ROOT "${CMAKE_SOURCE_DIR}/thirdparty/Compressonator")
set(COMPRESSONATOR_INCLUDE_DIR "${COMPRESSONATOR_ROOT}/${PLATFORM_DIR}/include")
set(COMPRESSONATOR_LIB_DIR "${COMPRESSONATOR_ROOT}/${PLATFORM_DIR}/lib/x64")

# Check if Compressonator exist
if(NOT EXISTS ${COMPRESSONATOR_ROOT})
    message(FATAL_ERROR "Compressonator include directory not found: ${COMPRESSONATOR_ROOT}")
endif()

if(NOT EXISTS ${COMPRESSONATOR_LIB_DIR})
    message(FATAL_ERROR "Compressonator library directory not found: ${COMPRESSONATOR_LIB_DIR}")
endif()

# Compressonator libraries
if(WIN32)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(COMPRESSONATOR_LIBRARIES
            "${COMPRESSONATOR_LIB_DIR}/Compressonator_MDd_DLL.lib"
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(COMPRESSONATOR_LIBRARIES
            "${COMPRESSONATOR_LIB_DIR}/Compressonator_MD_DLL.lib"
        )
    endif()
else()
    message(FATAL_ERROR "Only WIN32 is currently supported")
endif()

# Verify Compressonator libraries exist
foreach(LIB ${COMPRESSONATOR_LIBRARIES})
    if(NOT EXISTS ${LIB})
        message(FATAL_ERROR "Compressonator library not found: ${LIB}")
    endif()
endforeach()

# Add the source files to the project
set(SOURCE_FILES
    bcn_compressor.cpp
    vq_bcn_compressor.cpp
    vq_encoder.cpp
)

# Add the header files to the project
set(HEADER_FILES
    vq_bcn_types.h
    bcn_compressor.h
    vq_bcn_compressor.h
    vq_encoder.h
    oklab_math.hpp
)

# Create the shared library (DLL)
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${COMPRESSONATOR_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${COMPRESSONATOR_LIBRARIES}
	zstd::libzstd
)

target_compile_definitions(${PROJECT_NAME} PRIVATE YUPTEXTURE_EXPORT)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	    # Copy Compressonator DLLs to output directory
		add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"${COMPRESSONATOR_LIB_DIR}/Compressonator_MDd_DLL.dll"
			"$<TARGET_FILE_DIR:${PROJECT_NAME}>"
		)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
	    # Copy Compressonator DLLs to output directory
		add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"${COMPRESSONATOR_LIB_DIR}/Compressonator_MD_DLL.dll"
			"$<TARGET_FILE_DIR:${PROJECT_NAME}>"
		)
    endif()
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Debug/Release specific settings
target_compile_definitions(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)