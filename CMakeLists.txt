cmake_minimum_required(VERSION 3.10)

project(YupTexture)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add option for building example
option(BUILD_EXAMPLE "Build the example executable" ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find required packages via vcpkg
find_package(zstd CONFIG REQUIRED)

# Platform detection
if(WIN32)
    set(PLATFORM_DIR "windows")
    set(ARCH_DIR "x64")
elseif(UNIX AND NOT APPLE)
    set(PLATFORM_DIR "linux")
    set(ARCH_DIR "x86_64")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

# Compressonator libraries
set(COMPRESSONATOR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/Compressonator")
set(COMPRESSONATOR_INCLUDE_DIR "${COMPRESSONATOR_ROOT}/include")
set(COMPRESSONATOR_LIB_DIR "${COMPRESSONATOR_ROOT}/${PLATFORM_DIR}/lib/x64")

# Check if Compressonator exists
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
            "${COMPRESSONATOR_LIB_DIR}/Compressonator_MDd.lib"
        )
	elseif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
		set(COMPRESSONATOR_LIBRARIES
			"${COMPRESSONATOR_LIB_DIR}/Compressonator_MD.lib"
		)
    endif()
else()
    set(COMPRESSONATOR_LIBRARIES
        "${COMPRESSONATOR_LIB_DIR}/libCMP_Common.a"
        "${COMPRESSONATOR_LIB_DIR}/libCMP_Compressonator.a"
        "${COMPRESSONATOR_LIB_DIR}/libCMP_Core.a"
        "${COMPRESSONATOR_LIB_DIR}/libCMP_Core_AVX.a"
        "${COMPRESSONATOR_LIB_DIR}/libCMP_Core_AVX512.a"
        "${COMPRESSONATOR_LIB_DIR}/libCMP_Core_SSE.a"
    )
endif()

# Verify Compressonator libraries exist
foreach(LIB ${COMPRESSONATOR_LIBRARIES})
    if(NOT EXISTS ${LIB})
        message(FATAL_ERROR "Compressonator library not found: ${LIB}")
    endif()
endforeach()

# Collect all .cpp source files in the src directory
file(GLOB TEXTURE_SOURCES "src/*.cpp")

# Collect all .h header files in the include directory
file(GLOB TEXTURE_HEADERS "include/*.h")

# Create the shared library (DLL)
add_library(${PROJECT_NAME} SHARED
    thirdparty/colorm/include/colorm.h
    ${TEXTURE_HEADERS}
    ${TEXTURE_SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    include
    thirdparty/colorm/include/
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${COMPRESSONATOR_INCLUDE_DIR}
)

if(UNIX AND NOT APPLE) # I fucking hate linux
    find_package(Threads REQUIRED)
    find_package(TBB REQUIRED)

    # Link libraries, including OpenMP
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${COMPRESSONATOR_LIBRARIES}
        zstd::libzstd
        OpenMP::OpenMP_CXX
        Threads::Threads
        TBB::tbb
    )
else()
    # Link libraries, including OpenMP
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${COMPRESSONATOR_LIBRARIES}
        zstd::libzstd
        OpenMP::OpenMP_CXX
    )
endif()

# Add OpenMP compile options
if(OpenMP_CXX_FOUND)
    target_compile_options(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE YUPTEXTURE_EXPORT)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
elseif(UNIX AND NOT APPLE)
      target_compile_options(YupTexture PRIVATE -mavx2 -mfma)
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

# Build example executable if enabled
if(BUILD_EXAMPLE)
    # Add example executable
    add_executable(YupTextureExample example/example.cpp)

    # Find required packages via vcpkg
    find_package(Stb REQUIRED)
    
    # Include directories for example
    target_include_directories(YupTextureExample PRIVATE
        include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${COMPRESSONATOR_INCLUDE_DIR}
        ${Stb_INCLUDE_DIR}
    )
    
    # Link libraries for example, including OpenMP
    target_link_libraries(YupTextureExample PRIVATE
        ${PROJECT_NAME}
        ${COMPRESSONATOR_LIBRARIES}
        zstd::libzstd
        OpenMP::OpenMP_CXX
    )
    
    # Add OpenMP compile options for example
    if(OpenMP_CXX_FOUND)
        target_compile_options(YupTextureExample PRIVATE ${OpenMP_CXX_FLAGS})
    endif()
    
    # Apply same compiler warnings as main library
    if(MSVC)
        target_compile_options(YupTextureExample PRIVATE /W4)
    else()
        target_compile_options(YupTextureExample PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    
    # Apply same debug/release definitions
    target_compile_definitions(YupTextureExample PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
    )
endif()