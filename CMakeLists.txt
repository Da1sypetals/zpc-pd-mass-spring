cmake_minimum_required(VERSION 3.20)

# basic settings 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CUDA_ARCHITECTURES 86)
set(CUDA_SEPARABLE_COMPILATION ON)
set(ZS_ENABLE_OFB_ACCESS_CHECK ON) # enable container out-of-bound access check
set(ZS_ENABLE_JIT OFF) # disable jit

# add main targets 
project(mytarget LANGUAGES C CXX CUDA)
add_executable(mytarget)
add_library(lib)
target_include_directories(lib PUBLIC include)
file(GLOB_RECURSE sources CONFIGURE_DEPENDS src/*.cpp src/*.hpp src/*.cu src/*.cuh)
target_sources(lib PUBLIC ${sources})

# add zpc
find_package(CUDAToolkit REQUIRED COMPONENTS cufft REQUIRED)
add_subdirectory(extern/zpc)
add_library(zshelper INTERFACE)
target_include_directories(zshelper INTERFACE external/zpc/include)
target_link_libraries(zshelper INTERFACE zensim)
target_compile_features(zshelper INTERFACE cuda_std_17)

# add igl 
add_subdirectory(extern/eigen)
add_subdirectory(extern/libigl)
igl_include(glfw) # enable glfw
igl_include(imgui) # enable imgui

# add choldmod 
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(cholmod REQUIRED)

# handle link dependencies
target_sources(mytarget PUBLIC main.cu)
target_link_libraries(mytarget PUBLIC lib)
target_link_libraries(lib PUBLIC zshelper igl::glfw igl::imgui Eigen3::Eigen cholmod)

# handle resources
# reference: https://stackoverflow.com/questions/34799916/copy-file-from-source-directory-to-binary-directory-using-cmake
configure_file(${CMAKE_SOURCE_DIR}/resources/xxx.txt ${CMAKE_CURRENT_BINARY_DIR}/resources/xxx.txt COPYONLY)

add_definitions(-w)

include(FetchContent)

macro(LinkGLM TARGET ACCESS)
    FetchContent_Declare(
            glm
            GIT_REPOSITORY https://gitee.com/andrew0229/glm.git
            GIT_TAG 0.9.9.8
    )

    FetchContent_GetProperties(glm)

    if (NOT glm_POPULATED)
        FetchContent_Populate(glm)
    endif()

    target_include_directories(${TARGET} ${ACCESS} ${glm_SOURCE_DIR})
endmacro()

LinkGLM(mytarget PRIVATE)
