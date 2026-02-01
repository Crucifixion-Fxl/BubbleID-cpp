cmake_minimum_required(VERSION 3.16)
project(bubbleID VERSION 1.0.0 LANGUAGES CXX)

set(CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 默认单线程编译（cmake --build 时等效 -j 1，可在命令行用 -j N 覆盖）
set(CMAKE_BUILD_PARALLEL_LEVEL 128)

# 查找依赖包
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CURL REQUIRED)

# spdlog（日志）
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_MASTER_PROJECT OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog)

# ONNX Runtime 路径配置（用户需要根据实际情况修改）
if(NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "ONNX Runtime directory")
endif()
# 若使用 CPU 版 ONNX Runtime（无 CUDA Execution Provider），设为 OFF 避免链接未定义符号
option(USE_ORT_CUDA "Link ONNX Runtime CUDA provider (set OFF for CPU-only ORT)" OFF)

set(ONNXRUNTIME_INCLUDE_PATH "${ONNXRUNTIME_DIR}/include")
file(GLOB ONNXRUNTIME_LIB_PATH "${ONNXRUNTIME_DIR}/lib/*.so")

# Matplot++（Plotvf / Plotbc 绘图）
set(MATPLOTPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MATPLOTPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/matplotplusplus)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_PATH}
)

# 源文件（按功能分目录：yolov8_seg 检测/分割，ocsort 追踪，bubble_id 分析）
set(BUBBLEID_SOURCES
    src/bubble_id/bubble_id.cpp
    src/yolov8_seg/yolov8_seg_onnx.cpp
    src/yolov8_seg/yolov8_seg_utils.cpp
    src/ocsort/Association.cpp
    src/ocsort/KalmanBoxTracker.cpp
    src/ocsort/KalmanFilter.cpp
    src/ocsort/OCSort.cpp
    src/ocsort/LapJv.cpp
    src/ocsort/Utilities.cpp
)

if(NOT USE_ORT_CUDA)
    add_compile_definitions(BUBBLEID_ORT_CPU_ONLY)
endif()

# 创建静态库
add_library(bubble_id STATIC ${BUBBLEID_SOURCES})

# 链接库
target_link_libraries(bubble_id PUBLIC matplot
    spdlog::spdlog
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB_PATH}
)

# 安装配置
install(TARGETS bubble_id
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# 创建示例可执行文件（可选）
option(BUILD_EXAMPLES "Build example programs" ON)
if(BUILD_EXAMPLES)
    add_executable(bubbleID_example examples/example.cpp)
    target_link_libraries(bubbleID_example bubble_id CURL::libcurl)
endif()
