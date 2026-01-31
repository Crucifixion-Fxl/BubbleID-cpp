cmake_minimum_required(VERSION 3.16)
project(bubbleID VERSION 1.0.0 LANGUAGES CXX)

set(CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 默认单线程编译（cmake --build 时等效 -j 1，可在命令行用 -j N 覆盖）
set(CMAKE_BUILD_PARALLEL_LEVEL 1)

# 查找依赖包
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy)
find_package(CURL REQUIRED)

# ONNX Runtime 路径配置（用户需要根据实际情况修改）
if(NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "ONNX Runtime directory")
endif()
# 若使用 CPU 版 ONNX Runtime（无 CUDA Execution Provider），设为 OFF 避免链接未定义符号
option(USE_ORT_CUDA "Link ONNX Runtime CUDA provider (set OFF for CPU-only ORT)" OFF)

set(ONNXRUNTIME_INCLUDE_PATH "${ONNXRUNTIME_DIR}/include")
set(MATPLOTLIBCPP_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/matplotlib-cpp/")
file(GLOB ONNXRUNTIME_LIB_PATH "${ONNXRUNTIME_DIR}/lib/*.so")

# matplotlibcpp 路径（可选，如果使用 Plotvf 和 Plotbc 功能）
if(NOT DEFINED MATPLOTLIBCPP_DIR)
    set(MATPLOTLIBCPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/matplotlib-cpp" CACHE PATH "matplotlib-cpp directory")
endif()

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_PATH}
    ${MATPLOTLIBCPP_INCLUDE_PATH}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

# 源文件
set(BUBBLEID_SOURCES
    src/bubbleID.cpp
    src/yolov8_seg_onnx.cpp
    src/yolov8_seg_utils.cpp
    src/ocsort/Association.cpp
    src/ocsort/KalmanBoxTracker.cpp
    src/ocsort/KalmanFilter.cpp
    src/ocsort/OCSort.cpp
    src/ocsort/lapjv.cpp
    src/ocsort/Utilities.cpp
)

if(NOT USE_ORT_CUDA)
    add_compile_definitions(BUBBLEID_ORT_CPU_ONLY)
endif()

# 创建静态库
add_library(bubbleID STATIC ${BUBBLEID_SOURCES})

# 链接库
target_link_libraries(bubbleID
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB_PATH}
    Python3::Python
    Python3::Module
)

# 安装配置
install(TARGETS bubbleID
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
    target_link_libraries(bubbleID_example bubbleID CURL::libcurl)
endif()
