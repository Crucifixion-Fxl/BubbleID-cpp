# BubbleID 包信息

## 包结构

```
bubbleID-package/
├── CMakeLists.txt          # CMake 构建配置文件
├── README.md               # 主要文档
├── INSTALL.md              # 详细安装指南
├── install.sh              # 安装脚本
├── .gitignore              # Git 忽略文件
├── include/                # 头文件目录
│   ├── bubbleID/
│   │   └── bubbleID.h     # 主 API 头文件
│   ├── yolov8_seg/        # YOLOv8 分割相关
│   │   ├── yolov8_seg_onnx.h
│   │   └── yolov8_seg_utils.h
│   └── ocsort/            # OCSort 追踪器
│       ├── OCSort.hpp
│       ├── Association.hpp
│       ├── KalmanBoxTracker.hpp
│       ├── KalmanFilter.hpp
│       ├── lapjv.hpp
│       └── Utilities.hpp
├── src/                    # 源文件目录
│   ├── bubbleID.cpp       # 主实现文件
│   ├── yolov8_seg_onnx.cpp
│   ├── yolov8_seg_utils.cpp
│   └── ocsort/            # OCSort 实现
│       ├── Association.cpp
│       ├── KalmanBoxTracker.cpp
│       ├── KalmanFilter.cpp
│       ├── lapjv.cpp
│       ├── OCSort.cpp
│       └── Utilities.cpp
└── examples/              # 示例代码
    └── example.cpp
```

## 主要修改

1. **路径标准化**: 将所有 `opencv4/opencv2` 改为标准的 `opencv2`
2. **include 路径**: 修改为相对路径，适配新的包结构
3. **硬编码路径**: 移除了硬编码的模型路径，使用构造函数参数
4. **文件组织**: 按照标准的 C++ 库结构组织文件

## 依赖项

### 必需依赖
- OpenCV 4.x
- Eigen3
- ONNX Runtime
- C++17 编译器

### 可选依赖
- Python3 + NumPy + matplotlib (用于可视化功能)

## 使用方式

### 作为库使用

```cmake
# 在您的 CMakeLists.txt 中
add_subdirectory(bubbleID-package)
target_link_libraries(your_target bubbleID)
```

### 安装到系统

```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make
sudo make install
```

然后在代码中：
```cpp
#include <bubbleID/bubbleID.h>
```

## 注意事项

1. 确保 ONNX Runtime 已正确安装并配置路径
2. 模型文件路径需要在调用时提供，不再硬编码
3. 所有输出文件路径基于 `savefolder` 和 `extension` 参数生成
