# BubbleID - 气泡识别与追踪 C++ 库

BubbleID 是一个用于气泡检测、追踪和分析的 C++ 库。它基于 YOLOv8-seg 实例分割模型和 OCSort 多目标追踪算法，能够对视频中的气泡进行识别、追踪和数据分析。

## 功能特性

- **气泡检测**: 使用 YOLOv8-seg 模型进行实例分割，检测气泡位置和掩码
- **多目标追踪**: 使用 OCSort 算法追踪气泡在视频中的运动轨迹
- **数据分析**: 
  - 计算蒸汽区域和基础蒸汽区域
  - 统计气泡大小
  - 生成气泡ID索引文件
  - 绘制可视化图表（蒸汽分数、气泡数量、界面速度）

## 系统要求

- **操作系统**: Linux / macOS
- **编译器**: GCC 7+ 或 Clang 10+ (支持 C++17)
- **CMake**: 3.16 或更高版本
- **依赖库**:
  - OpenCV 4.x
  - Eigen3
  - ONNX Runtime
  - Python3 (用于 matplotlibcpp，可选)

## 安装依赖

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libeigen3-dev \
    python3-dev \
    python3-numpy
```

### macOS

```bash
brew install opencv eigen python3 numpy
```

### ONNX Runtime

下载并解压 ONNX Runtime 到 `third_party/onnxruntime` 目录，或设置 `ONNXRUNTIME_DIR` CMake 变量。

## 编译安装

```bash
# 创建构建目录
mkdir build && cd build

# 配置 CMake（根据需要设置 ONNXRUNTIME_DIR）
cmake .. \
    -DONNXRUNTIME_DIR=/path/to/onnxruntime \
    -DCMAKE_INSTALL_PREFIX=/usr/local

# 编译
make -j$(nproc)

# 安装（可选）
sudo make install
```

## 使用方法

### 基本使用

```cpp
#include "bubbleID/bubbleID.h"
#include <iostream>

int main() {
    // 初始化参数
    std::string imagesfolder = "./images";      // 图像文件夹路径
    std::string videopath = "./video.avi";      // 视频文件路径
    std::string savefolder = "./results";       // 结果保存文件夹
    std::string extension = "test1";            // 文件扩展名标识
    std::string modelweights = "./model.onnx";  // 模型权重路径
    std::string device = "cpu";                 // 设备类型: "cpu" 或 "gpu"
    
    // 创建分析对象
    DataAnalysis analysis(
        imagesfolder,
        videopath,
        savefolder,
        extension,
        modelweights,
        device
    );
    
    // 执行气泡检测和追踪（置信度阈值 0.5）
    analysis.Generate(0.5);
    
    // 绘制可视化图表
    analysis.Plotvf();  // 蒸汽分数图
    analysis.Plotbc();  // 气泡数量图
    
    // 分析特定气泡的界面速度（气泡ID=0）
    analysis.PlotInterfaceVelocity(0);
    
    return 0;
}
```

### 输出文件说明

执行 `Generate()` 后，会在 `savefolder` 目录下生成以下文件：

- `bb-Boiling-{extension}.txt`: 原始检测结果（帧ID x1 y1 x2 y2 置信度 类别）
- `bb-Boiling-output-{extension}.txt`: 追踪结果（帧ID,追踪ID,命中次数,x1,y1,x2,y2）
- `vapor_{extension}.txt`: 每帧的蒸汽区域像素数
- `vaporBase_bt-{extension}.txt`: 每帧的基础蒸汽区域像素数
- `bubble_size_bt-{extension}.txt`: 每帧每个气泡的大小
- `bubind_{extension}.txt`: 每个追踪目标在各帧中的边界框索引
- `frames_{extension}.txt`: 每个追踪目标出现的帧号列表
- `class_{extension}.txt`: 每帧中每个检测目标的类别
- `bubclass_{extension}.txt`: 每个追踪目标在各帧中的类别

## API 文档

### DataAnalysis 类

#### 构造函数

```cpp
DataAnalysis(
    const std::string& imagesfolder,    // 图像文件夹路径
    const std::string& videopath,       // 视频文件路径
    const std::string& savefolder,     // 结果保存文件夹
    const std::string& extension,      // 文件扩展名标识
    const std::string& modelweightsloc, // 模型权重路径
    const std::string& device          // 设备类型: "cpu" 或 "gpu"
);
```

#### 主要方法

- `void Generate(float thres = 0.5)`: 执行气泡检测和追踪
- `void Plotvf()`: 绘制蒸汽分数时间序列图
- `void Plotbc()`: 绘制气泡数量时间序列图
- `void PlotInterfaceVelocity(int bubble)`: 绘制指定气泡的界面速度图
- `void saveDataToFiles()`: 保存数据到文件

#### 静态工具方法

- `get_color(int number)`: 根据数字生成颜色
- `restoreMaskToOriginalSize(...)`: 将边界框内的掩码恢复到原图尺寸
- `iouBatch(...)`: 批量计算 IoU
- `getImagePaths(...)`: 递归获取图像文件路径

## 目录结构

```
bubbleID-package/
├── CMakeLists.txt          # CMake 构建配置
├── README.md               # 本文档
├── include/                # 头文件目录
│   ├── bubbleID/
│   │   └── bubbleID.h     # 主头文件
│   ├── yolov8_seg/        # YOLOv8 相关头文件
│   └── ocsort/            # OCSort 追踪器头文件
├── src/                    # 源文件目录
│   ├── bubbleID.cpp
│   ├── yolov8_seg_onnx.cpp
│   ├── yolov8_seg_utils.cpp
│   └── ocsort/            # OCSort 源文件
├── examples/               # 示例代码
│   └── example.cpp
└── third_party/           # 第三方依赖（需要用户自行添加）
    ├── onnxruntime/
    └── matplotlib-cpp/
```

## 注意事项

1. **模型路径**: 在 `Generate()` 和 `PlotInterfaceVelocity()` 方法中，模型路径是硬编码的。需要根据实际情况修改代码中的模型路径，或将其作为参数传入。

2. **图像格式**: 当前实现仅支持 `.jpg` 格式的图像文件。

3. **视频格式**: 支持 OpenCV 支持的所有视频格式（如 `.avi`, `.mp4` 等）。

4. **Python 依赖**: 如果使用 `Plotvf()` 和 `Plotbc()` 功能，需要安装 matplotlibcpp 和 Python matplotlib。

5. **性能**: 对于大型视频，处理时间可能较长。建议使用 GPU 加速。

## 许可证

请参考原项目的许可证文件。

## 贡献

欢迎提交 Issue 和 Pull Request。

## 联系方式

如有问题或建议，请通过 Issue 联系。
