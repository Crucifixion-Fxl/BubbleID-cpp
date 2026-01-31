# BubbleID-cpp - 气泡识别与追踪 C++ 核心算法库

> **注意**: 这是 [BubbleID](https://github.com/cldunlap73/BubbleID.git) 项目的 **C++ 核心算法版本**。原项目是一个基于 Python 的气泡识别与追踪框架，用于分析池沸腾图像。本项目将核心算法用 C++ 重新实现，提供更高的性能和更好的集成能力。

## 关于原项目

原项目 [BubbleID](https://github.com/cldunlap73/BubbleID.git) 是一个用于分析池沸腾图像的深度学习框架，来自论文 **"BubbleID: A deep learning framework for bubble interface dynamics analysis"**。它结合了追踪、分割和分类模型，用于离体分类、速度界面预测和气泡统计提取。原项目基于 ocsort 和 detectron2 构建。

## 本项目说明

BubbleID-cpp 是原 Python 项目的 **C++ 核心算法实现版本**，提供了：

- **高性能**: C++ 实现带来更快的执行速度
- **易于集成**: 可作为 C++ 库集成到其他项目中
- **核心功能**: 实现了原项目的核心算法，包括气泡检测、追踪和数据分析

BubbleID-cpp 是一个用于气泡检测、追踪和分析的 C++ 库。它基于 YOLOv8-seg 实例分割模型和 OCSort 多目标追踪算法，能够对视频中的气泡进行识别、追踪和数据分析。

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
    libcurl4-openssl-dev \
    python3-dev \
    python3-numpy
```

> **说明**: `libcurl4-openssl-dev` 用于满足 OpenCV 依赖链中 libgdal/libnetcdf 对 libcurl 的链接需求，缺少时链接示例程序会报 `undefined reference to 'curl_*@CURL_OPENSSL_4'`。

### macOS

```bash
brew install opencv eigen python3 numpy
```

### ONNX Runtime

- 从 [ONNX Runtime 发布页](https://github.com/microsoft/onnxruntime/releases) 下载 **Linux x64** 包，解压到项目下的 `third_party/onnxruntime`，保证存在：
  - `third_party/onnxruntime/include/onnxruntime_cxx_api.h`
  - `third_party/onnxruntime/lib/*.so`
- 若安装在其他路径，配置时指定：`-DONNXRUNTIME_DIR=/path/to/onnxruntime`
- **CPU 版与 CUDA 版**：若使用的是 **仅 CPU** 的 ONNX Runtime（未带 CUDA Execution Provider），配置时请加 `-DUSE_ORT_CUDA=OFF`（默认即为 OFF），否则链接会报 `undefined reference to 'OrtSessionOptionsAppendExecutionProvider_CUDA'`。若使用带 CUDA 的 ORT，可设 `-DUSE_ORT_CUDA=ON`。

### third_party 目录

```
third_party/
├── onnxruntime/    # 必须：ONNX Runtime 解压后的目录（含 include/、lib/）
└── matplotlib-cpp/  # 可选：用于 Plotvf/Plotbc 可视化，仅头文件
```

## 编译安装

```bash
# 创建构建目录
mkdir build && cd build

# 配置 CMake（按需设置）
cmake .. \
    -DONNXRUNTIME_DIR=/path/to/onnxruntime \
    -DUSE_ORT_CUDA=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local
```

**常用选项**：
- `ONNXRUNTIME_DIR`：ONNX Runtime 根目录（未放在 `third_party/onnxruntime` 时必设）。
- `USE_ORT_CUDA`：使用 **CPU 版** ONNX Runtime 时务必设为 `OFF`，否则会链接错误；使用带 CUDA 的 ORT 时可设为 `ON`。

```bash
# 仅使用默认 third_party 路径时的最简配置
cmake ..

# 编译
cmake --build . -j$(nproc)

# 安装（可选）
sudo cmake --install .
```

## 常见编译/链接错误与解决

| 错误现象 | 原因 | 解决方法 |
|----------|------|----------|
| `onnxruntime_cxx_api.h: No such file or directory` | 未放置或未正确指定 ONNX Runtime | 将 ONNX Runtime 解压到 `third_party/onnxruntime`（含 `include/`、`lib/`），或配置时加 `-DONNXRUNTIME_DIR=/path/to/onnxruntime` |
| `undefined reference to 'OrtSessionOptionsAppendExecutionProvider_CUDA'` | 使用的是 **CPU 版** ONNX Runtime，却链接了需要 CUDA 的代码 | 配置时加 `-DUSE_ORT_CUDA=OFF`（默认即为 OFF）；或改用带 CUDA 的 ONNX Runtime 并设 `-DUSE_ORT_CUDA=ON` |
| `undefined reference to 'curl_*@CURL_OPENSSL_4'`（来自 libgdal / libnetcdf） | OpenCV 依赖链需要 libcurl，但未安装或未链接 | 安装开发包：`sudo apt-get install libcurl4-openssl-dev`，并确保 CMake 能找到 CURL（项目已配置 `target_link_libraries(… CURL::libcurl)`） |
| `Association.hpp` / 其他 ocsort 头文件找不到 | 头文件路径与 `#include` 写法不一致 | 源码中应使用 `#include "ocsort/xxx.hpp"`，且 CMake 中已包含 `include/` 目录，无需再单独加 `include/ocsort` |
| 运行时报 `libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`（required by libgdal / libicuuc） | 环境中优先加载了 **Miniconda/conda** 里较旧的 `libstdc++`，而系统 libgdal 等需要更新版符号 | 运行时优先使用系统库：`LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH ./build/bin/bubbleID_example`；或先执行 `conda deactivate` 再运行；或使用 `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 ./build/bin/bubbleID_example` |

若清理后重新配置仍报错，可先删除 `build` 再执行：

```bash
rm -rf build && mkdir build && cd build
cmake .. -DUSE_ORT_CUDA=OFF
cmake --build . -j$(nproc)
```

## 使用方法

### 运行示例程序

若仓库中自带 ONNX 模型文件（如 `model.onnx`、`models/bubble.onnx` 等），可直接在命令行中把模型路径传给示例程序：

```bash
./build/bin/bubbleID_example <图像文件夹> <视频路径> <结果保存目录> <扩展名> <模型路径.onnx> <cpu|gpu>
```

示例（模型在项目根目录或 `models/` 下）：

```bash
./build/bin/bubbleID_example ./images ./video.avi ./results test1 ./model.onnx cpu
# 或
./build/bin/bubbleID_example ./images ./video.avi ./results test1 ./models/your_model.onnx cpu
```

### 基本使用（代码集成）

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
BubbleID-cpp/
├── CMakeLists.txt          # CMake 构建配置
├── README.md               # 本文档
├── INSTALL.md              # 安装说明
├── PACKAGE_INFO.md         # 包信息
├── install.sh              # 安装脚本
├── LICENSE
│
├── include/                # 头文件
│   ├── bubbleID/
│   │   └── bubbleID.h
│   ├── yolov8_seg/         # YOLOv8-seg 头文件
│   └── ocsort/             # OCSort 追踪器头文件
├── src/                    # 源文件
│   ├── bubbleID.cpp
│   ├── yolov8_seg_onnx.cpp
│   ├── yolov8_seg_utils.cpp
│   └── ocsort/
├── examples/               # 示例程序
│   └── example.cpp
│
├── data/                   # 输入数据（图像序列 + 视频，见 data/README.md）
│   ├── README.md
│   ├── Images-120W/        # 帧图像目录（示例）
│   └── 120W.avi            # 视频文件（示例）
├── weights/                # ONNX 模型
│   └── bubble_seg.onnx
├── result/                 # 运行输出目录（Generate 生成的文件）
│   └── .gitkeep
│
├── build/                  # 构建目录（cmake 生成，已在 .gitignore）
└── third_party/            # 第三方依赖（需自行添加 onnxruntime、matplotlib-cpp）
    ├── onnxruntime/
    └── matplotlib-cpp/
```

## 注意事项

1. **编译/链接报错**：若出现 ONNX Runtime、CUDA、curl 或头文件相关错误，请先查看上文 **「常见编译/链接错误与解决」** 表格。

2. **模型路径**: 示例程序已支持通过命令行传入模型路径；若仓库中自带 ONNX 模型，将 `<model_path>` 指向该文件即可（如 `./model.onnx`）。在自行集成代码时，需在构造 `DataAnalysis` 时传入正确的模型路径。

3. **图像格式**: 当前实现仅支持 `.jpg` 格式的图像文件。

4. **视频格式**: 支持 OpenCV 支持的所有视频格式（如 `.avi`, `.mp4` 等）。

5. **Python 依赖**: 如果使用 `Plotvf()` 和 `Plotbc()` 功能，需要安装 matplotlibcpp 和 Python matplotlib。

6. **性能**: 对于大型视频，处理时间可能较长。建议使用 GPU 加速。

## 相关项目

- **原 Python 项目**: [BubbleID](https://github.com/cldunlap73/BubbleID.git) - 基于 Python 的完整框架，包含 GUI 界面和更多功能
- **论文**: BubbleID: A deep learning framework for bubble interface dynamics analysis

## 许可证

本项目遵循原项目的许可证。原项目使用 Apache-2.0 和 MIT 双重许可证。

## 贡献

欢迎提交 Issue 和 Pull Request。

## 联系方式

如有问题或建议，请通过 Issue 联系。

---

**致谢**: 本项目基于 [cldunlap73/BubbleID](https://github.com/cldunlap73/BubbleID.git) 项目，感谢原作者的优秀工作。
