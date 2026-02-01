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

以下以 **Linux（Ubuntu/Debian）** 为例。

- **操作系统**: Linux
- **编译器**: GCC 7+（支持 C++17）
- **CMake**: 3.16 或更高版本
- **依赖库**: OpenCV 4.x、Eigen3、ONNX Runtime、libcurl、spdlog（日志）、gnuplot（Plotvf/Plotbc 绘图后端）

## 安装依赖

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libeigen3-dev \
    libcurl4-openssl-dev \
    gnuplot
```

> **说明**: `libcurl4-openssl-dev` 用于满足 OpenCV 依赖链中 libgdal/libnetcdf 对 libcurl 的链接需求，缺少时链接示例程序会报 `undefined reference to 'curl_*@CURL_OPENSSL_4'`。

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
└── matplotplusplus/  # 可选：用于 Plotvf/Plotbc 可视化（需系统安装 gnuplot）
```

## 编译与安装

```bash
mkdir build && cd build

# 配置（ONNX Runtime 在 third_party/onnxruntime 时可省略 -DONNXRUNTIME_DIR）
cmake .. \
    -DONNXRUNTIME_DIR=/path/to/onnxruntime \
    -DUSE_ORT_CUDA=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local

# 编译
cmake --build . -j$(nproc)

# 安装到系统（可选）
sudo cmake --install .
```

**CMake 选项**：

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `ONNXRUNTIME_DIR` | `third_party/onnxruntime` | ONNX Runtime 根目录（含 include/、lib/） |
| `USE_ORT_CUDA` | `OFF` | CPU 版 ORT 保持 OFF；带 CUDA 的 ORT 可设为 `ON` |

## 常见编译/链接错误与解决

| 错误现象 | 原因 | 解决方法 |
|----------|------|----------|
| `onnxruntime_cxx_api.h: No such file or directory` | 未放置或未正确指定 ONNX Runtime | 将 ONNX Runtime 解压到 `third_party/onnxruntime`（含 `include/`、`lib/`），或配置时加 `-DONNXRUNTIME_DIR=/path/to/onnxruntime` |
| `undefined reference to 'OrtSessionOptionsAppendExecutionProvider_CUDA'` | 使用的是 **CPU 版** ONNX Runtime，却链接了需要 CUDA 的代码 | 配置时加 `-DUSE_ORT_CUDA=OFF`（默认即为 OFF）；或改用带 CUDA 的 ONNX Runtime 并设 `-DUSE_ORT_CUDA=ON` |
| `undefined reference to 'curl_*@CURL_OPENSSL_4'`（来自 libgdal / libnetcdf） | OpenCV 依赖链需要 libcurl，但未安装或未链接 | 安装开发包：`sudo apt-get install libcurl4-openssl-dev`，并确保 CMake 能找到 CURL（项目已配置 `target_link_libraries(… CURL::libcurl)`） |
| `Association.hpp` / 其他 ocsort 头文件找不到 | 头文件路径与 `#include` 写法不一致 | 源码中应使用 `#include "ocsort/xxx.hpp"`，且 CMake 中已包含 `include/` 目录，无需再单独加 `include/ocsort` |
| 运行时报 `libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`（required by libgdal / libicuuc） | 环境中优先加载了 **Miniconda/conda** 里较旧的 `libstdc++`，而系统 libgdal 等需要更新版符号 | 运行时优先使用系统库：`LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH ./build/bin/bubbleID_example`；或先执行 `conda deactivate` 再运行；或使用 `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 ./build/bin/bubbleID_example` |
| 调用 Plotvf/Plotbc 时报 `gnuplot: not found` | Matplot++ 使用 gnuplot 作为绘图后端，系统未安装 | 安装 gnuplot：`sudo apt-get install gnuplot` |

若清理后重新配置仍报错，可先删除 `build` 再执行：

```bash
rm -rf build && mkdir build && cd build
cmake .. -DUSE_ORT_CUDA=OFF
cmake --build . -j$(nproc)
```

### 验证与运行

在项目根目录执行（需已准备图像目录、视频、ONNX 模型）：

```bash
./build/bin/bubbleID_example <图像文件夹> <视频路径> <结果保存目录> <扩展名> <模型.onnx> cpu
```

结果将保存在“结果保存目录”下，并按类型放入子目录：`data/`（txt）、`detection_vis/`（检测可视化图）、`figures/`（分析图表）。详见下文「输出文件说明」。

### 卸载（若曾安装到系统）

```bash
cd build
sudo xargs rm -f < install_manifest.txt
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
#include "bubble_id/bubble_id.h"
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

### 输出文件说明（分类保存）

所有结果均在 `savefolder` 下按类型放入子文件夹：

- **`savefolder/data/`**（txt 数据）：执行 `Generate()` 后生成
  - `bb-Boiling-{extension}.txt`: 原始检测结果（帧ID x1 y1 x2 y2 置信度 类别）
  - `bb-Boiling-output-{extension}.txt`: 追踪结果（帧ID,追踪ID,命中次数,x1,y1,x2,y2）
  - `vapor_{extension}.txt`: 每帧的蒸汽区域像素数
  - `vaporBase_bt-{extension}.txt`: 每帧的基础蒸汽区域像素数
  - `bubble_size_bt-{extension}.txt`: 每帧每个气泡的大小
  - `bubind_{extension}.txt`: 每个追踪目标在各帧中的边界框索引
  - `frames_{extension}.txt`: 每个追踪目标出现的帧号列表
  - `class_{extension}.txt`: 每帧中每个检测目标的类别
  - `bubclass_{extension}.txt`: 每个追踪目标在各帧中的类别
- **`savefolder/detection_vis/`**（检测可视化图）：`Generate(..., true)` 时每帧保存
  - `detection_vis_{extension}_000001.jpg` 等
- **`savefolder/figures/`**（分析图表）：调用 `Plotvf()`、`Plotbc()`、`PlotInterfaceVelocity(bubble)` 后生成

### 输出图表说明

以下 PNG 图保存在 **`savefolder/figures/`** 下，含义如下。

| 文件名 | 保存位置 | 对应方法 | 图的含义 | 如何解读 |
|--------|----------|----------|----------|----------|
| `vaporfig_{extension}.png` | `figures/` | `Plotvf()` | **蒸汽分数时间序列**：每帧蒸汽区域占整帧像素的比例随时间变化。 | **横轴**：时间（s）。**纵轴**：蒸汽分数（0～1）。灰色折线为原始值，蓝色折线为滑动平均。用于观察整体汽化强度随时间的变化。 |
| `bcfig_{extension}.png` | `figures/` | `Plotbc()` | **气泡数量时间序列**：每帧检测到的气泡个数随时间变化。 | **横轴**：时间（s）。**纵轴**：气泡数量。灰色折线为原始值，蓝色折线为滑动平均。用于观察沸腾强度、成核与合并等。 |
| `velocity_{extension}_{bubble}.png` | `figures/` | `PlotInterfaceVelocity(bubble)` | **指定气泡的界面速度空间-时间图**：该气泡边界上各位置、各时刻的界面法向速度。 | **横轴**：沿气泡轮廓的位置（轮廓重采样为约 200 点，相当于沿边界走一圈）。**纵轴**：时间（帧对顺序）。**颜色**：界面速度（暖色=界面向外长/生长，冷色=界面向内缩/冷凝）。用于分析单个气泡的生长/收缩在空间和时间上的分布。 |

**界面速度图补充说明**：`PlotInterfaceVelocity(bubble)` 针对**追踪 ID 为 `bubble` 的那一个气泡**，在其出现的各帧上提取轮廓，在相邻帧之间用最近邻配对轮廓点，用位移除以时间得到速度；若下一帧匹配点在气泡内部则速度取负（向内）。图中数据经高斯平滑并归一化到固定范围后以伪彩色显示。

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

- `void Generate(float thres = 0.5, bool save_detection_vis = true)`: 执行气泡检测和追踪；`save_detection_vis` 为 true 时每帧保存检测可视化到 `savefolder/detection_vis/`
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
├── LICENSE
│
├── include/                # 头文件
│   ├── bubble_id/
│   │   └── bubble_id.h
│   ├── yolov8_seg/         # YOLOv8-seg 头文件
│   └── ocsort/             # OCSort 追踪器头文件
├── src/                    # 源文件（按功能分目录）
│   ├── bubble_id/          # 分析库（蒸汽分数、气泡数量、界面速度等）
│   │   └── bubble_id.cpp
│   ├── yolov8_seg/         # 检测/分割模型（YOLOv8-seg 提取结果）
│   │   ├── yolov8_seg_onnx.cpp
│   │   └── yolov8_seg_utils.cpp
│   └── ocsort/             # 追踪（辅助 yolov8 多目标追踪）
│       ├── Association.cpp
│       ├── KalmanBoxTracker.cpp
│       ├── KalmanFilter.cpp
│       ├── LapJv.cpp
│       ├── OCSort.cpp
│       └── Utilities.cpp
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
└── third_party/            # 第三方依赖（需自行添加 onnxruntime；matplotplusplus、spdlog 已包含）
    ├── onnxruntime/
    ├── matplotplusplus/
    └── spdlog/
```

### 作为子项目集成

不安装到系统、在您的 CMake 项目中直接引用时：

```cmake
add_subdirectory(path/to/BubbleID-cpp)
target_link_libraries(your_target bubble_id CURL::libcurl)
```

```cpp
#include "bubble_id/bubble_id.h"
```

## 注意事项

1. **编译/链接报错**：若出现 ONNX Runtime、CUDA、curl 或头文件相关错误，请先查看上文 **「常见编译/链接错误与解决」** 表格。

2. **模型路径**: 示例程序已支持通过命令行传入模型路径；若仓库中自带 ONNX 模型，将 `<model_path>` 指向该文件即可（如 `./model.onnx`）。在自行集成代码时，需在构造 `DataAnalysis` 时传入正确的模型路径。

3. **图像格式**: 当前实现仅支持 `.jpg` 格式的图像文件。

4. **视频格式**: 支持 OpenCV 支持的所有视频格式（如 `.avi`, `.mp4` 等）。

5. **绘图依赖**: 如果使用 `Plotvf()` 和 `Plotbc()` 功能，需要系统安装 **gnuplot**（Matplot++ 后端）：`sudo apt-get install gnuplot`。

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
