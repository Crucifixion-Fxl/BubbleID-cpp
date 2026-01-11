# 安装指南

## 快速开始

### 1. 安装系统依赖

#### Ubuntu/Debian
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

#### macOS
```bash
brew install opencv eigen python3 numpy
```

### 2. 安装 ONNX Runtime

#### 方法一：从源码编译（推荐）
```bash
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release --build_shared_lib --parallel
export ONNXRUNTIME_DIR=$(pwd)
```

#### 方法二：下载预编译版本
1. 访问 [ONNX Runtime 发布页面](https://github.com/microsoft/onnxruntime/releases)
2. 下载适合您系统的预编译版本
3. 解压到 `third_party/onnxruntime` 目录，或设置环境变量：
```bash
export ONNXRUNTIME_DIR=/path/to/onnxruntime
```

### 3. 编译和安装 BubbleID

#### 使用安装脚本（推荐）
```bash
# 设置 ONNX Runtime 路径（如果不在默认位置）
export ONNXRUNTIME_DIR=/path/to/onnxruntime

# 编译
./install.sh

# 编译并安装到系统
./install.sh /usr/local install
```

#### 手动编译
```bash
mkdir build && cd build

# 配置（设置 ONNXRUNTIME_DIR 如果不在默认位置）
cmake .. \
    -DONNXRUNTIME_DIR=/path/to/onnxruntime \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_BUILD_TYPE=Release

# 编译
make -j$(nproc)

# 安装（可选）
sudo make install
```

### 4. 验证安装

编译示例程序：
```bash
cd build
./bin/bubbleID_example \
    ./images \
    ./video.avi \
    ./results \
    test1 \
    ./model.onnx \
    cpu
```

## 常见问题

### 找不到 OpenCV
```bash
# Ubuntu/Debian
sudo apt-get install libopencv-dev

# macOS
brew install opencv
```

### 找不到 Eigen3
```bash
# Ubuntu/Debian
sudo apt-get install libeigen3-dev

# macOS
brew install eigen
```

### ONNX Runtime 路径问题
确保设置了正确的 `ONNXRUNTIME_DIR` 环境变量，或通过 CMake 参数指定：
```bash
cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime
```

### Python 依赖问题
如果使用可视化功能（Plotvf, Plotbc），需要安装 Python matplotlib：
```bash
pip3 install matplotlib numpy
```

## 卸载

如果使用 `make install` 安装，可以通过以下方式卸载：
```bash
cd build
sudo xargs rm < install_manifest.txt
```

## 开发模式

如果要在开发模式下使用（不安装到系统）：
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

然后在您的项目中使用：
```cmake
add_subdirectory(path/to/bubbleID-package)
target_link_libraries(your_target bubbleID)
```
