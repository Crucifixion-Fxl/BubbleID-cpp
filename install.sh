#!/bin/bash

# BubbleID 安装脚本

set -e

echo "=== BubbleID 安装脚本 ==="
echo ""

# 检查 CMake
if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到 CMake，请先安装 CMake 3.16 或更高版本"
    exit 1
fi

# 创建构建目录
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "清理旧的构建目录..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 配置 CMake
echo "配置 CMake..."
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"

# 检查是否提供了 ONNXRUNTIME_DIR
if [ -n "$ONNXRUNTIME_DIR" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DONNXRUNTIME_DIR=$ONNXRUNTIME_DIR"
    echo "使用 ONNXRUNTIME_DIR: $ONNXRUNTIME_DIR"
else
    echo "警告: 未设置 ONNXRUNTIME_DIR，请确保 ONNX Runtime 已正确安装"
fi

# 检查是否提供了安装前缀
if [ -n "$1" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=$1"
    echo "安装前缀: $1"
else
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=/usr/local"
    echo "使用默认安装前缀: /usr/local"
fi

cmake .. $CMAKE_ARGS

# 编译
echo ""
echo "开始编译..."
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
make -j$CORES

# 安装（可选）
if [ "$2" == "install" ]; then
    echo ""
    echo "开始安装..."
    sudo make install
    echo "安装完成！"
else
    echo ""
    echo "编译完成！"
    echo "要安装库，请运行: sudo make install"
    echo "或者重新运行此脚本并添加 'install' 参数: $0 [prefix] install"
fi

echo ""
echo "=== 完成 ==="
