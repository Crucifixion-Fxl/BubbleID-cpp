# 数据目录说明

本目录用于存放**输入数据**，供示例程序 `bubbleID_example` 使用。

## 推荐结构

```
data/
├── Images-120W/     # 图像文件夹：按帧导出的 JPG 序列（与视频对应）
│   └── *.jpg
└── 120W.avi         # 视频文件：与图像序列对应的同一段视频
```

- **图像文件夹**：子目录名任意（如 `Images-120W`），内放与视频逐帧对应的 JPG 图像，用于 YOLOv8-seg 气泡检测。
- **视频路径**：与图像序列同源的视频文件（如 `120W.avi`），用于 OCSort 追踪时的帧顺序对齐。

## 运行示例

```bash
./build/bin/bubbleID_example \
  /path/to/BubbleID-cpp/data/Images-120W \
  /path/to/BubbleID-cpp/data/120W.avi \
  /path/to/BubbleID-cpp/result \
  ori \
  /path/to/BubbleID-cpp/weights/bubble_seg.onnx \
  cpu
```

若数据较大，可将 `data/` 加入 `.gitignore`，仅本地保留。
