# BubbleID-cpp — C++ Core for Bubble Detection and Tracking

**English** | [中文 (Chinese)](README.zh-CN.md)

> **Note:** This is the **C++ core** of the [BubbleID](https://github.com/cldunlap73/BubbleID.git) project. The original project is a Python-based framework for bubble detection and tracking in pool boiling imagery. This repository reimplements the core algorithms in C++ for better performance and easier integration.

## About the Original Project

[BubbleID](https://github.com/cldunlap73/BubbleID.git) is a deep learning framework for analyzing pool boiling images, from the paper **"BubbleID: A deep learning framework for bubble interface dynamics analysis"**. It combines tracking, segmentation, and classification models for off-body classification, interface velocity prediction, and bubble statistics. The original project is built on ocsort and detectron2.

## What This Project Is

BubbleID-cpp is a **C++ implementation of the core algorithms** of the Python BubbleID project:

- **Performance:** C++ for faster execution.
- **Integration:** Use as a C++ library in other projects.
- **Core features:** Bubble detection, tracking, and data analysis as in the original.

The library uses a **YOLOv8-seg** instance segmentation model and **OCSort** for multi-object tracking to detect, track, and analyze bubbles in image sequences or video.

## Features

- **Bubble detection:** YOLOv8-seg for instance segmentation (positions and masks).
- **Multi-object tracking:** OCSort to track bubbles across frames.
- **Data analysis:**
  - Vapor and base vapor region statistics
  - Bubble size statistics
  - Bubble ID index files
  - Plots: vapor fraction, bubble count, interface velocity

## System Requirements (Linux)

The following is for **Linux (Ubuntu/Debian)**.

- **OS:** Linux
- **Compiler:** GCC 7+ (C++17)
- **CMake:** 3.16+
- **Libraries:** OpenCV 4.x, Eigen3, ONNX Runtime, libcurl, spdlog (logging), gnuplot (for Plotvf/Plotbc)

## Dependencies

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

> **Note:** `libcurl4-openssl-dev` is needed for OpenCV’s dependency chain (e.g. libgdal/libnetcdf). Without it you may get `undefined reference to 'curl_*@CURL_OPENSSL_4'` when linking the example.

### ONNX Runtime

- Download a **Linux x64** build from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) and extract it under `third_party/onnxruntime` so that you have:
  - `third_party/onnxruntime/include/onnxruntime_cxx_api.h`
  - `third_party/onnxruntime/lib/*.so`
- If you use another path, set: `-DONNXRUNTIME_DIR=/path/to/onnxruntime`
- **CPU vs CUDA:** For a **CPU-only** ONNX Runtime build (no CUDA Execution Provider), use `-DUSE_ORT_CUDA=OFF` (default). Otherwise you may get `undefined reference to 'OrtSessionOptionsAppendExecutionProvider_CUDA'`. For a CUDA-enabled ORT build, use `-DUSE_ORT_CUDA=ON`.

### third_party Layout

```
third_party/
├── onnxruntime/      # Required: ONNX Runtime (include/, lib/)
├── matplotplusplus/  # Bundled: Plotvf/Plotbc (requires gnuplot)
└── spdlog/           # Bundled: logging
```

## Build and Install

```bash
mkdir build && cd build

# Configure (omit -DONNXRUNTIME_DIR if using third_party/onnxruntime)
cmake .. \
    -DONNXRUNTIME_DIR=/path/to/onnxruntime \
    -DUSE_ORT_CUDA=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local

# Build
cmake --build . -j$(nproc)

# Optional: install
sudo cmake --install .
```

**CMake options:**

| Option | Default | Description |
|--------|---------|-------------|
| `ONNXRUNTIME_DIR` | `third_party/onnxruntime` | ONNX Runtime root (include/, lib/) |
| `USE_ORT_CUDA` | `OFF` | Use OFF for CPU-only ORT; ON for CUDA ORT |
| `BUILD_EXAMPLES` | `ON` | Build the `bubbleID_example` executable |

## Common Build / Link Errors

| Symptom | Cause | Fix |
|--------|--------|-----|
| `onnxruntime_cxx_api.h: No such file or directory` | ONNX Runtime not found | Extract ORT to `third_party/onnxruntime` or set `-DONNXRUNTIME_DIR=/path/to/onnxruntime` |
| `undefined reference to 'OrtSessionOptionsAppendExecutionProvider_CUDA'` | CPU-only ORT but CUDA code linked | Use `-DUSE_ORT_CUDA=OFF` or switch to a CUDA ORT build and set `-DUSE_ORT_CUDA=ON` |
| `undefined reference to 'curl_*@CURL_OPENSSL_4'` (from libgdal/libnetcdf) | libcurl missing for OpenCV chain | Install `libcurl4-openssl-dev`; project already links `CURL::libcurl` |
| `Association.hpp` or other ocsort headers not found | Include path / style | Use `#include "ocsort/xxx.hpp"`; CMake already adds `include/` |
| `libstdc++.so.6: version 'GLIBCXX_3.4.30' not found` (e.g. under conda) | Older libstdc++ from conda | Run with system libs first: `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH ./build/bin/bubbleID_example`, or `conda deactivate` |
| `gnuplot: not found` when calling Plotvf/Plotbc | gnuplot not installed | `sudo apt-get install gnuplot` |

To reconfigure from scratch:

```bash
rm -rf build && mkdir build && cd build
cmake .. -DUSE_ORT_CUDA=OFF
cmake --build . -j$(nproc)
```

### Verify and Run

From the project root (with image dir, video, and ONNX model ready):

```bash
./build/bin/bubbleID_example <images_dir> <video_path> <output_dir> <extension> <model.onnx> cpu
```

Output is written under the given output directory in: `data/` (txt), `detection_vis/` (detection images), `figures/` (plots). See “Output files” below.

### Uninstall (if you ran install)

```bash
cd build
sudo xargs rm -f < install_manifest.txt
```

## Usage

### Example Program

Run the example with your image folder, video, output directory, extension tag, model path, and device:

```bash
./build/bin/bubbleID_example <images_dir> <video_path> <output_dir> <extension> <model.onnx> <cpu|gpu>
```

Example:

```bash
./build/bin/bubbleID_example ./images ./video.avi ./result test1 ./model.onnx cpu
```

### Basic Use (as a library)

```cpp
#include "bubble_id/bubble_id.h"

int main() {
    std::string imagesfolder = "./images";
    std::string videopath = "./video.avi";
    std::string savefolder = "./result";   // will contain data/, detection_vis/, figures/
    std::string extension = "test1";
    std::string modelweights = "./model.onnx";
    std::string device = "cpu";

    DataAnalysis analysis(
        imagesfolder,
        videopath,
        savefolder,
        extension,
        modelweights,
        device
    );

    // Run detection and tracking (threshold 0.5; true = save per-frame detection images to savefolder/detection_vis/)
    analysis.Generate(0.5, true);

    // Generate and save plots (file-only, no GUI)
    analysis.Plotvf();   // vapor fraction -> figures/vaporfig_*.png
    analysis.Plotbc();    // bubble count -> figures/bcfig_*.png
    analysis.PlotInterfaceVelocity(0);  // interface velocity -> figures/velocity_*_0.png

    return 0;
}
```

### Output Files (by category)

All results go under `savefolder` in subfolders:

- **`savefolder/data/`** (txt, from `Generate()`):
  - `bb-Boiling-{extension}.txt`: raw detections (frame_id x1 y1 x2 y2 conf class)
  - `bb-Boiling-output-{extension}.txt`: tracking (frame_id,track_id,hits,x1,y1,x2,y2)
  - `vapor_{extension}.txt`, `vaporBase_bt-{extension}.txt`, `bubble_size_bt-{extension}.txt`, `bubind_*.txt`, `frames_*.txt`, `class_*.txt`, `bubclass_*.txt`
- **`savefolder/detection_vis/`**: per-frame detection images when `Generate(..., true)` (e.g. `detection_vis_{extension}_000001.jpg`)
- **`savefolder/figures/`**: plots from `Plotvf()`, `Plotbc()`, `PlotInterfaceVelocity(bubble)` (PNG)

### Output Plots (in `savefolder/figures/`)

| File | Method | Description |
|------|--------|-------------|
| `vaporfig_{extension}.png` | `Plotvf()` | Vapor fraction vs time (raw + rolling average). |
| `bcfig_{extension}.png` | `Plotbc()` | Bubble count vs time (raw + rolling average). |
| `velocity_{extension}_{bubble}.png` | `PlotInterfaceVelocity(bubble)` | Interface velocity space–time for one tracked bubble (position along contour vs time; color = velocity). |

`PlotInterfaceVelocity(bubble)` uses the bubble with that track ID: contour points are matched between consecutive frames, velocity from displacement/time; data are smoothed and shown as pseudocolor.

## API Summary

### DataAnalysis

**Constructor:**

```cpp
DataAnalysis(
    const std::string& imagesfolder,
    const std::string& videopath,
    const std::string& savefolder,
    const std::string& extension,
    const std::string& modelweightsloc,
    const std::string& device   // "cpu" or "gpu"
);
```

**Main methods:**

- `void Generate(float thres = 0.5, bool save_detection_vis = true)` — run detection and tracking; when `save_detection_vis` is true, save per-frame images to `savefolder/detection_vis/`
- `void Plotvf()` — vapor fraction plot
- `void Plotbc()` — bubble count plot
- `void PlotInterfaceVelocity(int bubble)` — interface velocity plot for one bubble
- `void saveDataToFiles()` — write data files

**Static helpers:** `get_color`, `restoreMaskToOriginalSize`, `iouBatch`, `getImagePaths`, etc.

## Directory Layout

```
BubbleID-cpp/
├── CMakeLists.txt
├── README.md           # This file (English)
├── README.zh-CN.md     # Chinese
├── LICENSE
├── include/
│   ├── bubble_id/
│   ├── yolov8_seg/
│   └── ocsort/
├── src/
│   ├── bubble_id/
│   ├── yolov8_seg/
│   └── ocsort/
├── examples/
├── data/
├── weights/
├── result/             # Output (only .gitkeep committed; contents in .gitignore)
│   ├── data/
│   ├── detection_vis/
│   └── figures/
├── build/
└── third_party/
    ├── onnxruntime/
    ├── matplotplusplus/
    └── spdlog/
```

### Use as a subproject

```cmake
add_subdirectory(path/to/BubbleID-cpp)
target_link_libraries(your_target bubble_id CURL::libcurl)
```

```cpp
#include "bubble_id/bubble_id.h"
```

## Notes

1. **Build issues:** See the “Common build / link errors” table above.
2. **Model path:** Pass the ONNX model path on the command line or in the `DataAnalysis` constructor.
3. **Images:** Only `.jpg` is supported for input frames.
4. **Video:** Any format supported by OpenCV (e.g. `.avi`, `.mp4`).
5. **Plotting:** `Plotvf()` and `Plotbc()` need **gnuplot** (`sudo apt-get install gnuplot`). They run in quiet mode (save to file only, no GUI). If a window still appears, try `export GNUTERM=dumb` before running.
6. **Logging:** Output uses **spdlog**; you can set level and sinks as needed.
7. **Performance:** For long videos, consider GPU acceleration.

## Related

- **Original Python project:** [BubbleID](https://github.com/cldunlap73/BubbleID.git) — full framework with GUI and more.
- **Paper:** BubbleID: A deep learning framework for bubble interface dynamics analysis.

## License

Same as the original project (Apache-2.0 and MIT).

## Contributing

Issues and Pull Requests are welcome.

---

**Acknowledgments:** This project is based on [cldunlap73/BubbleID](https://github.com/cldunlap73/BubbleID.git).
