#include "bubbleID/bubbleID.h"
#include <iostream>

// 调试用默认参数（无命令行参数时使用）
static const std::string DEFAULT_IMAGES_FOLDER = "/root/workspace/BubbleID-cpp/data/Images-120W";
static const std::string DEFAULT_VIDEO_PATH    = "/root/workspace/BubbleID-cpp/data/120W.avi";
static const std::string DEFAULT_SAVE_FOLDER   = "/root/workspace/BubbleID-cpp/result";
static const std::string DEFAULT_EXTENSION    = "ori";
static const std::string DEFAULT_MODEL_PATH   = "/root/workspace/BubbleID-cpp/weights/bubble_seg.onnx";
static const std::string DEFAULT_DEVICE       = "cpu";

int main(int argc, char* argv[]) {
    std::string imagesfolder, videopath, savefolder, extension, modelweights, device;

    if (argc >= 7) {
        imagesfolder = argv[1];
        videopath    = argv[2];
        savefolder   = argv[3];
        extension    = argv[4];
        modelweights = argv[5];
        device       = argv[6];
    } else {
        std::cerr << "Usage: " << argv[0]
                  << " <images_folder> <video_path> <save_folder> <extension> <model_path> <device>"
                  << std::endl;
        std::cerr << "No args: using default paths (for debugging)." << std::endl;
        imagesfolder = DEFAULT_IMAGES_FOLDER;
        videopath    = DEFAULT_VIDEO_PATH;
        savefolder   = DEFAULT_SAVE_FOLDER;
        extension    = DEFAULT_EXTENSION;
        modelweights = DEFAULT_MODEL_PATH;
        device       = DEFAULT_DEVICE;
    }

    std::cout << "=== BubbleID 示例程序 ===" << std::endl;
    std::cout << "图像文件夹: " << imagesfolder << std::endl;
    std::cout << "视频路径: " << videopath << std::endl;
    std::cout << "保存文件夹: " << savefolder << std::endl;
    std::cout << "扩展名: " << extension << std::endl;
    std::cout << "模型路径: " << modelweights << std::endl;
    std::cout << "设备: " << device << std::endl;
    std::cout << std::endl;

    try {
        // 创建分析对象
        DataAnalysis analysis(
            imagesfolder,
            videopath,
            savefolder,
            extension,
            modelweights,
            device
        );

        std::cout << "开始执行气泡检测和追踪..." << std::endl;
        // 执行气泡检测和追踪（置信度阈值 0.5）
        analysis.Generate(0.5);
        std::cout << "气泡检测和追踪完成！" << std::endl;

        // 可选：绘制可视化图表
        std::cout << "绘制蒸汽分数图..." << std::endl;
        analysis.Plotvf();
        
        std::cout << "绘制气泡数量图..." << std::endl;
        analysis.Plotbc();
        
        std::cout << "分析气泡界面速度（气泡ID=0）..." << std::endl;
        analysis.PlotInterfaceVelocity(0);

        std::cout << "处理完成！结果已保存到: " << savefolder << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
