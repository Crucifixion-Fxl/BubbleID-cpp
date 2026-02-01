#include "bubble_id/bubble_id.h"
#include <spdlog/spdlog.h>

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
        spdlog::warn("Usage: {} <images_folder> <video_path> <save_folder> <extension> <model_path> <device>", argv[0]);
        spdlog::warn("No args: using default paths (for debugging).");
        imagesfolder = DEFAULT_IMAGES_FOLDER;
        videopath    = DEFAULT_VIDEO_PATH;
        savefolder   = DEFAULT_SAVE_FOLDER;
        extension    = DEFAULT_EXTENSION;
        modelweights = DEFAULT_MODEL_PATH;
        device       = DEFAULT_DEVICE;
    }

    spdlog::info("=== BubbleID 示例程序 ===");
    spdlog::info("图像文件夹: {}", imagesfolder);
    spdlog::info("视频路径: {}", videopath);
    spdlog::info("保存文件夹: {}", savefolder);
    spdlog::info("扩展名: {}", extension);
    spdlog::info("模型路径: {}", modelweights);
    spdlog::info("设备: {}", device);

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

        spdlog::info("开始执行气泡检测和追踪...");
        // 执行气泡检测和追踪（置信度阈值 0.5）
        analysis.Generate(0.5);
        spdlog::info("气泡检测和追踪完成！");

        spdlog::info("绘制蒸汽分数图...");
        analysis.Plotvf();
        
        spdlog::info("绘制气泡数量图...");
        analysis.Plotbc();
        
        spdlog::info("分析气泡界面速度（气泡ID=0）...");
        analysis.PlotInterfaceVelocity(0);

        spdlog::info("处理完成！结果已保存到: {}", savefolder);
    }
    catch (const std::exception& e) {
        spdlog::error("错误: {}", e.what());
        return 1;
    }

    return 0;
}
