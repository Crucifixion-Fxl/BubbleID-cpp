#include "bubbleID/bubbleID.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] 
                  << " <images_folder> <video_path> <save_folder> <extension> <model_path> <device>" 
                  << std::endl;
        std::cerr << "Example: " << argv[0]
                  << " ./images ./video.avi ./results test1 ./model.onnx cpu"
                  << std::endl;
        return 1;
    }

    std::string imagesfolder = argv[1];
    std::string videopath = argv[2];
    std::string savefolder = argv[3];
    std::string extension = argv[4];
    std::string modelweights = argv[5];
    std::string device = argv[6];

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
        // std::cout << "绘制蒸汽分数图..." << std::endl;
        // analysis.Plotvf();
        
        // std::cout << "绘制气泡数量图..." << std::endl;
        // analysis.Plotbc();
        
        // std::cout << "分析气泡界面速度（气泡ID=0）..." << std::endl;
        // analysis.PlotInterfaceVelocity(0);

        std::cout << "处理完成！结果已保存到: " << savefolder << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
