#ifndef BUBBLEID_H
#define BUBBLEID_H
#include <string>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <sys/types.h>
#include <sstream>
#include <iostream>
#include <vector>
#include <Eigen/Dense>

class DataAnalysis{

public:
    // 构造函数
    DataAnalysis(const std::string& imagesfolder,
                 const std::string& videopath,
                 const std::string& savefolder,
                 const std::string& extension,
                 const std::string& modelweightsloc,
                 const std::string& device);

    void Generate(float thres = 0.5);
    void Plotvf();
    void Plotbc();
    void PlotInterfaceVelocity(int bubble);
    void saveDataToFiles();
    static std::vector<int> get_color(int number);
    static int findMaxNumber(std::vector<std::vector<int>>& data);
    static cv::Mat restoreMaskToOriginalSize(const cv::Rect& bbox, const cv::Mat& boxMask, int imgHeight, int imgWidth);
    static void save2DVectorToFile(const std::string& filename, const std::vector<std::vector<int>>& data);
    static bool directoryExists(const std::string& path);
    static bool makeDirectories(const std::string& path);
    static bool hasJpgExtension(const std::string& filename);
    static std::vector<int> argmax(const std::vector<std::vector<double>>& matrix);
    static void print2DVector(const std::vector<std::vector<int>>& vec);
    static std::vector<std::vector<double>> iouBatch(const std::vector<std::vector<int>>& bboxes1, const std::vector<std::vector<int>>& bboxes2);
    static void getImagePathsRecursive(const std::string& dir_path, std::vector<std::string>& out_paths);
    static std::vector<std::string> getImagePaths(const std::string& root_dir);
    static Eigen::Matrix<float, Eigen::Dynamic, 6> Vector2Matrix(std::vector<std::vector<float>> data);
    static std::vector<std::vector<int>> load2DVectorFromFile(const std::string& filename);
private:
    // 成员变量
    std::string imagesfolder_;
    std::string videopath_;
    std::string savefolder_;
    std::string extension_;
    std::string modelweightsloc_;
    std::string device_;

    // 数据存储成员变量
    std::vector<int> vapor_;
    std::vector<int> vapor_base_;
    std::vector<std::vector<int>> bubble_size_;
    std::vector<std::vector<float>> bounding_box_;
};
#endif
