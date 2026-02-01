#include "bubble_id/bubble_id.h"
#include <spdlog/spdlog.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <map>
#include "yolov8_seg/yolov8_seg_onnx.h"
#include "yolov8_seg/yolov8_seg_utils.h"
#include "ocsort/OCSort.hpp"
#include <Eigen/Dense>
#include <matplot/matplot.h>
#include <opencv2/opencv.hpp>

std::vector<std::vector<int>> DataAnalysis::load2DVectorFromFile(const std::string& filename) {
    std::vector<std::vector<int>> data;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        spdlog::error("无法打开文件: {}", filename);
        return data;
    }
    std::string line;
    while (std::getline(infile, line)) {
        std::vector<int> row;
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) {
                row.push_back(std::stoi(item));
            }
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    infile.close();
    return data;
}

DataAnalysis::DataAnalysis(
                 const std::string& imagesfolder,
                 const std::string& videopath,
                 const std::string& savefolder,
                 const std::string& extension,
                 const std::string& modelweightsloc,
                 const std::string& device)
    : imagesfolder_(imagesfolder),
      videopath_(videopath),
      savefolder_(savefolder),
      extension_(extension),
      modelweightsloc_(modelweightsloc),
      device_(device){}



std::vector<int> DataAnalysis::get_color(int number) {
    // 将整数转换为颜色
    int hue = (number * 30) % 180;
    int saturation = (number * 103) % 256;
    int value = (number * 50) % 256;

    // 将HSV转换为RGB
    float h = hue / 179.0f;
    float s = saturation / 255.0f;
    float v = value / 255.0f;

    float r, g, b;

    int i = static_cast<int>(h * 6);
    float f = h * 6 - i;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);

    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }

    return {static_cast<int>(r * 255), static_cast<int>(g * 255), static_cast<int>(b * 255)};
}

Eigen::Matrix<float, Eigen::Dynamic, 6> DataAnalysis::Vector2Matrix(std::vector<std::vector<float>> data) {
    // Create an Eigen::Matrix with the same number of rows as the data and 6 columns
    Eigen::Matrix<float, Eigen::Dynamic, 6> matrix(data.size(), data[0].size());

    // Iterate over the rows and columns of the data vector
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[0].size(); ++j) {
            // Assign the value at position (i, j) of the matrix to the corresponding value from the data vector
            matrix(i, j) = data[i][j];
        }
    }

    // Return the resulting matrix
    return matrix;
}

// 计算两个边界框集合之间的IoU
std::vector<std::vector<double>> DataAnalysis::iouBatch(const std::vector<std::vector<int>>& bboxes1, const std::vector<std::vector<int>>& bboxes2) {
    std::vector<std::vector<double>> iouMatrix(bboxes1.size(), std::vector<double>(bboxes2.size(), 0.0));

    for (size_t i = 0; i < bboxes1.size(); ++i) {
        for (size_t j = 0; j < bboxes2.size(); ++j) {
            double xx1 = std::max(bboxes1[i][0], bboxes2[j][0]);
            double yy1 = std::max(bboxes1[i][1], bboxes2[j][1]);
            double xx2 = std::min(bboxes1[i][2], bboxes2[j][2]);
            double yy2 = std::min(bboxes1[i][3], bboxes2[j][3]);

            double w = std::max(0.0, xx2 - xx1);
            double h = std::max(0.0, yy2 - yy1);
            double intersection = w * h;

            double area1 = (bboxes1[i][2] - bboxes1[i][0]) * (bboxes1[i][3] - bboxes1[i][1]);
            double area2 = (bboxes2[j][2] - bboxes2[j][0]) * (bboxes2[j][3] - bboxes2[j][1]);

            double unionArea = area1 + area2 - intersection;

            iouMatrix[i][j] = intersection / unionArea;
        }
    }

    return iouMatrix;
}

// 找到每行中最大值的索引
std::vector<int> DataAnalysis::argmax(const std::vector<std::vector<double>>& matrix) {
    std::vector<int> indices(matrix.size(), 0);
    for (size_t i = 0; i < matrix.size(); ++i) {
        indices[i] = std::distance(matrix[i].begin(), std::max_element(matrix[i].begin(), matrix[i].end()));
    }
    return indices;
}

// 从bboxmask 恢复到 原图的mask
cv::Mat DataAnalysis::restoreMaskToOriginalSize(const cv::Rect& bbox, const cv::Mat& boxMask, int imgHeight, int imgWidth) {
    // 创建一个与原图相同尺寸的空掩码
    cv::Mat fullSizeMask = cv::Mat::zeros(imgHeight, imgWidth, CV_8UC1);

    // 获取边界框的位置和尺寸
    int x1 = bbox.x;
    int y1 = bbox.y;
    int width = bbox.width;
    int height = bbox.height;

    // 确保边界框在图像范围内
    if (x1 >= 0 && y1 >= 0 && x1 + width <= imgWidth && y1 + height <= imgHeight) {
        // 将boxMask放置到空掩码的正确位置
        cv::Mat roi = fullSizeMask(cv::Rect(x1, y1, width, height));
        boxMask.copyTo(roi);
    }

    return fullSizeMask;
}


void DataAnalysis::Generate(float thres, bool save_detection_vis){
    std::string directory_path = this->imagesfolder_;
    std::string video_file = this->videopath_;

    // 分类保存：txt 在 data/，检测可视化在 detection_vis/
    const std::string data_dir = savefolder_ + "/data";
    const std::string file_path = data_dir + "/bb-Boiling-" + extension_ + ".txt";
    const std::string output_file_path = data_dir + "/bb-Boiling-output-" + extension_ + ".txt";
    const std::string vapor_file = data_dir + "/vapor_" + extension_ + ".txt";
    const std::string vapor_base_file = data_dir + "/vaporBase_bt-" + extension_ + ".txt";
    const std::string bubble_size_file = data_dir + "/bubble_size_bt-" + extension_ + ".txt";
    const std::string bubind_file = data_dir + "/bubind_" + extension_ + ".txt";
    const std::string frameind_file = data_dir + "/frames_" + extension_ + ".txt";
    const std::string classind_file = data_dir + "/class_" + extension_ + ".txt";
    const std::string bubclassind_file = data_dir + "/bubclass_" + extension_ + ".txt";

    spdlog::info("{}", this->savefolder_);

    if(!directoryExists(this->savefolder_)){
        makeDirectories(this->savefolder_);
    }
    if (!directoryExists(data_dir))
        makeDirectories(data_dir);
    if (save_detection_vis) {
        std::string vis_sub = savefolder_ + "/detection_vis";
        if (!directoryExists(vis_sub))
            makeDirectories(vis_sub);
    }

    // 开始加载模型YOLOv8-seg
    spdlog::info("加载模型开始");
    std::string model_path = this->modelweightsloc_;
    std::unique_ptr<Yolov8SegOnnx> infer_engine=std::make_unique<Yolov8SegOnnx>();

    if(!infer_engine->ReadModel(model_path,true,0,true)){
        spdlog::error("加载模型失败");
        return;
    }

    spdlog::info("加载模型完成");
    
    // 设置置信度
    bool ok;
    float initial_threshold = 0.5;
    if(ok && initial_threshold>=0.0f && initial_threshold<=1.0f){
        infer_engine->SetClassThreshold(initial_threshold);
    }

    // 获取文件路径
    std::vector<std::string> image_paths = getImagePaths(directory_path);
    spdlog::info("文件获取完毕");
    spdlog::info("Run instance segmentation model and save data");

    // 初始化数据存储
    std::vector<std::vector<float>> Bounding_Box;  // 存储边界框数据
    std::vector<std::vector<int>> bubble_size;     // 存储每一帧中每一个气泡的占领的mask大小
    std::vector<int> vapor;                        // 存储蒸汽区域
    std::vector<int> vapor_base;                   // 存储基础蒸汽区域

    // 处理每一帧图像
    for (size_t i = 0; i < image_paths.size(); ++i) {
        cv::Mat new_im = cv::imread(image_paths[i]);
        std::vector<OutputSeg> output_seg;
        bool find = infer_engine->OnnxDetect(new_im, output_seg);
        if (save_detection_vis && find && !output_seg.empty()) {
            static std::vector<cv::Scalar> color;
            if (color.empty()) {
                srand(static_cast<unsigned>(time(0)));
                color.push_back(cv::Scalar(255, 255, 255));
                color.push_back(cv::Scalar(255, 255, 255));
            }
            cv::Mat vis_im = new_im.clone();
            DrawPred(vis_im, output_seg, infer_engine->_className, color);
            std::ostringstream oss;
            oss << savefolder_ << "/detection_vis/detection_vis_" << extension_ << "_" << std::setfill('0') << std::setw(6) << (i + 1) << ".jpg";
            cv::imwrite(oss.str(), vis_im);
        }
        if (find && !output_seg.empty()) {
            std::vector<std::vector<float>> converted_bounding_box;
            std::vector<cv::Mat> valid_masks;
            std::vector<float> valid_scores;
            std::vector<int> valid_classes;

            // 获取图像尺寸
            int img_height = new_im.rows;
            int img_width = new_im.cols;

            // 处理每个检测结果
            for (const auto& seg : output_seg) {
                float x1 = seg.box.x;
                float y1 = seg.box.y;
                float x2 = seg.box.x + seg.box.width;
                float y2 = seg.box.y + seg.box.height;

                // 应用过滤条件
                if (y2 > 0 || (y2 > 502 && y2 < 533 && x2 > 320 && x2 < 515)) {
                    converted_bounding_box.push_back({x1, y1, x2, y2});
                    
                    cv::Mat restored_mask = restoreMaskToOriginalSize(seg.box, seg.boxMask, img_height, img_width);
                    
                    valid_masks.push_back(restored_mask);
                    valid_scores.push_back(seg.confidence);
                    valid_classes.push_back(seg.id);
                }
            }

            if (!converted_bounding_box.empty()) {
                // 保存边界框数据
                for (size_t j = 0; j < converted_bounding_box.size(); ++j) {
                    std::vector<float> box_data = {
                        static_cast<float>(i + 1),  // 帧索引
                        converted_bounding_box[j][0],  // x1
                        converted_bounding_box[j][1],  // y1
                        converted_bounding_box[j][2],  // x2
                        converted_bounding_box[j][3],  // y2
                        valid_scores[j],  // 置信度
                        static_cast<float>(valid_classes[j])  // 类别ID
                    };
                    Bounding_Box.push_back(box_data);
                }

                // 计算蒸汽区域
                cv::Mat combined_mask = cv::Mat::zeros(img_height, img_width, CV_8UC1);
                for (const auto& mask : valid_masks) {
                    // 确保掩码是单通道的
                    cv::Mat single_channel_mask;
                    if (mask.channels() > 1) {
                        cv::cvtColor(mask, single_channel_mask, cv::COLOR_BGR2GRAY);
                    } else {
                        single_channel_mask = mask.clone();
                    }
                    // 确保掩码是二值的
                    cv::bitwise_or(combined_mask, single_channel_mask, combined_mask);
                }
                vapor.push_back(cv::countNonZero(combined_mask));

                // 计算基础蒸汽区域（类别为0的掩码）
                cv::Mat base_combined_mask = cv::Mat::zeros(img_height, img_width, CV_8UC1);
                std::vector<int> frame_bubble_sizes;
                for (size_t j = 0; j < valid_masks.size(); ++j) {
                    if (valid_classes[j] == 0) {
                        // 确保掩码是单通道的
                        cv::Mat single_channel_mask;
                        if (valid_masks[j].channels() > 1) {
                            cv::cvtColor(valid_masks[j], single_channel_mask, cv::COLOR_BGR2GRAY);
                        } else {
                            single_channel_mask = valid_masks[j].clone();
                        }
                        cv::bitwise_or(base_combined_mask, single_channel_mask, base_combined_mask);
                    }
                    frame_bubble_sizes.push_back(cv::countNonZero(valid_masks[j]));
                }
                vapor_base.push_back(cv::countNonZero(base_combined_mask));
                bubble_size.push_back(frame_bubble_sizes);
            }
        } else {
            spdlog::debug("No detections in frame {}", i);
        }
    }

    // 保存数据到文件
    // 保存边界框数据
    std::ofstream bb_file(file_path.c_str());  // 使用 c_str() 转换字符串
    if (!bb_file.is_open()) {
        spdlog::error("无法打开文件: {}", file_path);
        return;
    }
    for (const auto& box : Bounding_Box) {
        for (const auto& val : box) {
            bb_file << val << " ";
        }
        bb_file << std::endl;
    }
    bb_file.close();

    // 保存蒸汽数据
    std::ofstream vapor_out(vapor_file.c_str());
    if (!vapor_out.is_open()) {
        spdlog::error("无法打开文件: {}", vapor_file);
        return;
    }
    for (const auto& v : vapor) {
        vapor_out << v << std::endl;
    }
    vapor_out.close();

    // 保存基础蒸汽数据
    std::ofstream vapor_base_out(vapor_base_file.c_str());
    if (!vapor_base_out.is_open()) {
        spdlog::error("无法打开文件: {}", vapor_base_file);
        return;
    }
    for (const auto& v : vapor_base) {
        vapor_base_out << v << std::endl;
    }
    vapor_base_out.close();

    // 保存气泡大小数据
    std::ofstream bubble_size_out(bubble_size_file.c_str());
    if (!bubble_size_out.is_open()) {
        spdlog::error("无法打开文件: {}", bubble_size_file);
        return;
    }
    for (const auto& sizes : bubble_size) {
        for (const auto& size : sizes) {
            bubble_size_out << size << " ";
        }
        bubble_size_out << std::endl;
    }
    bubble_size_out.close();

    spdlog::info("数据处理完成，已保存到文件");

    spdlog::info("Perform ocsort tracking on saved data");

    ocsort::OCSort tracker(0.5, 10,20);

    cv::Size img_info(832,600);
    cv::Size img_size(832,600);

    cv::VideoCapture cap(video_file);
    if (!cap.isOpened()) {
        spdlog::error("Error opening video file");
        return;
    }

    int i = 0;

    std::map<int,std::vector<std::vector<float>>> frame_data;

    std::ifstream bb_file_1(file_path.c_str());
    if (!bb_file_1.is_open()) {
        spdlog::error("无法打开文件: {}", file_path);
        return;
    }

    std::string line;
    while (std::getline(bb_file_1, line)) {
        std::istringstream iss(line);
        std::string part;
        std::vector<float> data;
        int frame_id;

        // 读取帧 ID
        if (iss >> frame_id) {
            // 读取剩余的数据，排除最后一个元素
            float value;
            while (iss >> value) {
                data.push_back(value);
            }
        }

        frame_data[frame_id].push_back(data);
    }

    bb_file_1.close();

    std::ofstream bb_file_2(output_file_path.c_str());

    while (cap.isOpened()) {
        cv::Mat frame;
        bool ret = cap.read(frame);
        if(ret==true){
            std::vector<std::vector<float>> xyxyc = frame_data.at(i+1);
            i++;
            if (xyxyc.empty()) {
                xyxyc.resize(0, std::vector<float>(6, 0));
            }
            tracker.update(Vector2Matrix(xyxyc));
            for (const auto& track : tracker.trackers) {
                int track_id = track.id;
                int hits = track.hits;   
                std::vector<int> color = get_color(track_id * 15); // color 颜色有问题
                auto state = track.get_state();
                int x1 = state[0];
                int y1 = state[1];
                int x2 = state[2];
                int y2 = state[3];
                bb_file_2 << i << "," << track_id << "," << hits << "," << x1 << "," << y1 << "," << x2 << "," << y2 << std::endl;
            } 
        }else{
            break;
        }
    }

    bb_file_2.close();
    cap.release();

    spdlog::info("数据处理完成，已保存到文件");


    spdlog::info("Match tracking results to bubble indexs");

    // 读取数据
    std::ifstream bb_file_3(file_path.c_str());
    if (!bb_file_3.is_open()) {
        spdlog::error("无法打开文件: {}", file_path);
        return;
    }
    std::vector<std::vector<float>> real_data;
    std::string line_3;
    while (std::getline(bb_file_3, line_3)) { //real_data的帧id是从1开始的
        std::istringstream iss(line_3);
        std::vector<float> data;
        float value;
        while (iss >> value) {
            data.push_back(value);
        }
        real_data.push_back(data);
    }

    bb_file_3.close();


    std::vector<std::vector<int>> real;
    for (const auto& row : real_data) {
        if(row[row.size()-2]>=thres){
            std::vector<int> rounded_row(row.begin(), row.end() - 2);
            std::transform(rounded_row.begin(), rounded_row.end(), rounded_row.begin(), [](float val) { return static_cast<int>(std::round(val)); });
            real.push_back(rounded_row);
        }
    }

    std::vector<std::vector<int>> pred_data;
    std::ifstream bb_file_4(output_file_path.c_str());
    if (!bb_file_4.is_open()) {
        spdlog::error("无法打开文件: {}", output_file_path);
        return;
    }
    std::string line_4;
    while (std::getline(bb_file_4, line_4)) {  // 逐行读取
        std::istringstream iss(line_4);
        std::string value;
        std::vector<int> data;
        while (std::getline(iss, value, ',')) {  // 使用逗号作为分隔符
            data.push_back(std::stoi(value));   // 将字符串转换为整数
        }
        // 打印数据
        spdlog::debug("{},{},{}", data[0], data[1], data[2]);
        spdlog::debug("------------");
        pred_data.push_back(data);
    }

    bb_file_4.close();


    // realg 
    std::map<int,std::vector<std::vector<int>>> my_dict;

    int num_frames = real.back()[0];

    for(int i=1;i<=num_frames;i++){
        my_dict[i] = std::vector<std::vector<int>>();
    }

    for(const auto& item : real){
        int key = item[0];
        std::vector<int> value(item.begin()+1,item.end());
        my_dict[key].push_back(value);
    }

    std::vector<std::vector<std::vector<int>>> realg;

    for(const auto& [key,value] : my_dict){
        realg.push_back(value);
        // print2DVector(value);
    }

    //relgG
    std::map<int,std::vector<int>> my_dict_2;
    int num_frames_2 = real.back()[0];
    for(int i=1;i<=num_frames_2;i++){
        my_dict_2[i] = std::vector<int>();
    }
    for(const auto& item : real){
        int key = item[0];
        int value = item[1];
        my_dict_2[key].push_back(value);
    }

    std::vector<std::vector<int>> relgG;
    for(const auto& [key,value] : my_dict_2){
        relgG.push_back(value);
        // print2DVector(value);
    }

    // predg
    std::map<int,std::vector<std::vector<int>>> my_dict_3;
    int num_frames_3 = pred_data.back()[0];
    for(int i=1;i<=num_frames_3;i++){
        my_dict_3[i] = std::vector<std::vector<int>>();
    }
    for(const auto& item : pred_data){
        int key = item[0];
        std::vector<int> value(item.begin()+1,item.end());
        my_dict_3[key].push_back(value);
    }

    std::vector<std::vector<std::vector<int>>> predg;
    for(const auto& [key,value] : my_dict_3){
        predg.push_back(value);
        // print2DVector(value);
    }

    // 拷贝一份
    std::vector<std::vector<std::vector<int>>> values=realg;
    std::vector<std::vector<int>> tracks=relgG;

    for(int k=0;k<predg.size()-1;k++){
        if(predg[k].size()>0 && predg[k+1].size()>0){
            std::vector<std::vector<int>> frame1=predg[k];
            std::vector<std::vector<int>> frame2=predg[k+1];
            
            std::vector<int> vector1,vector2;
            for(const auto& item : frame2){
                vector1.push_back(item[0]);
            }
            for(const auto& item : frame1){
                vector2.push_back(item[0]);
            }

            std::vector<int> result_vector(vector1.size(),-1);
            for(int i=0;i<vector1.size();i++){
                auto it = std::find(vector2.begin(),vector2.end(),vector1[i]);
                if(it!=vector2.end()){
                    result_vector[i]=std::distance(vector2.begin(),it);
                }
            }

            size_t j=0;
            for(int i=0;i<result_vector.size();i++){
                if(result_vector[i]!=-1){
                    if(frame2[i][1]!=frame1[result_vector[i]][1]){ // hits在两帧之间发生了变化,匹配到了
                        // 把预测帧的track_id 给更新到tracks的id上
                        tracks[k+1][j]=frame2[i][0]; 
                        values[k+1][j]= {frame2[i][2], frame2[i][3], frame2[i][4],frame2[i][5]};
                        ++j;
                    }
                }
                if(result_vector[i]==-1){
                    tracks[k+1][j]=frame2[i][0];
                    values[k+1][j]= {frame2[i][2], frame2[i][3], frame2[i][4],frame2[i][5]};
                    ++j;
                }
            }
        }
    }
    // 补充第一个trackers 和 values
    tracks[0].clear();
    values[0].clear();
    for(const auto& item : predg[0]){
        tracks[0].push_back(item[0]);
        values[0].push_back({item[2],item[3],item[4],item[5]});
    }

    // 更新 tracks
    for (size_t i = 0; i < values.size(); ++i) {
        if (!values[i].empty()) {
            auto iouMatrix = iouBatch(realg[i], values[i]); // 对检测目标和追踪的目标进行
            auto sort = argmax(iouMatrix);
            std::vector<int> sortedTracks;
            for (int idx : sort) {
                sortedTracks.push_back(tracks[i][idx]);
            }
            tracks[i] = sortedTracks;
        }
    }


    // Original data
    std::vector<std::vector<int>> data = tracks;

    int max_number = findMaxNumber(data);

    // 存储每一个追踪对象所在的帧
    std::vector<std::vector<int>> frames(max_number+1); // 是否需要+1

    for(size_t initial_row=0;initial_row<tracks.size();initial_row++){ // 每一个对象分别出现在第几帧
        for(int number:tracks[initial_row]){
            frames[number].push_back(initial_row); // TODO + 1
        }
    }


    // 存储每个追踪目标对应的边界框索引 ,一共13个追踪目标，在每一帧中对应的data的目标框索引地址
    std::vector<std::vector<int>> bubInd(max_number+1);

    for(const auto& row:tracks){
        for(size_t index=0;index<row.size();index++){
            bubInd[row[index]].push_back(index);
        }
    }

    save2DVectorToFile(bubind_file, bubInd);
    save2DVectorToFile(frameind_file, frames);

    std::vector<int> classes;
    for(const auto& item:real_data){
        classes.push_back(item.back());
    }

    std::map<int,std::vector<int>> my_dict_4;

    for(const auto& item: real_data){
        int key = item[0];
        int value = item.back();
        my_dict_4[key].push_back(value);
    }

    std::vector<std::vector<int>> realgG; // ！！！ 一切错误的原因都是因为之前my_dict_4 使用的是std::unordered_map,是无序的
    for(const auto& [key,value] : my_dict_4){
        realgG.push_back(value);
    }

    save2DVectorToFile(classind_file, realgG);

     // Create bub_class
    std::vector<std::vector<int>> bub_class = frames; // 记住：frames 表示的是每一个追踪的目标，在第几帧出现
    for (size_t j = 0; j < bubInd.size(); ++j) {
        for (size_t i = 0; i < frames[j].size(); ++i) {
            int frame_index = frames[j][i];
            int bub_index = bubInd[j][i];
            bub_class[j][i] = realgG[frame_index][bub_index];
            spdlog::debug("bub_class[j][i]: {}", bub_class[j][i]);
        }
    }

    // Save bub_class to file
    save2DVectorToFile(bubclassind_file, bub_class);

    spdlog::info("Finish");

    return ;
}


void DataAnalysis::save2DVectorToFile(const std::string& filename, const std::vector<std::vector<int>>& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        spdlog::error("Error opening file: {}", filename);
        return;
    }

    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}



int DataAnalysis::findMaxNumber(std::vector<std::vector<int>>& data){
    int max_number = std::numeric_limits<int>::min();
    for (const auto& sublist : data) {
        if (!sublist.empty()) {
            int local_max = *std::max_element(sublist.begin(), sublist.end());
            if (local_max > max_number) {
                max_number = local_max;
            }
        }
    }
    return max_number;
}

// 打印二维向量的函数
void DataAnalysis::print2DVector(const std::vector<std::vector<int>>& vec) {
    spdlog::debug("------------");
    for (const auto& row : vec) {
        for (const auto& val : row) {
            spdlog::debug("{} ", val);
        }
    }
    spdlog::debug("------------");
}

void DataAnalysis::saveDataToFiles() {
    std::string data_dir = savefolder_ + "/data";
    if (!directoryExists(data_dir))
        makeDirectories(data_dir);

    // 保存蒸汽数据
    std::ofstream vapor_file((data_dir + "/vapor_" + extension_ + ".txt").c_str());
    if (!vapor_file.is_open()) {
        spdlog::error("无法打开文件: vapor_{}.txt", extension_);
        return;
    }
    for (const auto& v : vapor_) {
        vapor_file << v << std::endl;
    }
    vapor_file.close();

    // 保存基础蒸汽数据
    std::ofstream vapor_base_file((data_dir + "/vaporBase_bt-" + extension_ + ".txt").c_str());
    if (!vapor_base_file.is_open()) {
        spdlog::error("无法打开文件: vaporBase_bt-{}.txt", extension_);
        return;
    }
    for (const auto& v : vapor_base_) {
        vapor_base_file << v << std::endl;
    }
    vapor_base_file.close();

    // 保存气泡大小数据
    std::ofstream bubble_size_file((data_dir + "/bubble_size_bt-" + extension_ + ".txt").c_str());
    if (!bubble_size_file.is_open()) {
        spdlog::error("无法打开文件: bubble_size_bt-{}.txt", extension_);
        return;
    }
    for (const auto& sizes : bubble_size_) {
        for (const auto& size : sizes) {
            bubble_size_file << size << " ";
        }
        bubble_size_file << std::endl;
    }
    bubble_size_file.close();

    // 保存边界框数据
    std::ofstream bounding_box_file((data_dir + "/bb-Boiling-" + extension_ + ".txt").c_str());
    if (!bounding_box_file.is_open()) {
        spdlog::error("无法打开文件: bb-Boiling-{}.txt", extension_);
        return;
    }
    for (const auto& box : bounding_box_) {
        for (const auto& val : box) {
            bounding_box_file << val << " ";
        }
        bounding_box_file << std::endl;
    }
    bounding_box_file.close();

    spdlog::info("数据已保存到文件");
}

void DataAnalysis::Plotvf() {
    std::string vf_path = savefolder_ + "/data/vapor_" + extension_ + ".txt";
    int width = 1280, height = 800;
    int window = 300;
    double vidstart = 0.0;

    // 1. 读取txt文件
    std::vector<double> vf;
    std::ifstream infile(vf_path);
    double value;
    while (infile >> value) {
        vf.push_back(value / (width * height));
    }
    spdlog::info("vf size: {}", vf.size());
    // 2. 生成时间序列
    std::vector<double> time(vf.size());
    for (size_t i = 0; i < vf.size(); ++i) {
        time[i] = (i / 150.0) + vidstart;
    }

    // 3. 计算滑动平均
    std::vector<double> rolling_avg(vf.size(), 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < vf.size(); ++i) {
        sum += vf[i];
        if (i >= window) sum -= vf[i - window];
        if (i >= window - 1)
            rolling_avg[i] = sum / window;
    }

    std::vector<double> time_pos, rolling_avg_pos;
    for (size_t i = 0; i < rolling_avg.size(); ++i) {
        if (rolling_avg[i] > 0) {
            time_pos.push_back(time[i]);
            rolling_avg_pos.push_back(rolling_avg[i]);
        }
    }
    // 4. 画图并保存（Matplot++，quiet 模式仅保存不弹窗）
    matplot::figure(true);  // quiet mode: 不更新交互窗口，仅 save 时输出
    matplot::plot(time, vf, "k");
    matplot::hold(matplot::on);
    matplot::plot(time_pos, rolling_avg_pos, "b");
    matplot::xlabel("Time (s)");
    matplot::ylabel("Vapor Fraction");
    matplot::legend({"Raw", "Rolling Average"});
    std::string fig_dir = savefolder_ + "/figures";
    if (!directoryExists(fig_dir)) makeDirectories(fig_dir);
    std::string saveloc = fig_dir + "/vaporfig_" + extension_ + ".png";
    spdlog::info("Saving figure to: {}", saveloc);
    matplot::save(saveloc);
}

void DataAnalysis::Plotbc() {
    // 1. 构造txt文件路径（data 子文件夹）
    std::string bs_path = savefolder_ + "/data/bubble_size_bt-" + extension_ + ".txt";
    int width = 1280, height = 800;;
    int window = 300;
    double vidstart = 0.0;

    // 2. 读取txt文件，每行统计气泡数量
    std::vector<int> bubble_counts;
    std::ifstream infile(bs_path);
    if (!infile.is_open()) {
        spdlog::error("无法打开文件: {}", bs_path);
        return;
    }
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        int count = 0;
        std::string tmp;
        while (iss >> tmp) {
            count++;
        }
        bubble_counts.push_back(count);
    }
    infile.close();

    // 3. 生成时间序列
    std::vector<double> time;
    for (size_t i = 0; i < bubble_counts.size(); ++i) {
        time.push_back(i / 150.0 + vidstart);
    }

    // 4. 计算滑动平均
    double sum = 0.0;
    std::vector<double> rolling_avg(bubble_counts.size(), 0.0);
    for (size_t i = 0; i < bubble_counts.size(); ++i) {
        sum += bubble_counts[i];
        if (i >= window) sum -= bubble_counts[i - window];
        if (i >= window - 1)
            rolling_avg[i] = sum / window;
    }

    std::vector<double> time_pos, rolling_avg_pos;
    for (size_t i = 0; i < rolling_avg.size(); ++i) {
        if (rolling_avg[i] > 0) {
            time_pos.push_back(time[i]);
            rolling_avg_pos.push_back(rolling_avg[i]);
        }
    }

    // 修正类型：将bubble_counts转为double类型
    std::vector<double> bubble_counts_d(bubble_counts.begin(), bubble_counts.end());

    // 5. 画图并保存（Matplot++，quiet 模式仅保存不弹窗）
    matplot::figure(true);  // quiet mode: 不更新交互窗口，仅 save 时输出
    matplot::plot(time, bubble_counts_d, "k");
    matplot::hold(matplot::on);
    matplot::plot(time_pos, rolling_avg_pos, "b");
    matplot::xlabel("Time (s)");
    matplot::ylabel("Bubble Count");
    matplot::legend({"Raw", "Rolling Average"});
    std::string fig_dir = savefolder_ + "/figures";
    if (!directoryExists(fig_dir)) makeDirectories(fig_dir);
    std::string saveloc = fig_dir + "/bcfig_" + extension_ + ".png";
    spdlog::info("Saving figure to: {}", saveloc);
    matplot::save(saveloc);
}

void DataAnalysis::PlotInterfaceVelocity(int bubble) {
    std::string directory_path = this->imagesfolder_;

    std::string bubind_file = this->savefolder_ + "/data/bubind_" + this->extension_ + ".txt";
    std::string frameind_file = this->savefolder_ + "/data/frames_" + this->extension_ + ".txt";

    // 开始加载模型YOLOv8-seg
    spdlog::info("加载模型开始");
    std::string model_path = this->modelweightsloc_;
    std::unique_ptr<Yolov8SegOnnx> infer_engine=std::make_unique<Yolov8SegOnnx>();

    if(!infer_engine->ReadModel(model_path,true,0,true)){
        spdlog::error("加载模型失败");
        return;
    }

    spdlog::info("加载模型完成");
    
    // 设置置信度
    bool ok;
    float initial_threshold = 0.5;
    if(ok && initial_threshold>=0.0f && initial_threshold<=1.0f){
        infer_engine->SetClassThreshold(initial_threshold);
    }

    std::vector<std::string> image_paths = getImagePaths(directory_path);


    std::vector<std::vector<int>> bubInd = load2DVectorFromFile(bubind_file);
    std::vector<std::vector<int>> frames = load2DVectorFromFile(frameind_file);


    int skip = 5;
    std::vector<std::vector<double>> angles;
    
    std::vector<std::string> image_paths_sub = image_paths;

    std::vector<int> indexs;

    std::vector<std::vector<cv::Point>> contours;

    for(int i=0;i<frames[bubble].size();i++){
        std::string img_path = image_paths_sub[frames[bubble][i]];
        cv::Mat img = cv::imread(img_path);

        std::vector<OutputSeg> output_seg;
        bool find = infer_engine->OnnxDetect(img,output_seg);

        if(find && !output_seg.empty()){
            std::vector<std::vector<float>> bbox;
            std::vector<cv::Mat> masks;
            std::vector<float> scores;

            int img_height = img.rows;
            int img_width = img.cols;

            for(const auto& seg: output_seg){
                float x1 = seg.box.x;
                float y1 = seg.box.y;
                float x2 = seg.box.x + seg.box.width;
                float y2 = seg.box.y + seg.box.height;

                bbox.push_back({x1,y1,x2,y2});

                cv::Mat restore_mask = restoreMaskToOriginalSize(seg.box,seg.boxMask,img_height,img_width);

                masks.push_back(restore_mask);
                scores.push_back(seg.confidence);

            }
            int k = 0;
            for(int j=0;j<bbox.size();j++){
                int x1,y1,x2,y2;
                x1 = bbox[j][0];
                y1 = bbox[j][1];
                x2 = bbox[j][2];
                y2 = bbox[j][3];

                if(y2 > -1000){
                    if(k==bubInd[bubble][i]){
                        // 假设masks[j]为cv::Mat类型
                        cv::Mat mask;
                        masks[j].convertTo(mask, CV_8U);
                        mask = mask * 255;
                        std::vector<std::vector<cv::Point>> contours1;
                        std::vector<cv::Vec4i> hierarchy;
                        cv::findContours(mask, contours1, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
                        std::vector<cv::Point> contour_points;
                        if(contours1.size() > 1){
                            // 取面积最大的轮廓
                            double max_area = 0.0;
                            int max_idx = 0;
                            for(size_t ci=0; ci<contours1.size(); ++ci){
                                double area = cv::contourArea(contours1[ci]);
                                if(area > max_area){
                                    max_area = area;
                                    max_idx = ci;
                                }
                            }
                            contour_points = contours1[max_idx];
                        }else if(contours1.size() == 1){
                            contour_points = contours1[0];
                        }
                        // reshape为(-1,2)等价于直接存vector<cv::Point>
                        contours.push_back(contour_points);
                        indexs.push_back(j);
                    }
                    k += 1;
                }
            }
        }
    }
    // test_mate
    std::vector<double> avg_mag;
    std::vector<std::vector<double>> mag;
    std::vector<std::vector<cv::Point>> contour_connections;
    // contours: std::vector<std::vector<cv::Point>>
    // frames: std::vector<std::vector<int>>
    // bubble: int
    // image_paths_sub: std::vector<std::string>
    for (size_t j = 0; j + skip + 1 < frames[bubble].size(); ++j) {

        int frame1 = frames[bubble][j];
        int frame2 = frames[bubble][j + skip];
        const std::vector<cv::Point>& set1 = contours[j];
        const std::vector<cv::Point>& set2 = contours[j + skip];

        // 最近邻查找
        std::vector<double> distances(set1.size());
        std::vector<int> indices(set1.size());
        for (size_t i = 0; i < set1.size(); ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int min_idx = -1;
            for (size_t k = 0; k < set2.size(); ++k) {
                double dx = set1[i].x - set2[k].x;
                double dy = set1[i].y - set2[k].y;
                double dist = std::sqrt(dx * dx + dy * dy);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_idx = k;
                }
            }
            distances[i] = min_dist;
            indices[i] = min_idx;
        }

        // 速度归一化
        double dt = (frame2 - frame1) / 3000.0;
        std::vector<double> velocitys(distances.size());
        for (size_t i = 0; i < distances.size(); ++i) {
            velocitys[i] = (distances[i] / 184.0) / dt;
        }
        mag.push_back(velocitys);
        double avg = std::accumulate(velocitys.begin(), velocitys.end(), 0.0) / velocitys.size();
        avg_mag.push_back(avg);

        // 方向角
        std::vector<double> ang;
        std::vector<cv::Point> connections;
        for (size_t i = 0; i < set1.size(); ++i) {
            const cv::Point& point1 = set1[i];
            const cv::Point& point2 = set2[indices[i]];
            double dx = point1.x - point2.x;
            double dy = point1.y - point2.y;
            double angle_rad = std::atan2(dy, dx) * 180.0 / CV_PI;
            if (angle_rad < 0) angle_rad += 360.0;
            ang.push_back(angle_rad);
            connections.push_back(point2);
        }
        angles.push_back(ang);
        contour_connections.push_back(connections);  
    }

    std::vector<std::vector<int>> direction;
    for (size_t i = 0; i + skip + 1 < contours.size(); ++i) {
        // 读取图片
        cv::Mat new_im = cv::imread(image_paths_sub[frames[bubble][i]]);
        // 推理获得所有mask
        std::vector<OutputSeg> output_seg;
        bool find = infer_engine->OnnxDetect(new_im, output_seg);
        std::vector<cv::Mat> masks;
        for (const auto& seg : output_seg) {
            cv::Mat restore_mask = restoreMaskToOriginalSize(seg.box, seg.boxMask, new_im.rows, new_im.cols);
            masks.push_back(restore_mask);
        }
        // 取当前indexs[i]的mask
        cv::Mat mask;
        masks[indexs[i]].convertTo(mask, CV_8U);
        mask = mask * 255;

        std::vector<int> class_val;
        for (size_t j = 0; j < contours[i].size(); ++j) {
            int x_coord = contour_connections[i][j].x;
            int y_coord = contour_connections[i][j].y;
            // 边界检查
            if (x_coord >= 832) x_coord = 831;
            else if (x_coord < 0) x_coord = 0;
            if (y_coord < 0) y_coord = 0;
            else if (y_coord >= 600) y_coord = 599;

            if (mask.at<uchar>(y_coord, x_coord) == 255) {
                class_val.push_back(1);
            } else {
                class_val.push_back(0);
            }
        }
        direction.push_back(class_val);
    }
    spdlog::info("direction size: {}", direction.size());   

    // 1. direction==1的mag取反
    for (size_t i = 0; i < direction.size(); ++i) {
        for (size_t j = 0; j < direction[i].size(); ++j) {
            if (direction[i][j] == 1) {
                mag[i][j] = -mag[i][j];
            }
        }
    }

    // 2. 统一mag长度，采样为num_entries=200
    int num_entries = 200;
    int min_length = mag.empty() ? 0 : mag[0].size();
    for (const auto& row : mag) {
        if ((int)row.size() < min_length) min_length = row.size();
    }
    // 采样
    cv::Mat data((int)mag.size(), num_entries, CV_64F);
    for (size_t i = 0; i < mag.size(); ++i) {
        std::vector<double> &row = mag[i];
        std::vector<int> indices(num_entries);
        for (int k = 0; k < num_entries; ++k) {
            indices[k] = static_cast<int>(std::round(k * (row.size() - 1.0) / (num_entries - 1)));
        }
        for (int k = 0; k < num_entries; ++k) {
            data.at<double>((int)i, k) = row[indices[k]];
        }
    }

    // 3. 循环移位 split_val=100
    int split_val = 100;
    cv::Mat data1 = cv::Mat::zeros(data.size(), CV_64F);
    if (split_val < num_entries) {
        data.colRange(split_val, num_entries).copyTo(data1.colRange(0, num_entries - split_val));
        data.colRange(0, split_val).copyTo(data1.colRange(num_entries - split_val, num_entries));
    } else {
        data.copyTo(data1);
    }
    data = data1;

    // 4. 高斯平滑
    cv::Mat data_smoothed;
    cv::GaussianBlur(data, data_smoothed, cv::Size(0, 0), 2);

    // 5. 画图并保存
    double total_time = (double)data.rows / 3000.0;
    // 归一化到[-30, 30]
    cv::Mat data_norm;
    cv::normalize(data_smoothed, data_norm, -30, 30, cv::NORM_MINMAX);
    // 伪彩色
    cv::Mat data_color;
    data_norm.convertTo(data_norm, CV_8U, 255.0 / 60.0, 127.5); // [-30,30]映射到[0,255]
    cv::applyColorMap(data_norm, data_color, cv::COLORMAP_TURBO);
    // 转置（与imshow.T一致）
    cv::Mat data_color_T;
    cv::transpose(data_color, data_color_T);
    // 保存图片到 figures 子文件夹
    std::string fig_dir = savefolder_ + "/figures";
    if (!directoryExists(fig_dir)) makeDirectories(fig_dir);
    std::string saveloc = fig_dir + "/velocity_" + extension_ + "_" + std::to_string(bubble) + ".png";
    cv::imwrite(saveloc, data_color_T);
    spdlog::info("Velocity image saved to: {}", saveloc);
}


// 判断目录是否存在
bool DataAnalysis::directoryExists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

// 创建多级目录（Linux 版）
bool DataAnalysis::makeDirectories(const std::string& path) {
        if (path.empty()) return false;

    std::istringstream iss(path);
    std::string token;
    std::string currentPath;

    // Linux 使用 '/' 作为分隔符
    const char delim = '/';

    // 如果是绝对路径，保留前导 /
    if (path[0] == delim) {
        currentPath += delim;
    }

    while (std::getline(iss, token, delim)) {
        if (token.empty()) continue;
        currentPath += token + delim;

        if (!directoryExists(currentPath)) {
            if (mkdir(currentPath.c_str(), 0755) != 0) {
                spdlog::error("无法创建目录: {}", currentPath);
                return false;
            }
        }
    }
    return true;
}

// 判断是否是目标图像扩展名（不区分大小写）
bool DataAnalysis::hasJpgExtension(const std::string& filename) {
    if (filename.size() < 4) return false;
    std::string ext = filename.substr(filename.size() - 4);
    for (auto& c : ext) c = std::tolower(c);
    return ext == ".jpg";
}

void DataAnalysis::getImagePathsRecursive(const std::string& dir_path, std::vector<std::string>& out_paths) {
    DIR* dir = opendir(dir_path.c_str());
    if (!dir) return;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;

        // Skip '.' and '..'
        if (name == "." || name == "..") continue;

        std::string full_path = dir_path + "/" + name;

        struct stat st;
        if (stat(full_path.c_str(), &st) == -1) continue;

        if (S_ISDIR(st.st_mode)) {
            // Recursive call for subdirectory
            getImagePathsRecursive(full_path, out_paths);
        } else if (S_ISREG(st.st_mode)) {
            if (hasJpgExtension(name)) {
                out_paths.push_back(full_path);
            }
        }
    }

    closedir(dir);
}

// 主函数：递归查找目录下所有 .jpg 文件
std::vector<std::string> DataAnalysis::getImagePaths(const std::string& root_dir) {
    std::vector<std::string> image_paths;
    getImagePathsRecursive(root_dir, image_paths);
    std::sort(image_paths.begin(), image_paths.end());
    return image_paths;
}