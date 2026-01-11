#include "yolov8_seg/yolov8_seg_onnx.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace Ort;
 
bool Yolov8SegOnnx::ReadModel(const std::string& modelPath, bool isCuda, int cudaID, bool warmUp) {
    if (_batchSize < 1) _batchSize = 1;
    try
    {
        std::vector<std::string> available_providers = GetAvailableProviders(); //  "CPUExecutionProvider" 、 "CUDAExecutionProvider"、"TensorrtExecutionProvider"
        auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");
        // TensorrtExecutionProvider 相比于 CUDAExecutionProvider 有进一步的优化

        if (isCuda && (cuda_available == available_providers.end()))
        {
            std::cout << "Your ORT build without GPU. Change to CPU." << std::endl;
            std::cout << "************* Infer model on CPU! *************" << std::endl;
        }
        else if (isCuda && (cuda_available != available_providers.end()))
        {
            std::cout << "************* Infer model on GPU! *************" << std::endl;
#if ORT_API_VERSION < ORT_OLD_VISON
			OrtCUDAProviderOptions cudaOption;
			cudaOption.device_id = cudaID; 
			_OrtSessionOptions.AppendExecutionProvider_CUDA(cudaOption);
#else
			OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(_OrtSessionOptions, cudaID); // CUDA Execution Provider 加入到 ONNX Runtime 会话的配置中，并指定使用哪个 GPU 进行推理
#endif
        }
        else
        {
            std::cout << "************* Infer model on CPU! *************" << std::endl;
        }
        // 启用计算图的优化
        _OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
 
#ifdef _WIN32
        std::wstring model_path(modelPath.begin(), modelPath.end());
        _OrtSession = new Ort::Session(_OrtEnv, model_path.c_str(), _OrtSessionOptions);
#else
        _OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#endif
 
        Ort::AllocatorWithDefaultOptions allocator;
        //init input
        _inputNodesNum = _OrtSession->GetInputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
        _inputName = _OrtSession->GetInputName(0, allocator);
        _inputNodeNames.push_back(_inputName);
#else
        _inputName = std::move(_OrtSession->GetInputNameAllocated(0, allocator)); // allocator – to allocate memory for the copy of the name returned
        _inputNodeNames.push_back(_inputName.get());
#endif
 
        Ort::TypeInfo inputTypeInfo = _OrtSession->GetInputTypeInfo(0);
        auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
        _inputNodeDataType = input_tensor_info.GetElementType();
        _inputTensorShape = input_tensor_info.GetShape();
 
        if (_inputTensorShape[0] == -1)
        {
            _isDynamicShape = true;
            _inputTensorShape[0] = _batchSize;
 
        }
        if (_inputTensorShape[2] == -1 || _inputTensorShape[3] == -1) {
            _isDynamicShape = true;
            _inputTensorShape[2] = _netHeight;
            _inputTensorShape[3] = _netWidth;
        }
        //init output
        _outputNodesNum = _OrtSession->GetOutputCount();
        if (_outputNodesNum != 2) {
            cout << "This model has " << _outputNodesNum << "output, which is not a segmentation model.Please check your model name or path!" << endl;
            return false;
        }
#if ORT_API_VERSION < ORT_OLD_VISON
        _output_name0 = _OrtSession->GetOutputName(0, allocator);
        _output_name1 = _OrtSession->GetOutputName(1, allocator);
#else
        _output_name0 = std::move(_OrtSession->GetOutputNameAllocated(0, allocator)); // 获取第一个输出节点的名称 
        _output_name1 = std::move(_OrtSession->GetOutputNameAllocated(1, allocator));
#endif
        Ort::TypeInfo type_info_output0(nullptr);
        Ort::TypeInfo type_info_output1(nullptr);
        bool flag = false;
#if ORT_API_VERSION < ORT_OLD_VISON
        flag = strcmp(_output_name0, _output_name1) < 0;
#else   
        flag = strcmp(_output_name0.get(), _output_name1.get()) < 0; // 比较字典序
#endif
        if (flag)  //make sure "output0" is in front of  "output1"
        {
            type_info_output0 = _OrtSession->GetOutputTypeInfo(0);  //output0
            type_info_output1 = _OrtSession->GetOutputTypeInfo(1);  //output1
#if ORT_API_VERSION < ORT_OLD_VISON
            _outputNodeNames.push_back(_output_name0);
            _outputNodeNames.push_back(_output_name1);
#else
            _outputNodeNames.push_back(_output_name0.get());
            _outputNodeNames.push_back(_output_name1.get());
#endif
 
        }
        else {
            type_info_output0 = _OrtSession->GetOutputTypeInfo(1);  //output0
            type_info_output1 = _OrtSession->GetOutputTypeInfo(0);  //output1
#if ORT_API_VERSION < ORT_OLD_VISON
            _outputNodeNames.push_back(_output_name1);
            _outputNodeNames.push_back(_output_name0);
#else
            _outputNodeNames.push_back(_output_name1.get());
            _outputNodeNames.push_back(_output_name0.get());
#endif
        }
 
        auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
        _outputNodeDataType = tensor_info_output0.GetElementType();
        _outputTensorShape = tensor_info_output0.GetShape();
        auto tensor_info_output1 = type_info_output1.GetTensorTypeAndShapeInfo();
        //_outputMaskNodeDataType = tensor_info_output1.GetElementType(); //the same as output0
        //_outputMaskTensorShape = tensor_info_output1.GetShape();
        //if (_outputTensorShape[0] == -1)
        //{
        //	_outputTensorShape[0] = _batchSize;
        //	_outputMaskTensorShape[0] = _batchSize;
        //}
        //if (_outputMaskTensorShape[2] == -1) {
        //	//size_t ouput_rows = 0;
        //	//for (int i = 0; i < _strideSize; ++i) {
        //	//	ouput_rows += 3 * (_netWidth / _netStride[i]) * _netHeight / _netStride[i];
        //	//}
        //	//_outputTensorShape[1] = ouput_rows;
 
        //	_outputMaskTensorShape[2] = _segHeight;
        //	_outputMaskTensorShape[3] = _segWidth;
        //}
        //warm up
        if (isCuda && warmUp) { //warmup能够预热cuda的硬件资源、 提前缓存好优化后的计算图
            //draw run
            cout << "Start warming up" << endl;
            size_t input_tensor_length = VectorProduct(_inputTensorShape); // 计算所有的元素的个数
            float* temp = new float[input_tensor_length];
            std::vector<Ort::Value> input_tensors;
            std::vector<Ort::Value> output_tensors;
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                _OrtMemoryInfo, temp, input_tensor_length, _inputTensorShape.data(), // 存放vector元素的内部指针
                _inputTensorShape.size()));
            for (int i = 0; i < 3; ++i) {
                output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
                    _inputNodeNames.data(),
                    input_tensors.data(),
                    _inputNodeNames.size(),
                    _outputNodeNames.data(),
                    _outputNodeNames.size());
            }
 
            delete[]temp;
        }
    }
    catch (const std::exception&) {
        return false;
    }
    return true;
}
 
int Yolov8SegOnnx::Preprocessing(const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs, std::vector<cv::Vec4d>& params) {
    outSrcImgs.clear();
    Size input_size = Size(_netWidth, _netHeight);
    for (int i = 0; i < srcImgs.size(); ++i) {
        Mat temp_img = srcImgs[i];
        Vec4d temp_param = { 1,1,0,0 }; // 目前这个参数的作用未知
        if (temp_img.size() != input_size) { // 640 1280!= 640 640 
            Mat borderImg;
            LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
            //cout << borderImg.size() << endl;
            outSrcImgs.push_back(borderImg);
            params.push_back(temp_param); // temp里面记录的是 缩放比例 以及两个方向填充的像素大小
        }
        else {
            outSrcImgs.push_back(temp_img);
            params.push_back(temp_param);
        }
    }
 
    int lack_num = srcImgs.size() % _batchSize;
    if (lack_num != 0) {
        for (int i = 0; i < lack_num; ++i) {
            Mat temp_img = Mat::zeros(input_size, CV_8UC3);
            Vec4d temp_param = { 1,1,0,0 };
            outSrcImgs.push_back(temp_img);
            params.push_back(temp_param);
        }
    }
    return 0;
 
}

bool Yolov8SegOnnx::OnnxDetect(cv::Mat& srcImg, std::vector<OutputSeg>& output) {
    std::vector<cv::Mat> input_data = { srcImg };
    std::vector<std::vector<OutputSeg>> tenp_output;
    if (OnnxBatchDetect(input_data, tenp_output)) {
        output = tenp_output[0];
        return true;
    }
    else return false;
}
bool Yolov8SegOnnx::OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputSeg>>& output) {
    vector<Vec4d> params;
    vector<Mat> input_images;
    cv::Size input_size(_netWidth, _netHeight);  // 模型输入的大小 640 640 
    //preprocessing
    Preprocessing(srcImgs, input_images, params); // 对输入的图像进行预处理
    cv::Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, Scalar(0, 0, 0), true, false);
    // blobFromImages的作用是返回一个符合深度学习输入格式的cv::Mat，图像一般都是3维的 Mat 这个返回一个四维的Mat，即神经网络的输入格式 N C H W
    int64_t input_tensor_length = VectorProduct(_inputTensorShape);
    std::vector<Ort::Value> input_tensors; // Ort::Value 是模型输入的类型，也就是ORT C++ API中表示Tensor（张量）的类型。
    std::vector<Ort::Value> output_tensors; // _OrtMemoryInfo表示的是在cpu上创建 还是在gpu上创建
    input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data, input_tensor_length, _inputTensorShape.data(), _inputTensorShape.size()));
    

    output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
        _inputNodeNames.data(),
        input_tensors.data(),
        _inputNodeNames.size(),
        _outputNodeNames.data(),
        _outputNodeNames.size()
    ); // 因为有两个输出的节点 因此output_tensors的长度是2  0代表的output-0 1代表的是output-1 
 
    //post-process       对输出进行后处理
    // 第一个输出 output0: type float32 [1,37,8400] 
        // 其中37表示的是类别数(1) + 定位框的值(4) + 32（掩码用的字段）
    // 第二个输出 output1: type float32 [1,32,160,160]
        // output0 后32个字段 与 output1的数据做矩阵乘法后的得到的结果就是对应目标的掩膜数据
    int net_width = _className.size() + 4 + _segChannels; // 定位框的值 + 类别 + 32（掩码用的字段）
    float* all_data = output_tensors[0].GetTensorMutableData<float>(); // 1 37 8400  8400表示的是一张图上会检测8400个对象 转置成 （8400，32+4+1) 
    _outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();  // 1 32 160 160
    _outputMaskTensorShape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    vector<int> mask_protos_shape = { 1,(int)_outputMaskTensorShape[1],(int)_outputMaskTensorShape[2],(int)_outputMaskTensorShape[3] };
    int mask_protos_length = VectorProduct(mask_protos_shape); //  1 * 32 * 160 * 160
    int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0]; // 计算出一个图形的输出
    for (int img_index = 0; img_index < srcImgs.size(); ++img_index) {
        Mat output0 = Mat(Size((int)_outputTensorShape[2], (int)_outputTensorShape[1]), CV_32F, all_data).t();  //[bs,37,8400]=>[bs,8400,37]
        all_data += one_output_length; // 数据跳到下一个指针
        float* pdata = (float*)output0.data;
        int rows = output0.rows; // 8400
        std::vector<int> class_ids;//\BD\E1\B9\FBid\CA\FD\D7\E9
        std::vector<float> confidences;//\BD\E1\B9\FBÿ\B8\F6id\B6\D4Ӧ\D6\C3\D0Ŷ\C8\CA\FD\D7\E9
        std::vector<cv::Rect> boxes;//ÿ\B8\F6id\BE\D8\D0ο\F2
        std::vector<vector<float>> picked_proposals;  //output0[:,:, 5 + _className.size():net_width]===> for mask
        for (int r = 0; r < rows; ++r) {    // 检测框 + 类别 + 32
                cv::Mat scores(1, _className.size(), CV_32F, pdata + 4); // 参数: 矩阵的行数, 矩阵的列数，矩阵元素的类型,数据偏移量
                Point classIdPoint;
                double max_class_socre; //下面函数的作用是找到scores矩阵中最大值和最小值的位置，方便找到最有可能的类别
                minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint); //参数: 矩阵、最小值的输出位置，矩阵的最大值
                max_class_socre = (float)max_class_socre;
                if (max_class_socre >= _classThreshold) { // 
                    vector<float> temp_proto(pdata + 4 + _className.size(), pdata + net_width); // 把用于mask的32个元素拷贝走
                    picked_proposals.push_back(temp_proto);
                    //rect [x,y,w,h]
                    float x = (pdata[0] - params[img_index][2]) / params[img_index][0];  //x 符合条件的检测 恢复到原图的坐标
                    float y = (pdata[1] - params[img_index][3]) / params[img_index][1];  //y
                    float w = pdata[2] / params[img_index][0];  //w
                    float h = pdata[3] / params[img_index][1];  //h
                    int left = MAX(int(x - 0.5 * w + 0.5), 0); 
                    int top = MAX(int(y - 0.5 * h + 0.5), 0); // 形式为左上角坐标值 以及对应的宽和高
                    class_ids.push_back(classIdPoint.x); // 我觉得这个值应该是classIdPoint.y
                    confidences.push_back(max_class_socre);
                    boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
                }
                pdata += net_width; // 取计算下一个张量的值  
        }
 
        vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
        std::vector<vector<float>> temp_mask_proposals; 
        cv::Rect holeImgRect(0, 0, srcImgs[img_index].cols, srcImgs[img_index].rows); // 创建一个矩形区域 大小与原始图像保持一致
        std::vector<OutputSeg> temp_output;    /// nms_result // 存储抑制后保留下来的框的索引
        for (int i = 0; i < nms_result.size(); ++i) {
            int idx = nms_result[i];
            OutputSeg result;
            result.id = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx] & holeImgRect; // 返回边界框与整体的大框之间的交集 ，将检测框限制在大框内
            temp_mask_proposals.push_back(picked_proposals[idx]);
            temp_output.push_back(result);
        }
 
        MaskParams mask_params;
        mask_params.params = params[img_index];
        mask_params.srcImgShape = srcImgs[img_index].size();
        Mat mask_protos = Mat(mask_protos_shape, CV_32F, output_tensors[1].GetTensorMutableData<float>() + img_index * mask_protos_length);
        for (int i = 0; i < temp_mask_proposals.size(); ++i) {
            GetMask2(Mat(temp_mask_proposals[i]).t(), mask_protos, temp_output[i], mask_params);
        }
 
        output.push_back(temp_output);
 
    }
 
    if (output.size())
        return true;
    else
        return false;
}