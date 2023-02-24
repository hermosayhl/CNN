// C++
#include <string>
#include <iostream>
#include <filesystem>
// OpenCV
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
// self
#include "func.h"
#include "architectures.h"

namespace {
    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

int main() {
    // 输出不要放在缓冲区, 到时间了及时输出
    std::setbuf(stdout, 0);

    using namespace architectures;
    std::cout << "inference\n";

    // 指定一些参数
    const std::vector<std::string> categories({"dog", "panda", "bird"});

    // 定义网络结构
    const int num_classes = categories.size(); // 分类的数目
    AlexNet network(num_classes);

    // 直接加载
    network.load_weights("../checkpoints/AlexNet_aug_1e-3/iter_395000_train_0.918_valid_0.913.model");

    // 准备测试的图片
    std::vector<std::string> images_list({
        "../../datasets/images/dog.jpg",
        "../../datasets/images/panda.jpg",
        "../../datasets/images/bird.jpg"
    });

    // 准备一块图像内容存放的空间
    const std::tuple<int, int, int> image_size({3, 224, 224});
    tensor buffer_data(new Tensor3D(image_size, "inference_buffer"));
    std::vector<tensor> image_buffer({buffer_data});

    // 去掉梯度计算
    WithoutGrad guard;

    // 逐一读取图像, 做变换
    for(const auto& image_path : images_list) {
        // 读取图像
        cv::Mat origin = cv::imread(image_path);
        if(origin.empty() || !std::filesystem::exists(image_path)) {
            std::cout << "Failed to read image file  " << image_path << "\n";
            continue;
        }
        // 图像 resize 到规定的大小, 224 X 224
        cv::resize(origin, origin, {std::get<1>(image_size), std::get<2>(image_size)});
        // 转化为 tensor 数据
        image_buffer[0]->read_from_opencv_mat(origin.data);
        // 经过卷积神经网络得到输出
        const auto output = network.forward(image_buffer);
        // softmax 得到输出
        const auto prob = softmax(output);
        // 找到最大概率的输出
        const int max_index = prob[0]->argmax();
        std::cout << image_path << "===> [classification: " << categories[max_index] << "] [prob: " << prob[0]->data[max_index] << "]\n";
        cv_show(origin);
    }
}