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

    // 指定一些参数
    const std::vector<std::string> categories({"dog", "panda", "bird"});

    // 定义网络结构
    const int num_classes = categories.size(); // 分类的数目
    AlexNet network(num_classes, false);

    // 直接加载
    network.load_weights("../checkpoints/AlexNet_aug_1e-3/iter_395000_train_0.918_valid_0.913.model");

    // 准备测试的图片
    std::vector<std::string> images_list({
        "../../datasets/images/dog.jpg",
        "../../datasets/images/bird_2.jpg",
        "../../datasets/images/panda.jpg",
        "../../datasets/images/dog_3.jpg",
        "../../datasets/images/panda_2.jpg",
        "../../datasets/images/bird.jpg",
    });

    // 结果保存到哪里
    const std::filesystem::path visualize_dir("../output/");
    if(!std::filesystem::exists(visualize_dir))
        std::filesystem::create_directories(visualize_dir);

    // 准备一块图像内容存放的空间
    const std::tuple<int, int, int> image_size({3, 224, 224});
    tensor buffer_data(new Tensor3D(image_size, "inference_buffer"));
    std::vector<tensor> image_buffer({buffer_data});

    // 打开梯度计算
    no_grad = false;

    // 逐一读取图像, 做变换
    int image_no = 0;
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
        // 接下来做 grad cam 可视化
        cv::Mat cam = 255 - network.grad_cam("conv_layer_3");
        // 将 6x6 特征图放大到 origin 大小
        cv::resize(cam, cam, {std::get<1>(image_size), std::get<2>(image_size)});
        // 转化成热力图
        cv::Mat heat_map;
        cv::applyColorMap(cam, heat_map, cv::COLORMAP_JET);
        origin.convertTo(origin, CV_32FC3);
        heat_map.convertTo(heat_map, CV_32FC3);
        heat_map = heat_map / 255 + origin / 255;
        float maxValue = *std::max_element(heat_map.begin<float>(), heat_map.end<float>());
        heat_map = heat_map / maxValue;
        heat_map = heat_map * 255;
        heat_map.convertTo(heat_map, CV_8UC3);
        cv::imwrite((visualize_dir / std::to_string(image_no++)).string() + ".png", heat_map);
        cv_show(heat_map);
    }
}