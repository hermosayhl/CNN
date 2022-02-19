//C++
#include <random>
#include <vector>
#include <iostream>
// OpenCV
#include <opencv2/highgui.hpp>
// self
#include "architectures.h"


namespace {
    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}



void test_augment() {
    // 增强器
    pipeline::ImageAugmentor augmentor({{"hflip", 1.0}, {"vflip", 0.1}, {"crop", 1.0}, {"rotate", 1.0}});
    // 读取图像
    cv::Mat origin = cv::imread("../datasets/images/dog.jpg");
    assert(not origin.empty());
    cv_show(origin);
    // 做变换
    augmentor.make_augment(origin, true);
}


void test_dataloader() {
    std::setbuf(stdout, 0);

    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    using namespace architectures;

    // 指定一些参数
    const int train_batch_size = 4;
    const std::tuple<int, int, int> image_size({224, 224, 3});
    const std::filesystem::path dataset_path("../datasets/animals");
    const std::vector<std::string> categories({"dog", "panda", "bird"});

    // 获取图片
    auto dataset = pipeline::get_images_for_classification(dataset_path, categories);

    // 构造数据流
    pipeline::DataLoader train_loader(dataset["train"], train_batch_size, false, true, image_size);

    // 开始不断获取 batch
    for(int i = 0;i < 10; ++i) {
        // 获取一个 batch
        auto sample = train_loader.generate_batch();
        // 拆成 tensor 和 类别序号
        const auto& images = sample.first;
        const auto& labels = sample.second;
        // 从 tensor 恢复成 opencv::Mat 格式
        for(int b = 0;b < train_batch_size; ++b) {
            std::cout << "[Batch " << i << "] " << " [" << b + 1 << "/" << train_batch_size << "]===> " << categories[labels[b]] << std::endl;
            const auto origin = images[b]->opecv_mat(3);
            cv_show(origin);
        }
    }
}



void test_relu_layer() {
    // 定义变量
    std::vector<tensor> input;
    const std::tuple<int, int, int> _shape({16, 7, 7});
    input.emplace_back(new Tensor3D(_shape));
    // 随机数生成
    std::default_random_engine e;
    e.seed(212);
    std::normal_distribution<float> engine(0.0, 1.0);
    // 初始化
    data_type* const data_ptr = input[0]->data;
    const int length = input[0]->get_length();
    for(int i = 0;i < length; ++i)
        data_ptr[i] = engine(e);
    // 打印第 0 张图像特征的第 3 个通道内容
    const int CH = 3;
    input[0]->print(CH);
    // 声明 ReLU 层
    architectures::ReLU relu_layer("relu_test");
    // relu 过滤
    auto output = relu_layer.forward(input);
    // 打印同个位置过滤之后的信息
    output[0]->print(CH);

    // 模拟反向传播回来的 delta
    std::vector<tensor> delta({tensor(new Tensor3D(_shape))});
    // 随机生成 delta 的数据
    for(int i = 0;i < length; ++i)
        delta[0]->data[i] = engine(e);
    // 打印所有梯度的内容
    delta[0]->print(CH);

    // 得到回传给上一层的梯度
    auto delta_backward = relu_layer.backward(delta);
    // 打印同个位置回传给上一层的梯度
    delta_backward[0]->print(CH);

}

void test_maxpool2d_layer() {
    // 定义变量
    std::vector<tensor> input;
    const std::tuple<int, int, int> _shape({16, 6, 6});
    input.emplace_back(new Tensor3D(_shape));
    // 随机数生成
    std::default_random_engine e;
    e.seed(1998);
    std::normal_distribution<float> engine(0.0, 1.0);
    // 初始化
    data_type* const data_ptr = input[0]->data;
    const int length = input[0]->get_length();
    for(int i = 0;i < length; ++i)
        data_ptr[i] = engine(e);
    // 打印第 0 张图像特征的第 3 个通道内容
    const int CH = 3;
    input[0]->print(CH);

    // 声明 ReLU 层
    architectures::MaxPool2D maxpool_layer("relu_test", 2, 2);
    // relu 过滤
    auto output = maxpool_layer.forward(input);
    // 打印同个位置过滤之后的信息
    output[0]->print(CH);
    output[0]->print_shape();

    // 模拟反向传播回来的 delta
    const std::tuple<int, int, int> _shape2({16, 3, 3});
    std::vector<tensor> delta({tensor(new Tensor3D(_shape2))});
    // 随机生成 delta 的数据
    const int length_2 = delta[0]->get_length();
    for(int i = 0;i < length_2; ++i)
        delta[0]->data[i] = engine(e);
    // 打印所有梯度的内容
    delta[0]->print(CH);

    // 得到回传给上一层的梯度
    auto delta_backward = maxpool_layer.backward(delta);
    // 打印同个位置回传给上一层的梯度
    delta_backward[0]->print(CH);
}




void test_AlexNet() {
    using namespace architectures;
    // 网络
    AlexNet network(3, false);
    network.print_info = true;
    // 定义输入
    std::vector<tensor> input;
    const int batch_size = 8;
    for(int i = 0;i < batch_size; ++i)
        input.emplace_back(new Tensor3D(3, 224, 224));
    // forward
    auto output = network.forward(input);
    // 定义梯度
    std::vector<tensor> delta;
    for(int i = 0;i < batch_size; ++i)
        delta.emplace_back(new Tensor3D(3, 1, 1, "delta_from_loss_" + std::to_string(i)));
    // backward
    network.backward(delta);
}




int main() {
    test_AlexNet();
    return 0;
}