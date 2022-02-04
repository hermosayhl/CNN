// C++
#include <iostream>
// self
#include "architectures.h"

using namespace architectures;

AlexNet::AlexNet(const int num_classes)
    : classifier(LinearLayer("linear_1", 6 * 6 * 128, num_classes)) {}

std::vector<tensor1D> AlexNet::forward(const std::vector<tensor>& input) {
    // 对输入的形状做检查
    assert(input.size() > 0);
    // batch_size X 3 X 224 X 224
    auto conv_output_1 = this->conv_layer_1.forward(input);
    auto relu_output_1 = this->relu_layer_1.forward(conv_output_1);
    // batch_size X 16 X 111 X 111
    auto pool_output_1 = this->max_pool_1.forward(relu_output_1);
    // batch_size X 16 X 55 X 55
    auto conv_output_2 = this->conv_layer_2.forward(pool_output_1);
    auto relu_output_2 = this->relu_layer_2.forward(conv_output_2);
    // batch_size X 32 X 27 X 27
    auto conv_output_3 = this->conv_layer_3.forward(relu_output_2);
    auto relu_output_3 = this->relu_layer_3.forward(conv_output_3);
    // batch_size X 64 X 13 X 13
    auto conv_output_4 = this->conv_layer_4.forward(relu_output_3);
    auto relu_output_4 = this->relu_layer_4.forward(conv_output_4);
    // batch_size X 128 X 6 X 6
    auto output = this->classifier.forward(relu_output_4);
    // batch_size X num_classes
    return output;
}

// 梯度反传
void AlexNet::backward(const std::vector<tensor1D>& delta_start) {
    auto delta = this->classifier.backward(delta_start);
    delta = this->relu_layer_4.backward(delta);
    delta = this->conv_layer_4.backward(delta);
    delta = this->relu_layer_3.backward(delta);
    delta = this->conv_layer_3.backward(delta);
    delta = this->relu_layer_2.backward(delta);
    delta = this->conv_layer_2.backward(delta);
    delta = this->max_pool_1.backward(delta);
    delta = this->relu_layer_1.backward(delta);
    delta = this->conv_layer_1.backward(delta);
}

// 这种写法灵活性差点, 新添加一层要改动很多; 后面可以考虑用多态, 存储指针试试
void AlexNet::update_gradients(const data_type learning_rate) {
    this->classifier.update_gradients(learning_rate);
    this->conv_layer_4.update_gradients(learning_rate);
    this->conv_layer_3.update_gradients(learning_rate);
    this->conv_layer_2.update_gradients(learning_rate);
    this->conv_layer_1.update_gradients(learning_rate);
}


// 保存模型权值, 灵活性很差
void AlexNet::save_weights(const std::filesystem::path& save_path) const {
    // 首先明确, 需要保存权值的只有 Conv2d, linear, batchnorm2D 这些
    // 写法上不是字典的, 只能做个 demo, 反射偶尔开开还是可以的, 可惜
    std::ofstream writer(save_path.c_str(), std::ios::binary);
    // 首先这里本来应该写一下有哪些组件, 然后写一下组件的具体信息, 但是免了
    this->conv_layer_1.save_weights(writer);
    this->conv_layer_2.save_weights(writer);
    this->conv_layer_3.save_weights(writer);
    this->conv_layer_4.save_weights(writer);
    this->classifier.save_weights(writer);
    std::cout << "weights have been saved to " << save_path.string() << std::endl;
    writer.close();
}

// 加载模型权值
void AlexNet::load_weights(const std::filesystem::path& checkpoint_path) {
    if(not std::filesystem::exists(checkpoint_path)) {
        std::cout << "预训练权重文件  " << checkpoint_path << " 不存在 !\n";
        return;
    }
    std::ifstream reader(checkpoint_path.c_str(), std::ios::binary);
    this->conv_layer_1.load_weights(reader);
    this->conv_layer_2.load_weights(reader);
    this->conv_layer_3.load_weights(reader);
    this->conv_layer_4.load_weights(reader);
    this->classifier.load_weights(reader);
    std::cout << "load weights from" << checkpoint_path.string() << std::endl;
    reader.close();
}