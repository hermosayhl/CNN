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