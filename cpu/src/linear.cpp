// C++
#include <vector>
#include <random>
// self
#include "architectures.h"

using namespace architectures;

LinearLayer::LinearLayer(std::string _name, const int _in_channels, const int _out_channels)
        : Layer(_name), in_channels(_in_channels), out_channels(_out_channels),
          weights(_in_channels * _out_channels, 0),
          bias(_out_channels) {
    // 随机种子初始化
    std::default_random_engine e(1998);
    std::normal_distribution<float> engine(0.0, 1.0);
    for(int i = 0;i < _out_channels; ++i) bias[i] = engine(e) / random_times;
    const int length = _in_channels * _out_channels;
    for(int i = 0;i < length; ++i) weights[i] = engine(e) / random_times;
}

// 做 Wx + b 矩阵运算
std::vector<tensor> LinearLayer::forward(const std::vector<tensor>& input) {
    // 获取输入信息
    const int batch_size = input.size();
    this->delta_shape = input[0]->get_shape();
    // 清空之前的结果, 重新开始
    std::vector<tensor>().swap(this->output);
    for(int b = 0;b < batch_size; ++b)
        this->output.emplace_back(new Tensor3D(out_channels, this->name + "_output_" + std::to_string(b)));
    // 记录输入
    if(!no_grad) this->__input = input;
    // batch 每个图象分开算
    for(int b = 0;b < batch_size; ++b) {
        // 矩阵相乘,   dot
        data_type* src_ptr = input[b]->data; // 1 X 4096
        data_type* res_ptr = this->output[b]->data; // 1 X 10
        for(int i = 0;i < out_channels; ++i) {
            data_type sum_value = 0;
            for(int j = 0;j < in_channels; ++j)
                sum_value += src_ptr[j] * this->weights[j * out_channels + i];
            res_ptr[i] = sum_value + bias[i];
        }
    }
    return this->output;
}

std::vector<tensor> LinearLayer::backward(std::vector<tensor>& delta) {
    // 获取 delta 信息
    const int batch_size = delta.size();
    // 第一次回传, 给缓冲区的梯度 W, b 分配空间
    if(this->weights_gradients.empty()) {
        this->weights_gradients.assign(in_channels * out_channels, 0);
        this->bias_gradients.assign(out_channels, 0);
    }
    // 计算 W 的梯度
    for(int i = 0;i < in_channels; ++i) {
        data_type* w_ptr = this->weights_gradients.data() + i * out_channels;
        for(int j = 0;j < out_channels; ++j) {
            data_type sum_value = 0;
            for(int b = 0;b < batch_size; ++b)
                sum_value += this->__input[b]->data[i] * delta[b]->data[j];
            w_ptr[j] = sum_value / batch_size;
        }
    }
    // 计算 bias 的梯度
    for(int i = 0;i < out_channels; ++i) {
        data_type sum_value = 0;
        for(int b = 0;b < batch_size; ++b)
            sum_value += delta[b]->data[i];
        this->bias_gradients[i] = sum_value / batch_size;
    }
    // 如果是第一次回传
    if(this->delta_output.empty()) {
        // 分配空间
        this->delta_output.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b)
            this->delta_output.emplace_back(new Tensor3D(delta_shape, "linear_delta_" + std::to_string(b)));
    }
    // 计算返回的梯度, 大小和 __input 一致
    for(int b = 0;b < batch_size; ++b) {  // 每个 batch
        data_type* src_ptr = delta[b]->data;
        data_type* res_ptr = this->delta_output[b]->data;
        for(int i = 0;i < in_channels; ++i) {  // 每个输入神经元
            data_type sum_value = 0;
            data_type* w_ptr = this->weights.data() + i * out_channels;
            for(int j = 0;j < out_channels; ++j)  // 每个输出都由第 i 个神经元参与计算得到
                sum_value += src_ptr[j] * w_ptr[j];
            res_ptr[i] = sum_value;
        }
    }
    // 返回到上一层给的梯度
    return this->delta_output;
}

void LinearLayer::update_gradients(const data_type learning_rate) {
    // 这里要判断一下, 是否空的
    assert(!this->weights_gradients.empty());
    // 梯度更新到权值
    const int total_length = in_channels * out_channels;
    for(int i = 0;i < total_length; ++i) this->weights[i] -= learning_rate *  this->weights_gradients[i];
    for(int i = 0;i < out_channels; ++i) this->bias[i] -= learning_rate *  this->bias_gradients[i];
}

// 保存权值
void LinearLayer::save_weights(std::ofstream& writer) const {
    writer.write(reinterpret_cast<const char *>(&weights[0]), static_cast<std::streamsize>(sizeof(data_type) * in_channels * out_channels));
    writer.write(reinterpret_cast<const char *>(&bias[0]), static_cast<std::streamsize>(sizeof(data_type) * out_channels));
}

// 加载权值
void LinearLayer::load_weights(std::ifstream& reader) {
    reader.read((char*)(&weights[0]), static_cast<std::streamsize>(sizeof(data_type) * in_channels * out_channels));
    reader.read((char*)(&bias[0]), static_cast<std::streamsize>(sizeof(data_type) * out_channels));
}