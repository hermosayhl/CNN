// C++
// self
#include "architectures.h"


using namespace architectures;


std::vector<tensor>  ReLU::forward(const std::vector<tensor>& input) {
    // 获取图像信息
    const int batch_size = input.size();
    // 如果是第一次经过这一层
    if(output.empty()) {
        // 给输出分配空间
        this->output.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b)
            this->output.emplace_back(new Tensor3D(input[0]->C, input[0]->H, input[0]->W, this->name + "_output_" + std::to_string(b)));
    }
    // 只保留 > 0 的部分
    const int total_length = input[0]->get_length();
    for(int b = 0;b < batch_size; ++b) {
        data_type* const src_ptr = input[b]->data;
        data_type* const out_ptr = this->output[b]->data;
        for(int i = 0;i < total_length; ++i)
            out_ptr[i] = src_ptr[i] >= 0 ? src_ptr[i] : 0;
    }
    return this->output;
}

std::vector<tensor> ReLU::backward(std::vector<tensor>& delta) { // 这个没有 delta_output, 因为形状一模一样, 可以减少一些空间使用, 但为了多态要统一
    // 获取信息
    const int batch_size = delta.size();
    // 从这一层的输出中,  < 0 的部分过滤掉
    const int total_length = delta[0]->get_length();
    for(int b = 0;b < batch_size; ++b) {
        data_type* src_ptr = delta[b]->data;
        data_type* out_ptr = this->output[b]->data;
        for(int i = 0;i < total_length; ++i)
            src_ptr[i] = out_ptr[i] <= 0 ? 0 : src_ptr[i]; // 输出 > 0 的才有梯度从输出 src_ptr 传到输入
    }
    // 改下名字, 方便观察
    for(int b = 0;b < batch_size; ++b) delta[b]->name = this->name + "_delta_" + std::to_string(b);
    return delta;
}
