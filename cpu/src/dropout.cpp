// self
#include "architectures.h"

using namespace architectures;


std::vector<tensor> Dropout::forward(const std::vector<tensor>& input) {
    // 获取信息
    const int batch_size = input.size();
    const int out_channels = input[0]->C;
    const int area = input[0]->H * input[0]->W;
    // 定义第一次的变量
    if(this->sequence.empty()) {
        this->sequence.assign(out_channels, 0);
        for(int o = 0;o < out_channels; ++o) this->sequence[o] = o;
        this->selected_num = int(p * out_channels); // 失活的卷积核个数
        assert(out_channels > this->selected_num);
        this->mask.assign(out_channels, 0);
        // 给输出分配
        this->output.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b) this->output.emplace_back(new Tensor3D(out_channels, input[0]->H, input[0]->W));
    }
    // 打乱列表
    std::shuffle(this->sequence.begin(), this->sequence.end(), this->drop);
    // 如果是训练阶段
    if(!no_grad) {
        // 记录被选中的卷积核(输出通道), 前 selected_num 个失活, 其它置为 -1
        for(int i = 0;i < out_channels; ++i)
            this->mask[i] = i >= selected_num ? this->sequence[i] : -1;
        const auto copy_size = sizeof(data_type) * area;
        for(int b = 0;b < batch_size; ++b) {
            for(int o = 0;o < out_channels; ++o) { // 考察 o 通道
                if(o >= this->selected_num)  // 如果是保留的神经元
                    std::memcpy(this->output[b]->data + o * area, input[b]->data + o * area, copy_size);
                else std::memset(this->output[b]->data + o * area, 0, copy_size);
            }
        }
    }
    else { // 验证或者测试阶段
        // 直接将结果乘以 1 - p, 选中了 1 - p 个输出
        const int length = input[0]->get_length();
        const data_type prob = 1 - this->p;
        for(int b = 0;b < batch_size; ++b) {
            data_type* const src_ptr = input[b]->data;
            data_type* const des_ptr = this->output[b]->data;
            for(int i = 0;i < length; ++i) des_ptr[i] = src_ptr[i] * prob;
        }
    }
    return this->output;
}

std::vector<tensor> Dropout::backward(std::vector<tensor>& delta) {
    // 获取信息
    const int batch_size = delta.size();
    const int out_channels = delta[0]->C;
    const int area = delta[0]->H * delta[0]->W;
    // 根据 mask 来操作
    for(int b = 0;b < batch_size; ++b)
        for(int o = 0;o < out_channels; ++o)
            if(this->mask[o] == -1) // 如果这个卷积核失活了, 梯度不传回去, 置为 0
                std::memset(delta[b]->data + o * area, 0, sizeof(data_type) * area);
    return delta;
}