// self
#include "architectures.h"


using namespace architectures;

std::vector<tensor> MaxPool2D::forward(const std::vector<tensor>& input) {
    // 获取输入信息
    const int batch_size = input.size();
    const int H = input[0]->H;
    const int W = input[0]->W;
    const int C = input[0]->C;
    // 计算输出的大小
    const int out_H = std::floor(((H - kernel_size + 2 * padding) / step)) + 1;
    const int out_W = std::floor(((W - kernel_size + 2 * padding) / step)) + 1;
    // 第一次经过该池化层(同样 batch_size 如果变得更大, 这个会出问题, 要重新申请)
    if(this->output.empty()) {
        // 给输出分配空间
        this->output.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b)
            this->output.emplace_back(new Tensor3D(C, out_H, out_W, this->name + "_output_" + std::to_string(b)));
        // 给反向传播的 delta 分配空间
        if(!no_grad) {
            this->delta_output.reserve(batch_size);
            for(int b = 0;b < batch_size; ++b)
                this->delta_output.emplace_back(new Tensor3D(C, H, W, this->name + "_delta_" + std::to_string(b)));
            // mask 对 batch 的每一张图都分配空间
            this->mask.reserve(batch_size);
            for(int b = 0;b < batch_size; ++b)
                this->mask.emplace_back(std::vector<int>(C * out_H * out_W, 0));
        }
        // 第一次经过这一层, 根据 kernel_size 计算 offset
        int pos = 0;
        for(int i = 0;i < kernel_size; ++i)
            for(int j = 0;j < kernel_size; ++j)
                this->offset[pos++] = i * W + j;
    }
    // 如果存在 backward, 每次 forward 要记得把 mask 全部填充为 0
    const int out_length = out_H * out_W;
    int* mask_ptr = nullptr;
    if(!no_grad) {
        const int mask_size = C * out_length;
        for(int b = 0;b < batch_size; ++b) {
            int* const mask_ptr = this->mask[b].data();
            for(int i = 0;i < mask_size; ++i) mask_ptr[i] = 0;
        }
    }
    // 开始池化
    const int length = H * W;
    const int H_kernel = H - kernel_size;
    const int W_kernel = W - kernel_size;
    const int window_range = kernel_size * kernel_size;
    for(int b = 0;b < batch_size; ++b) { // batch 的每一张图像对应的特征图分开池化
        // 16 X 111 X 111 → 16 X 55 X 55
        for(int i = 0;i < C; ++i) {  // 每个通道
            // 现在我拿到了第 b 张图的第 i 个通道, 一个指向内容大小 55 X 55 的指针
            data_type* const cur_image_features = input[b]->data + i * length;
            // 第 b 个输出的第 i 个通道的, 同样是指向内容大小 55 X 55 的指针
            data_type* const output_ptr = this->output[b]->data + i * out_length;
            // 记录第 b 个输出, 记录有效点在 111 X 111 这个图上的位置, 一共有 55 X 55 个值
            if(!no_grad) mask_ptr = this->mask[b].data() + i * out_length;
            int cnt = 0;  // 当前池化输出的位置
            for(int x = 0; x <= H_kernel; x += step) {
                data_type* const row_ptr = cur_image_features + x * W; // 获取这个通道图像的第 x 行指针
                for(int y = 0; y <= W_kernel; y += step) {
                    // 找到局部的 kernel_size X kernel_size 的区域, 找最大值
                    data_type max_value = row_ptr[y];
                    int max_index = 0; // 记录最大值的位置
                    for(int k = 1; k < window_range; ++k) { // 从 1 开始因为 0 已经比过了, max_value = row_ptr[y]
                        data_type comp = row_ptr[y + offset[k]];
                        if(max_value < comp) {
                            max_value = comp;
                            max_index = offset[k];
                        }
                    }
                    // 局部最大值填到输出的对应位置上
                    output_ptr[cnt] = max_value;
                    // 如果后面要 backward, 记录 mask
                    if(!no_grad) {
                        max_index += x * W + y; // 第 i 个通道, i * out_H * out_W 为起点的二维平面, 偏移量 max_index
                        mask_ptr[cnt] = i * length + max_index;
                    }
                    ++cnt;
                }
            } // if(this->name == "max_pool_2" and b == 0 and i == 0)
        }
    }
    return this->output;
}

// 反向传播
std::vector<tensor> MaxPool2D::backward(std::vector<tensor>& delta) {
    // 获取输入的梯度信息
    const int batch_size = delta.size();
    // B X 128 X 6 X 6, 先填 0
    for(int b = 0;b < batch_size; ++b) this->delta_output[b]->set_zero();
    // batch 每张图像, 根据 mask 标记的位置, 把 delta 中的值填到 delta_output 中去
    const int total_length = delta[0]->get_length();
    for(int b = 0;b < batch_size; ++b) {
        int* mask_ptr = this->mask[b].data();
        // 获取 delta 第 b 张输出传回来的梯度起始地址
        data_type* const src_ptr = delta[b]->data;
        // 获取返回到输入的梯度, 第 b 张梯度的起始地址
        data_type* const res_ptr = this->delta_output[b]->data;
        for(int i = 0;i < total_length; ++i)
            res_ptr[mask_ptr[i]] = src_ptr[i]; // res_ptr 在有效位置 mask_ptr[i] 上填“输出返回来的梯度” src_ptr[i]
    }
    return this->delta_output;
}