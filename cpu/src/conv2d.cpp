// C++
#include <iostream>
// self
#include "architectures.h"


using namespace architectures;

Conv2D::Conv2D(std::string _name, const int _in_channels, const int _out_channels, const int _kernel_size, const int _stride)
        : Layer(_name), bias(_out_channels), in_channels(_in_channels), out_channels(_out_channels), kernel_size(_kernel_size), stride(_stride),
          params_for_one_kernel(_in_channels * _kernel_size * _kernel_size),
          offset(_kernel_size * _kernel_size) {
    // 验证参数合法性
    assert(_kernel_size & 1 && _kernel_size >= 3 && "卷积核的大小必须是正奇数 !");
    assert(_in_channels > 0 && _out_channels > 0 && _stride > 0);
    // 首先给权值矩阵 weights 和偏置 b 分配空间
    this->weights.reserve(out_channels);
    for(int o = 0;o < out_channels; ++o) {
        // 一共有 out_channels 个卷积核, 每个卷积核有 in_channels X kernel_size X kernel_size 个参数
        weights.emplace_back(new Tensor3D(in_channels, kernel_size, kernel_size, this->name + "_" + std::to_string(o)));
    }
    // 随机初始化, 这里用的是正态分布初始化
    this->seed.seed(212);
    std::normal_distribution<float> engine(0.0, 1.0);
    for(int o = 0;o < out_channels; ++o) bias[o] = engine(this->seed) / random_times;
    for(int o = 0;o < out_channels; ++o) {
        data_type* data_ptr = this->weights[o]->data;
        for(int i = 0;i < params_for_one_kernel; ++i)
            data_ptr[i] = engine(this->seed) / random_times;
    }
}

// 卷积操作的 forward 过程, batch_num X in_channels X H X W
std::vector<tensor> Conv2D::forward(const std::vector<tensor>& input) {
    // 获取输入特征图的信息
    const int batch_size = input.size();
    const int H = input[0]->H;
    const int W = input[0]->W;
    const int length = H * W; // 一个二维特征图的大小, 用来算偏移量的
    // 计算输出的特征图大小
    const int out_H = std::floor((H - kernel_size - 2 * padding) / stride) + 1;
    const int out_W = std::floor((W - kernel_size - 2 * padding) / stride) + 1;
    const int out_length = out_H * out_W; // 输出的特征图, 一个通道的输出有多大, 111 X 111, 7 X 7 这种
    // 为卷积做准备
    const int radius = int((kernel_size - 1) / 2);
    // 如果是第一次经过这一层, 分配空间(这里灵活性差点, 如果形状不是一样的, 可能会崩溃, 要重新分配, 暂时不搞了)
    if(this->output.empty()) {
        // std::cout << this->name << " 第一次分配输出的向量\n";
        // 分配输出的张量
        this->output.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b)  // B X 16 X 111 X 111
            this->output.emplace_back(new Tensor3D(out_channels, out_H, out_W, this->name + "_output_" + std::to_string(b)));
        // 辅助变量 offset 只求一遍, 虽然方便, 但没有局部变量快
        int pos = 0;
        for(int x = -radius;x <= radius; ++x)
            for(int y = -radius; y <= radius; ++y) {
                this->offset[pos] = x * W + y;
                ++pos;
            }
    }
    // 记录输入, 如果存在 backward 的话, 后面 backward 算 w 的梯度要用
    if(!no_grad) this->__input = input;
    // 为卷积做准备
    const int H_radius = H - radius; // 避免每次循环重新计算 H - radius
    const int W_radius = W - radius;
    const int window_range = kernel_size * kernel_size; // 卷积核一个二维平面的大小, 用来算偏移的
    const int* const __offset = this->offset.data(); // 获取偏移量指针
    // 首先每张图像分开卷积
    for(int b = 0;b < batch_size; ++b) {
        // 获取第 b 张图像的起始地址, in_channels X 224 X 224
        data_type* const cur_image_features = input[b]->data;
        for(int o = 0;o < out_channels; ++o) { // 每个卷积核
            data_type* const out_ptr = this->output[b]->data + o * out_length;// 第 o 个卷积核会得到一张 out_H X out_W 的特征图
            data_type* const cur_w_ptr = this->weights[o]->data;  // in_channels x 3 x 3
            int cnt = 0; // 记录每次卷积结果存放的位置
            for(int x = radius; x < H_radius; x += stride) {
                for(int y = radius; y < W_radius; y += stride) { // 遍历图像平面每一个点
                    data_type sum_value = 0.f;
                    const int coord = x * W + y; // 当前点对于这个通道的特征图的位移
                    for(int i = 0;i < in_channels; ++i) { // 每个点有多个通道
                        const int start = i * length + coord; // 输入的第 i 张特征图在 (x, y) 处的位移
                        const int start_w = i * window_range; // 第 o 个卷积核的第 i 个通道
                        for(int k = 0;k < window_range; ++k)// 遍历局部窗口
                            sum_value += cur_image_features[start + __offset[k]] * cur_w_ptr[start_w + k];
                    }
                    sum_value += this->bias[o]; // 别忘记加上 b
                    out_ptr[cnt] = sum_value;   // 一次线性相乘求和的结果, 放到输出的 cnt 位置上, 按行优先存储
                    ++cnt;  // 存放的位置 + 1
                }
            } // std::cout << "cnt = " << std::sqrt(cnt) << std::endl;
        }
    }
    return this->output;  // 返回卷积结果, 在上面的 out_ptr 被更新
}

// 优化的话, 把一些堆上的数据放到栈区, 局部变量快
std::vector<tensor> Conv2D::backward(std::vector<tensor>& delta) {
    // 获取回传的梯度信息, 之前 forward 输出是多大, delta 就是多大(不考虑异常输入)
    const int batch_size = delta.size();
    const int out_H = delta[0]->H;
    const int out_W = delta[0]->W;
    const int out_length = out_H * out_W;
    // 获取之前 forward 的输入特征图信息
    const int H = this->__input[0]->H;
    const int W = this->__input[0]->W;
    const int length = H * W;
    // 第一次经过这里, 给缓冲区的梯度分配空间
    if(this->weights_gradients.empty()) {
        // weights
        this->weights_gradients.reserve(out_channels);
        for(int o = 0;o < out_channels; ++o)
            this->weights_gradients.emplace_back(new Tensor3D(in_channels, kernel_size, kernel_size, this->name + "_weights_gradients_" + std::to_string(o)));
        // bias
        this->bias_gradients.assign(out_channels, 0);
    }
    // 这里默认不记录梯度的历史信息, W, b 之前的梯度全部清空
    for(int o = 0;o < out_channels; ++o) this->weights_gradients[o]->set_zero();
    for(int o = 0;o < out_channels; ++o) this->bias_gradients[o] = 0;
    // 先求 weights, bias 的梯度
    for(int b = 0;b < batch_size; ++b) { // 这个 batch 每张图像对应一个梯度, 多个梯度取平均
        // 首先, 遍历每个卷积核
        for(int o = 0;o < out_channels; ++o) {
            // 第 b 张图像的梯度, 找到第 o 个通道的起始地址
            data_type* o_delta = delta[b]->data + o * out_H * out_W;
            // 卷积核的每个 in 通道, 分开求
            for(int i = 0;i < in_channels; ++i) {
                // 第 b 张输入，找到第 i 个通道的起始地址
                data_type* in_ptr = __input[b]->data + i * H * W;
                // 第 o 个卷积核, 找到第 i 个通道的起始地址
                data_type* w_ptr = weights_gradients[o]->data + i * kernel_size * kernel_size;
                // // 遍历的是卷积核的一个通道，求每个参数的梯度
                for(int k_x = 0; k_x < kernel_size; ++k_x) {
                    for(int k_y = 0;k_y < kernel_size; ++k_y) {
                        // 记录一张图像的 W 梯度
                        data_type sum_value = 0;
                        for(int x = 0;x < out_H; ++x) {
                            // delta 在这个通道的第 x 行，每行 out_W 个数
                            data_type* delta_ptr = o_delta + x * out_W;
                            // 对应的输入 I 在这个通道的第 (x * stride + k_x) 行， 每行 W 个数，
                            // 注意 * stride 每次在 input 中是跳着找的, + k_x 是找竖直方向上的偏移量; 下面的 y * stride + k_y 同理
                            data_type* input_ptr = in_ptr + (x * stride + k_x) * W;
                            for(int y = 0;y < out_W; ++y) {
                                // 当前 w 的梯度, 由参与计算的输入和返回的梯度相乘，累加
                                sum_value += delta_ptr[y] * input_ptr[y * stride + k_y];
                            }
                        }
                        // 更新到 weight_gradients, 注意除以了 batch_size；这里是 +=， 不是 =, 一个 batch 的梯度累加
                        w_ptr[k_x * kernel_size + k_y] += sum_value / batch_size;
                    }
                }
            }
            // 计算 b 的梯度
            data_type sum_value = 0;
            // 需要计算多个通道输出的梯度
            for(int d = 0;d < out_length; ++d) sum_value += o_delta[d];
            // 除以 batch_size
            bias_gradients[o] += sum_value / batch_size;
        }
    }
    // 接下来求输出到输入的梯度 delta_output
    // 第一次经过 backward 给 delta_output 分配内存
    if(this->delta_output.empty()) {
        this->delta_output.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b)
            this->delta_output.emplace_back(new Tensor3D(in_channels, H, W, this->name + "_delta_" + std::to_string(b)));
    }
    // delta_output 初始化为 0, 这一步不是多余的
    for(int o = 0;o < batch_size; ++o) this->delta_output[o]->set_zero();
    // 翻转 180, padding 那个太难了, 下面直接采用最笨的方法, 用卷积来定位每个输入对应的参与计算的权值 w
    const int radius = (kernel_size - 1) / 2;
    const int H_radius = H - radius;
    const int W_radius = W - radius;
    const int window_range = kernel_size * kernel_size;
    // 多个 batch 的梯度分开算
    for(int b = 0;b < batch_size; ++b) {
        // in_channels X 224 X 224
        data_type* const cur_image_features = this->delta_output[b]->data;
        // 16 个卷积核的输出, 16 x 111 x 111
        for(int o = 0;o < out_channels; ++o) { // 每个卷积核
            data_type* const out_ptr = delta[b]->data + o * out_length;// 第 o 个卷积核会得到一张 out_H X out_W 的特征图
            data_type* const cur_w_ptr = this->weights[o]->data;  // in_channels x 3 x 3
            int cnt = 0; // 记录每次卷积结果存放的位置
            for(int x = radius; x < H_radius; x += stride) {
                for(int y = radius; y < W_radius; y += stride) { // 遍历图像平面每一个点
                    // data_type sum_value = 0.f;
                    const int coord = x * W + y; // 当前点对于这个通道的特征图的位移
                    for(int i = 0;i < in_channels; ++i) { // 每个点有多个通道
                        const int start = i * length + coord; // 输入的第 i 张特征图在 (x, y) 处的位移
                        const int start_w = i * window_range; // 第 o 个卷积核的第 i 个通道
                        for(int k = 0;k < window_range; ++k) { // 遍历局部窗口
                            // sum_value += cur_image_features[start + offset[k]] * cur_w_ptr[start_w + k];
                            cur_image_features[start + offset[k]] += cur_w_ptr[start_w + k] * out_ptr[cnt];
                        }
                    }
                    ++cnt; // 用来定位当前是输出的第几个数
                }
            }
        }
    }
    // 返回
    return this->delta_output;
}

// 更新参数
void Conv2D::update_gradients(const data_type learning_rate) {
    assert(!this->weights_gradients.empty());
    // 把梯度更新到 W 和 b
    for(int o = 0;o < out_channels; ++o) {
        data_type* w_ptr = weights[o]->data;
        data_type* wg_ptr = weights_gradients[o]->data;
        // 逐个卷积核做权值更新
        for(int i = 0;i < params_for_one_kernel; ++i)
            w_ptr[i] -= learning_rate *  wg_ptr[i];
        // 更新 bias
        bias[o] -= learning_rate *  bias_gradients[o];
    }
}

// 保存权值
void Conv2D::save_weights(std::ofstream& writer) const {
    // 需要保存的是 weights, bias
    const int filter_size = sizeof(data_type) * params_for_one_kernel;
    for(int o = 0;o < out_channels; ++o)
        writer.write(reinterpret_cast<const char *>(&weights[o]->data[0]), static_cast<std::streamsize>(filter_size));
    writer.write(reinterpret_cast<const char *>(&bias[0]), static_cast<std::streamsize>(sizeof(data_type) * out_channels));
}

// 加载权值
void Conv2D::load_weights(std::ifstream& reader) {
    const int filter_size = sizeof(data_type) * params_for_one_kernel;
    for(int o = 0;o < out_channels; ++o)
        reader.read((char*)(&weights[o]->data[0]), static_cast<std::streamsize>(filter_size));
    reader.read((char*)(&bias[0]), static_cast<std::streamsize>(sizeof(data_type) * out_channels));
}


// 获取这一层卷积层的参数值
int Conv2D::get_params_num() const {
    return (this->params_for_one_kernel + 1) * this->out_channels;
}
