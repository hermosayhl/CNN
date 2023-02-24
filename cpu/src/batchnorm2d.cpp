// C++
#include <string>
#include <iostream>
// self
#include "architectures.h"

using namespace architectures;


namespace {
    inline data_type square(const data_type x) {
        return x * x;
    }
}


BatchNorm2D::BatchNorm2D(std::string _name, const int _out_channels, const data_type _eps, const data_type _momentum)
        : Layer(_name), out_channels(_out_channels), eps(_eps), momentum(_momentum),
          gamma(_out_channels, 1.0), beta(_out_channels, 0),
          moving_mean(_out_channels, 0), moving_var(_out_channels, 0),
          buffer_mean(_out_channels, 0), buffer_var(_out_channels, 0) {}


std::vector<tensor> BatchNorm2D::forward(const std::vector<tensor>& input) {
    // 获取输入信息
    const int batch_size = input.size();
    const int H = input[0]->H;
    const int W = input[0]->W;
    // 第一次经过 forward, 分配空间
    if(this->output.empty()) {
        this->output.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b) this->output.emplace_back(new Tensor3D(out_channels, H, W, this->name + "_output_" + std::to_string(b)));
        this->normed_input.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b) this->normed_input.emplace_back(new Tensor3D(out_channels, H, W, this->name + "_normed_" + std::to_string(b)));
    }
    // 记录输入
    if(!no_grad) this->__input = input;
    // 这里注意是否要置为 0
    // 开始归一化
    const int feature_map_length = H * W;  // 一张二维平面特征图的大小
    const int output_length = batch_size * feature_map_length;  // 一个输出包含的数个数
    for(int o = 0;o < out_channels; ++o) {
        // 如果是训练
        if(!no_grad) {
            // 先求均值 u
            data_type u = 0;
            for(int b = 0;b < batch_size; ++b) {
                data_type* const src_ptr = input[b]->data + o * feature_map_length; // 现在找到了第 o 个输出的第 b 张图像的相关指针
                for(int i = 0;i < feature_map_length; ++i)
                    u += src_ptr[i];
            }
            // 如果有 backward, 记住均值
            u = u / output_length;
            // 求方差 sigma
            data_type var = 0;
            for(int b = 0;b < batch_size; ++b) {
                data_type* const src_ptr = input[b]->data + o * feature_map_length;
                for(int i = 0;i < feature_map_length; ++i)
                    var += square(src_ptr[i] - u);
            }
            var = var / output_length;
            if(!no_grad) {
                buffer_mean[o] = u;
                buffer_var[o] = var;
            }
            // 对第 o 个输出做归一化
            const data_type var_inv = 1. / std::sqrt(var + eps);
            for(int b = 0;b < batch_size; ++b) {
                data_type* const src_ptr = input[b]->data + o * feature_map_length;
                data_type* const norm_ptr = normed_input[b]->data + o * feature_map_length;
                data_type* const des_ptr = output[b]->data + o * feature_map_length;
                for(int i = 0;i < feature_map_length; ++i) {
                    norm_ptr[i] = (src_ptr[i] - u) * var_inv;
                    des_ptr[i] = gamma[o] * norm_ptr[i] + beta[o];
                }
            }
            // 更新历史的均值和方差(这里要区分训练和非训练期间, 也就是 train 和 eval 的区别!!!!!!)
            moving_mean[o] = (1 - momentum) * moving_mean[o] + momentum * u;
            moving_var[o] = (1 - momentum) * moving_var[o] + momentum * var;
        }
        else {
            // 直接归一化
            const data_type u = moving_mean[o];
            const data_type var_inv = 1. / std::sqrt(moving_var[o] + eps);
            for(int b = 0;b < batch_size; ++b) {
                data_type* const src_ptr = input[b]->data + o * feature_map_length;
                data_type* const norm_ptr = normed_input[b]->data + o * feature_map_length;
                data_type* const des_ptr = output[b]->data + o * feature_map_length;
                for(int i = 0;i < feature_map_length; ++i) {
                    norm_ptr[i] = (src_ptr[i] - u) * var_inv;
                    des_ptr[i] = gamma[o] * norm_ptr[i] + beta[o];
                }
            }
        }
    }
    return this->output;
}

// batch norm 的 delta 也可以就地修改
std::vector<tensor> BatchNorm2D::backward(std::vector<tensor>& delta) {
    // 获取信息
    const int batch_size = delta.size();
    const int feature_map_length = delta[0]->H * delta[0]->W;
    const int output_length = batch_size * feature_map_length;
    // 因为是第一次返回, 所以给 gradients 分配空间
    if(gamma_gradients.empty()) {
        gamma_gradients.assign(out_channels, 0);
        beta_gradients.assign(out_channels, 0);
        norm_gradients = std::shared_ptr<Tensor3D>(new Tensor3D(batch_size, delta[0]->H, delta[0]->W));
    }
    // 每次都先清空, 不考虑历史梯度信息
    for(int o = 0;o < out_channels; ++o) gamma_gradients[o] = beta_gradients[o] = 0;
    // 从后往前推
    for(int o = 0;o < out_channels; ++o) {
        // 清空 u, var, norm 的梯度
        norm_gradients->set_zero(); // B X H X W
        // 第一个是 beta 和 gamma 比较简单, 还有 norm 的梯度
        for(int b = 0;b < batch_size; ++b) {
            data_type* const delta_ptr = delta[b]->data + o * feature_map_length;
            data_type* const norm_ptr = normed_input[b]->data + o * feature_map_length;
            data_type* const norm_g_ptr = norm_gradients->data + b * feature_map_length;
            for(int i = 0;i < feature_map_length; ++i) {
                gamma_gradients[o] += delta_ptr[i] * norm_ptr[i];
                beta_gradients[o] += delta_ptr[i];
                norm_g_ptr[i] += delta_ptr[i] * gamma[o];
            }
        }
        // 接下来, 是对方差 var 的梯度, mean 依赖于 var, 所以要先求 var 的梯度
        data_type var_gradient = 0;
        const data_type u = buffer_mean[o];
        const data_type var_inv = 1. / std::sqrt(buffer_var[o] + eps);
        const data_type var_inv_3 = var_inv * var_inv * var_inv;
        for(int b = 0;b < batch_size; ++b) {
            data_type* const src_ptr = __input[b]->data + o * feature_map_length;
            data_type* const norm_g_ptr = norm_gradients->data + b * feature_map_length;
            for(int i = 0;i < feature_map_length; ++i)
                var_gradient += norm_g_ptr[i] * (src_ptr[i] - u) * (-0.5) * var_inv_3;
        }
        // 接下来求对均值 u 的均值
        data_type u_gradient = 0;
        const data_type inv = var_gradient / output_length;
        for(int b = 0;b < batch_size; ++b) {
            data_type* const src_ptr = __input[b]->data + o * feature_map_length;
            data_type* const norm_g_ptr = norm_gradients->data + b * feature_map_length;
            for(int i = 0;i < feature_map_length; ++i)
                u_gradient += norm_g_ptr[i] * (- var_inv) + inv * (-2) * (src_ptr[i] - u);
        }
        // 最后是求返回给输入层的梯度
        for(int b = 0;b < batch_size; ++b) {
            data_type* const src_ptr = __input[b]->data + o * feature_map_length;
            data_type* const norm_g_ptr = norm_gradients->data + b * feature_map_length;
            data_type* const back_ptr = delta[b]->data + o * feature_map_length;
            for(int i = 0;i < feature_map_length; ++i)
                back_ptr[i] = norm_g_ptr[i] * var_inv + inv * 2 * (src_ptr[i] - u) + u_gradient / output_length;
        }
    }
    return delta;
}


void BatchNorm2D::update_gradients(const data_type learning_rate) {
    for(int o = 0;o < out_channels; ++o) {
        gamma[o] -= learning_rate * gamma_gradients[o];
        beta[o] -= learning_rate * beta_gradients[o];
    }
}

void BatchNorm2D::save_weights(std::ofstream& writer) const {
    const int stream_size = sizeof(data_type) * out_channels;
    writer.write(reinterpret_cast<const char *>(&gamma[0]), static_cast<std::streamsize>(stream_size));
    writer.write(reinterpret_cast<const char *>(&beta[0]), static_cast<std::streamsize>(stream_size));
    writer.write(reinterpret_cast<const char *>(&moving_mean[0]), static_cast<std::streamsize>(stream_size));
    writer.write(reinterpret_cast<const char *>(&moving_var[0]), static_cast<std::streamsize>(stream_size));
}

void BatchNorm2D::load_weights(std::ifstream& reader) {
    const int stream_size = sizeof(data_type) * out_channels;
    reader.read((char*)(&gamma[0]), static_cast<std::streamsize>(stream_size));
    reader.read((char*)(&beta[0]), static_cast<std::streamsize>(stream_size));
    reader.read((char*)(&moving_mean[0]), static_cast<std::streamsize>(stream_size));
    reader.read((char*)(&moving_var[0]), static_cast<std::streamsize>(stream_size));
}









