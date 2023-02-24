// C++
#include <iostream>
// self
#include "architectures.h"

using namespace architectures;



AlexNet::AlexNet(const int num_classes, const bool batch_norm)   {
    // batch_size X 3 X 224 X 224
    this->layers_sequence.emplace_back(new Conv2D("conv_layer_1", 3, 16, 3));
    if(batch_norm) this->layers_sequence.emplace_back(new BatchNorm2D("bn_layer_1", 16));
    this->layers_sequence.emplace_back(new ReLU("relu_layer_1"));
    // batch_size X 16 X 111 X 111
    this->layers_sequence.emplace_back(new MaxPool2D("max_pool_1", 2, 2));
    // batch_size X 16 X 55 X 55
    this->layers_sequence.emplace_back(new Conv2D("conv_layer_2", 16, 32, 3));
    if(batch_norm) this->layers_sequence.emplace_back(new BatchNorm2D("bn_layer_2", 32));
    this->layers_sequence.emplace_back(new ReLU("relu_layer_2"));
    // batch_size X 32 X 27 X 27
    this->layers_sequence.emplace_back(new Conv2D("conv_layer_3", 32, 64, 3));
    if(batch_norm) this->layers_sequence.emplace_back(new BatchNorm2D("bn_layer_3", 64));
    this->layers_sequence.emplace_back(new ReLU("relu_layer_3"));
    // batch_size X 64 X 13 X 13
    this->layers_sequence.emplace_back(new Conv2D("conv_layer_4", 64, 128, 3));
    if(batch_norm) this->layers_sequence.emplace_back(new BatchNorm2D("bn_layer_4", 128));
    // this->layers_sequence.emplace_back(new Dropout("dropout_layer_1", 0.4));
    this->layers_sequence.emplace_back(new ReLU("relu_layer_4"));
    // batch_size X 128 X 6 X 6
    this->layers_sequence.emplace_back(new LinearLayer("linear_1", 6 * 6 * 128, num_classes));
    // batch_size X num_classes
}

std::vector<tensor> AlexNet::forward(const std::vector<tensor>& input) {
    // 对输入的形状做检查
    assert(input.size() > 0);
    if(this->print_info) input[0]->print_shape();

    std::vector<tensor> output(input);
    for(const auto& layer : this->layers_sequence) {
        output = layer->forward(output);
        if(this->print_info) output[0]->print_shape();
    }
    return output;
}

// 梯度反传
void AlexNet::backward(std::vector<tensor>& delta_start) {

    if(this->print_info) delta_start[0]->print_shape();

    for(auto layer = layers_sequence.rbegin(); layer != layers_sequence.rend(); ++layer) {

        delta_start = (*layer)->backward(delta_start);

        if(this->print_info) delta_start[0]->print_shape();
    }
}

// 这种写法灵活性差点, 新添加一层要改动很多; 后面可以考虑用多态, 存储指针试试
void AlexNet::update_gradients(const data_type learning_rate) {
    for(auto& layer : this->layers_sequence)
        layer->update_gradients(learning_rate);
}


// 保存模型权值, 灵活性很差
void AlexNet::save_weights(const std::filesystem::path& save_path) const {
    // 首先明确, 需要保存权值的只有 Conv2d, linear, batchnorm2D 这些
    std::ofstream writer(save_path.c_str(), std::ios::binary);
    // 首先这里本来应该写一下有哪些组件, 然后写一下组件的具体信息, 但是免了
    for(const auto& layer : this->layers_sequence)
        layer->save_weights(writer);
    std::cout << "weights have been saved to " << save_path.string() << std::endl;
    writer.close();
}

// 加载模型权值
void AlexNet::load_weights(const std::filesystem::path& checkpoint_path) {
    if(!std::filesystem::exists(checkpoint_path)) {
        std::cout << "预训练权重文件  " << checkpoint_path << " 不存在 !\n";
        return;
    }
    std::ifstream reader(checkpoint_path.c_str(), std::ios::binary);
    for(auto& layer : this->layers_sequence)
        layer->load_weights(reader);
    std::cout << "load weights from" << checkpoint_path.string() << std::endl;
    reader.close();
}



// GradCAM 可视化
cv::Mat AlexNet::grad_cam(const std::string& layer_name) const {
    // 首先默认经过了 forward, 从 softmax 之前开始
    std::vector<tensor> delta = layers_sequence.back()->get_output();
    auto layer = layers_sequence.rbegin();
    for(; layer != layers_sequence.rend(); ++layer) {
        if((*layer)->name == layer_name) break;
        delta = (*layer)->backward(delta);
    }
    // delta[0]->print_shape();
    // 获得最后一层卷积层输出的特征图
    const auto& feature_map = (*layer)->get_output();
    // 获取信息
    const int batch_size = feature_map.size();
    const int out_channels = feature_map[0]->C;
    const int area = feature_map[0]->H * feature_map[0]->W;
    // 首先给梯度, 每个特征维度求均值, batch_size X 128 X 6 X 6 -> batch_size X 128
    std::vector<data_type> weights(batch_size * out_channels, 0);
    for(int b = 0;b < batch_size; ++b) {
        for(int o = 0;o < out_channels; ++o) {
            data_type* const fea_ptr = feature_map[b]->data + o * area;
            data_type mean_value = 0;
            for(int i = 0;i < area; ++i) mean_value += fea_ptr[i];
            weights[b * out_channels + o] = mean_value / area;
        }
    }
    // 接下来, batch_size X 128 乘以 batch_size X 128 X 6 X 6 -> 6x6
    tensor cam(new Tensor3D(batch_size, feature_map[0]->H, feature_map[0]->W));
    cam->set_zero();
    for(int b = 0;b < batch_size; ++b) {
        data_type* const cam_ptr = cam->data + b * area;
        for(int o = 0;o < out_channels; ++o) {
            const data_type w = weights[b * out_channels + o];
            data_type* const fea_ptr = feature_map[b]->data + o * area;
            for(int i = 0;i < area; ++i)
                cam_ptr[i] += w * fea_ptr[i];
        }
    }
    // 做 relu 操作
    const int length = cam->get_length();
    for(int i = 0;i < length; ++i) if(cam->data[i] < 0) cam->data[i] = 0;
    // 对图像做最大最小值归一化
    const data_type min_value = cam->min();
    const data_type max_value = cam->max();
    const data_type res_value = max_value - min_value;
    for(int i = 0;i < length; ++i) cam->data[i] = (cam->data[i] - min_value) / res_value;
    // 返回图像
    return cam->opecv_mat(1);
}