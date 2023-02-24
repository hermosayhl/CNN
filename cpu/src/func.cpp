// self
#include "func.h"



namespace {
    inline data_type __exp(const data_type x) {
        if(x >= 88) return FLT_MAX; // 直接返回 float 的最大值, 如果 data_type 换成 double 这个还得改
        else if(x <= -50) return 0.f;
        return std::exp(x);
    }
}


// 给 batch_size 个向量, 每个向量 softmax 成多类别的概率
std::vector<tensor> softmax(const std::vector<tensor>& input) {
    const int batch_size = input.size();
    const int num_classes = input[0]->get_length();
    std::vector<tensor> output;
    output.reserve(batch_size);
    for(int b = 0;b < batch_size; ++b) {
        tensor probs(new Tensor3D(num_classes)); // 如果大于 1000 类这种就有浪费
        // 首先算出输出的最大值, 防止溢出, 还是改变不了什么, 大于 -37 直接等于 1, 这样并不能解决问题, 欸
        const data_type max_value = input[b]->max();
        data_type sum_value = 0;
        for(int i = 0;i < num_classes; ++i) {
            probs->data[i] = __exp(input[b]->data[i] - max_value);
            sum_value += probs->data[i];
        }
        // 概率之和 = 1
        for(int i = 0;i < num_classes; ++i) probs->data[i] /= sum_value;
        // 去掉一些 nan
        for(int i = 0;i < num_classes; ++i) if(std::isnan(probs->data[i])) probs->data[i] = 0.f;
        output.emplace_back(std::move(probs));
    }
    return output;
}

// batch_size 个样本, 每个样本 0, 1, 2 这种, 例如  1 就得到 [0.0, 1.0, 0.0, 0.0]
std::vector<tensor> one_hot(const std::vector<int>& labels, const int num_classes) {
    const int batch_size = labels.size();
    std::vector<tensor> one_hot_code;
    one_hot_code.reserve(batch_size);
    for(int b = 0;b < batch_size; ++b) {
        tensor sample(new Tensor3D(num_classes));
        for(int i = 0;i < num_classes; ++i)
            sample->data[i] = 0;
        assert(labels[b] >= 0 && labels[b] < num_classes);
        sample->data[labels[b]] = 1.0;
        one_hot_code.emplace_back(sample);
    }
    return one_hot_code;
}

// 给输出概率 probs, 和标签 label 计算交叉熵损失, 返回损失值和回传的梯度
std::pair<data_type, std::vector<tensor> > cross_entroy_backward(
        const std::vector<tensor>& probs, const std::vector<tensor>& labels) {
    const int batch_size = labels.size();
    const int num_classes = probs[0]->get_length();
    std::vector<tensor> delta;
    delta.reserve(batch_size);
    data_type loss_value = 0;
    for(int b = 0;b < batch_size; ++b) {
        tensor piece(new Tensor3D(num_classes));
        for(int i = 0;i < num_classes; ++i) {
            piece->data[i] = probs[b]->data[i] - labels[b]->data[i];
            loss_value += std::log(probs[b]->data[i]) * labels[b]->data[i];
        }
        delta.emplace_back(piece);
    }
    loss_value = loss_value * (-1.0) / batch_size;
    return std::make_pair(loss_value, delta);
}

// 小数变成 string
std::string float_to_string(const float value, const int precision) {
    std::stringstream buffer;
	buffer.precision(precision);
	buffer.setf(std::ios::fixed);
	buffer << value;
	return buffer.str();
}