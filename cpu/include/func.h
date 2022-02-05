#ifndef CNN_FUNC_H
#define CNN_FUNC_H

#include "data_format.h"


// 给 batch_size 个向量, 每个向量 softmax 成多类别的概率
std::vector<tensor> softmax(const std::vector<tensor>& input);

// batch_size 个样本, 每个样本 0, 1, 2 这种, 例如  1 就得到 [0.0, 1.0, 0.0, 0.0]
std::vector<tensor> one_hot(const std::vector<int>& labels, const int num_classes);

// 给输出概率 probs, 和标签 label 计算交叉熵损失, 返回损失值和回传的梯度
std::pair<data_type, std::vector<tensor> > cross_entroy_backward(
        const std::vector<tensor>& probs, const std::vector<tensor>& labels);

// 小数变成 string
std::string float_to_string(const float value, const int precision);

#endif //CNN_FUNC_H
