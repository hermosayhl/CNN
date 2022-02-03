// self
#include "metrics.h"


// 这一个 batch 猜对了几个
void ClassificationEvaluator::compute(const std::vector<int>& predict, const std::vector<int>& labels) {
    const int batch_size = labels.size();  // 这里不能是 predict 的 size, 程序设计问题, 没办法
    for(int b = 0;b < batch_size; ++b)
        if(predict[b] == labels[b])
            ++this->correct_num;
    this->sample_num += batch_size;
}
// 查看累计的正确率
float ClassificationEvaluator::get() const {
    return this->correct_num * 1.f / this->sample_num;
}
// 重新开始统计
void ClassificationEvaluator::clear() {
    this->correct_num = this->sample_num = 0;
}