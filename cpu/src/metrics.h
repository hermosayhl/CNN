#ifndef CNN_METRICS_H
#define CNN_METRICS_H

// C++
#include <vector>


class ClassificationEvaluator {
private:
    int correct_num = 0;  // 当前累计的判断正确的样本数目
    int sample_num = 0;   // 当前累计的样本数目
public:
    ClassificationEvaluator() = default;
    // 这一个 batch 猜对了几个
    void compute(const std::vector<int>& predict, const std::vector<int>& labels);
    // 查看累计的正确率
    float get() const;
    // 重新开始统计
    void clear();
};



#endif //CNN_METRICS_H
