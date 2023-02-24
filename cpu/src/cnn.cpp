//C++
#include <vector>
#include <memory>
#include <iostream>
#include <filesystem>
// self
#include "func.h"
#include "metrics.h"
#include "architectures.h"




// 完全可复现, 随机种子定了
// 还需要实现的功能
// 1. 模型参数的存储和加载  OK
// 2. 动量, Adam 这些, 暂时没想到优雅的解决办法
// 3. batch norm 的实现  OK, 但测试阶段效果很差
// 4. dropout 的实现   OK, 但测试阶段效果很差
// 5. 网络结构有点差劲, 虽然可以跑, 凑合用
// 6. 自动求导, 重头戏, 有时间再说
// 7. 後面有時間加上 AvgPool2D、Global Pool 等组件
// 8. 目前的卷积层无法加上 padding
// 9. 混淆矩阵没有写, 还有没有统计历史的损失画图什么的, 有些麻烦了, 暂时不搞了


int main() {

    std::setbuf(stdout, 0);

    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    using namespace architectures;

    // 指定一些参数
    const int train_batch_size = 4;
    const int valid_batch_size = 1;
    const int test_batch_size = 1;
    assert(train_batch_size >= valid_batch_size && train_batch_size >= test_batch_size); // 设计问题, train 的 batch 必须更大
    assert(valid_batch_size == 1 && test_batch_size == 1); // 设计问题, 暂时只支持这个
    const std::tuple<int, int, int> image_size({224, 224, 3});
    const std::filesystem::path dataset_path("../../datasets/animals");
    const std::vector<std::string> categories({"dog", "panda", "bird"});

    // 获取图片
    auto dataset = pipeline::get_images_for_classification(dataset_path, categories);

    // 构造数据流
    pipeline::DataLoader train_loader(dataset["train"], train_batch_size, false, true, image_size);
    pipeline::DataLoader valid_loader(dataset["valid"], valid_batch_size, false, false, image_size);

    // 定义网络结构
    const int num_classes = categories.size(); // 分类的数目
    AlexNet network(num_classes, false);

    // 直接加载
    // network.load_weights("../checkpoints/AlexNet_aug_2e-4/iter_100000_train_0.846_valid_0.863.model");

    // 保存
    const std::filesystem::path checkpoints_dir("../checkpoints/AlexNet_aug_1e-3");
    if(!std::filesystem::exists(checkpoints_dir))
        std::filesystem::create_directories(checkpoints_dir);
    std::filesystem::path best_checkpoint;  // 当前正确率最高的模型
    float current_best_accuracy = -1; // 记录当前最高的正确率

    // 开始训练
    const int start_iters = 1;        // 从第几个 iter 开始
    const int total_iters = 400000;   // 训练 batch 的总数
    const float learning_rate = 1e-3; // 学习率
    const int valid_inters = 1000;    // 验证一次的间隔
    const int save_iters = 5000;      // 保存模型的间隔
    float mean_loss = 0.f;            // 平均损失
    float cur_iter = 0;               // 计算平均损失用的
    ClassificationEvaluator train_evaluator;  // 计算累计的准确率
    std::vector<int> predict(train_batch_size, -1); // 存储每个 batch 的预测结果, 和 labels 算准确率用的
    // 开始训练
    for(int iter = start_iters; iter <= total_iters; ++iter) {
        // 从训练集中采样一个 batch
        const auto sample = train_loader.generate_batch();
        // 送到网络中
        const auto output = network.forward(sample.first);
        // 网络输出经过 softmax 转化成概率
        const auto probs = softmax(output);
        // 输出概率和标签计算交叉熵损失, 返回损失项和梯度
        auto loss_delta = cross_entroy_backward(probs, one_hot(sample.second, num_classes));
        mean_loss += loss_delta.first;
        // 根据损失, 回传梯度
        network.backward(loss_delta.second);
        // 更新权值
        network.update_gradients(learning_rate);
        // 根据 predict 和 label 计算准确率
        for(int b = 0;b < train_batch_size; ++b) predict[b] = probs[b]->argmax(); // 概率最大的下标作为分类
        train_evaluator.compute(predict, sample.second);
        // 打印信息
        ++cur_iter;
        printf("\rTrain===> [batch %d/%d] [loss %.3f] [Accuracy %4.3f]", iter, total_iters, mean_loss / cur_iter, train_evaluator.get());
        // 更新一次信息
        if(iter % valid_inters == 0) {
            printf("开始验证.....\n");
            WithoutGrad guard;  // 暂时关闭中间的梯度计算
            float mean_valid_loss = 0.f;
            ClassificationEvaluator valid_evaluator;  // 衡量 valid 期间的分类性能
            const int samples_num = valid_loader.length();  // 目前只支持 batch_size = 1
            for(int s = 1;s <= samples_num; ++s) {
                const auto sample = valid_loader.generate_batch();
                const auto output = network.forward(sample.first);
                const auto probs = softmax(output);
                const auto loss_delta = cross_entroy_backward(probs, one_hot(sample.second, num_classes));
                mean_valid_loss += loss_delta.first;
                for(int b = 0;b < train_batch_size; ++b) predict[b] = probs[b]->argmax(); // 概率最大的下标作为分类
                valid_evaluator.compute(predict, sample.second);
                printf("\rValid===> [batch %d/%d] [loss %.3f] [Accuracy %4.3f]", s, samples_num, mean_valid_loss / s, valid_evaluator.get());
            }
            printf("\n\n");
            // 保存模型
            if(iter % save_iters == 0) {
                // 查看当前性能
                const float train_accuracy = train_evaluator.get();
                const float valid_accuracy = valid_evaluator.get();
                // 决定保存的名字
                std::string save_string("iter_" + std::to_string(iter));
                save_string +=  "_train_" + float_to_string(train_accuracy, 3);
                save_string +=  "_valid_" + float_to_string(valid_accuracy, 3) + ".model";
                std::filesystem::path save_path = checkpoints_dir / save_string;
                // 保存权值
                network.save_weights(save_path);
                // 记录最佳的正确率和对应的路径
                if(valid_accuracy > current_best_accuracy) {
                    best_checkpoint = save_path;
                    current_best_accuracy = valid_accuracy;
                }
            }
            // 更新一波训练的信息
            cur_iter = 0;
            mean_loss = 0.f;
            train_evaluator.clear();
        }
    }
    std::cout << "训练结束!\n";

    {
        // 加载模型, 在测试集上做
        network.load_weights(best_checkpoint);
        // 准备测试数据
        pipeline::DataLoader test_loader(dataset["test"], test_batch_size, false, false, image_size);
        // 循环
        WithoutGrad guard;  // 暂时关闭中间的梯度计算, return 0 之前才恢复
        float mean_test_loss = 0.f;
        ClassificationEvaluator test_evaluator;  // 衡量 test 期间的分类性能
        const int samples_num = test_loader.length();  // 目前只支持 batch_size = 1
        for(int s = 1;s <= samples_num; ++s) {
            const auto sample = test_loader.generate_batch();
            const auto output = network.forward(sample.first);
            const auto probs = softmax(output);
            const auto loss_delta = cross_entroy_backward(probs, one_hot(sample.second, num_classes));
            mean_test_loss += loss_delta.first;
            for(int b = 0;b < train_batch_size; ++b) predict[b] = probs[b]->argmax(); // 概率最大的下标作为分类
            test_evaluator.compute(predict, sample.second);
            printf("\rTest===> [batch %d/%d] [loss %.3f] [Accuracy %4.3f]", s, samples_num, mean_test_loss / s, test_evaluator.get());
        }
    }
    return 0;
}
