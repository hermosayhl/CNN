#ifndef CNN_ARCHITECTURES_H
#define CNN_ARCHITECTURES_H

#include "pipeline.h"

namespace architectures {
    using namespace pipeline;

    // 随机初始化用的, C++ 这个生成的数字过大, softmax 之前的都好几百, 直接爆了, 坑爹
    extern data_type random_times;

    // 全局变量, 是否要 backward, 访问速度上要慢一些
    extern bool no_grad;

    class Conv2D {
    private:
        // 卷积层的固有信息
        const std::string name;  // 这一层的名字
        std::vector<tensor> weights; // 卷积核的权值参数, out_channels X in_channels X kernel_size X kernel_size
        std::vector<data_type> bias; // 偏置
        const int in_channels;  // 要滤波的特征图有几个通道
        const int out_channels; // 这一层卷积有几个卷积核
        const int kernel_size;  // 卷积核的边长
        const int stride;       // 卷积的步长
        const int params_for_one_kernel;   // 一个卷积核的参数个数
        const int padding = 0;  // padding 填充量, 这个会破坏这个脆弱的程序, 还会牺牲性能, 以后有时间再说吧
        std::default_random_engine seed;   // 初始化的种子
         std::vector<int> offset; // 卷积的偏移量, 辅助用的
        // 历史信息
        std::vector<tensor> __input; // 求梯度需要, 其实存储的是指针
        // 缓冲区, 避免每次重新分配
        std::vector<tensor> output;  // 输出的张量
        std::vector<tensor> delta_output; // 存储回传到上一层的梯度
        std::vector<tensor> weights_gradients; // 权值的梯度
        std::vector<data_type> bias_gradients; // bias 的梯度
    public:
        Conv2D(std::string _name, const int _in_channels=3, const int _out_channels=16, const int _kernel_size=3, const int _stride=2);
        // 卷积操作的 forward 过程, batch_num X in_channels X H X W
        std::vector<tensor> forward(const std::vector<tensor>& input);
        // 优化的话, 把一些堆上的数据放到栈区, 局部变量快
        std::vector<tensor> backward(const std::vector<tensor>& delta);
        // 更新梯度
        void update_gradients(const data_type learning_rate=1e-4);
        // 获取这一层卷积层的参数值
        int get_params_num() const;
    };


    class MaxPool2D {
    private:
        // 这一层的固有属性
        const std::string name;   // 这一层的名字
        const int kernel_size;
        const int step;
        const int padding;
        // 缓冲区, 避免每次重新分配的
        std::vector<tensor> output;
        std::vector< std::vector<int> > mask; // 记录哪些位置是有梯度回传的, 第 b 张图, 每张图一个 std::vector<int>
        std::vector<tensor> delta_output; // 返回的 delta
        std::vector<int> offset;  // 偏移量指针, 和之前 Conv2D 的一样
    public:
        MaxPool2D(std::string _name, const int _kernel_size=2, const int _step=2)
                : name(std::move(_name)), kernel_size(_kernel_size), step(_step), padding(0),
                  offset(_kernel_size * _kernel_size, 0) {}
        // 前向
        std::vector<tensor> forward(const std::vector<tensor>& input);
        // 反向传播
        std::vector<tensor> backward(const std::vector<tensor>& delta);
    };


    class ReLU {
    private:
        // 固有属性
        std::string name;
        // 缓冲区, 避免每次重新申请
        std::vector<tensor> output;
    public:
        ReLU(std::string _name) : name(std::move(_name)) {}
        std::vector<tensor> forward(const std::vector<tensor>& input);
        std::vector<tensor> backward(std::vector<tensor>& delta);
    };


    // 线性变换层
    class LinearLayer {
    private:
        // 线性层的固有信息
        const int in_channels;                // 输入的神经元个数
        const int out_channels;               // 输出的神经元个数
        std::vector<data_type> weights;       // 权值矩阵
        std::vector<data_type> bias;          // 偏置
        // 历史信息
        std::tuple<int, int, int> delta_shape;// 记下来, delta 的形状, 从 1 X 4096 到 128 * 4 * 4 这种
        std::vector<tensor> __input;          // 梯度回传的时候需要输入 Wx + b, 需要保留 x
        // 以下是缓冲区
        std::vector<tensor1D> output;         // 记录输出
        std::vector<tensor> delta_output;     // delta 回传到输入的梯度
        std::vector<data_type> weights_gradients; // 缓存区, 权值矩阵的梯度
        std::vector<data_type> bias_gradients;    // bias 的梯度
    public:
        LinearLayer(std::string _name, const int _in_channels, const int _out_channels);
        // 做 Wx + b 矩阵运算
        std::vector<tensor1D> forward(const std::vector<tensor>& input);
        std::vector<tensor> backward(const std::vector<tensor1D>& delta);
        void update_gradients(const data_type learning_rate=1e-4);
    };


    // 记得最后开 O1 优化
    class AlexNet {
    private:
        Conv2D conv_layer_1 = Conv2D("conv_layer_1", 3, 16, 3);
        Conv2D conv_layer_2 = Conv2D("conv_layer_2", 16, 32, 3);
        Conv2D conv_layer_3 = Conv2D("conv_layer_3", 32, 64, 3);
        Conv2D conv_layer_4 = Conv2D("conv_layer_4", 64, 128, 3);
        MaxPool2D max_pool_1 = MaxPool2D("max_pool_1", 2, 2);
        ReLU relu_layer_1 = ReLU("relu_layer_1");
        ReLU relu_layer_2 = ReLU("relu_layer_2");
        ReLU relu_layer_3 = ReLU("relu_layer_3");
        ReLU relu_layer_4 = ReLU("relu_layer_4");
        LinearLayer classifier;
    public:
        AlexNet(const int num_classes=3);
        // 前向
        std::vector<tensor1D> forward(const std::vector<tensor>& input);
        // 梯度反传
        void backward(const std::vector<tensor1D>& delta_start);
        // 梯度更新到权值
        void update_gradients(const data_type learning_rate=1e-4);
    };

    // 这个 BatchNorm 是不同通道做, 不具体实现还真不知道, 后面有时间填坑
    class BatchNorm2D {
    private:
        data_type gamma;
    };
}



#endif //CNN_ARCHITECTURES_H
