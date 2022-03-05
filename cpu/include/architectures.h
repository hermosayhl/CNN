#ifndef CNN_ARCHITECTURES_H
#define CNN_ARCHITECTURES_H


// C++
#include <list>
#include <fstream>
// self
#include "pipeline.h"


namespace architectures {
    using namespace pipeline;

    // 随机初始化用的, C++ 这个生成的数字过大, softmax 之前的都好几百, 直接爆了, 坑爹
    extern data_type random_times;

    // 全局变量, 是否要 backward, 访问速度上要慢一些
    extern bool no_grad;

    // 在作用域内关闭梯度相关计算
    class WithoutGrad final {
    public:
        explicit WithoutGrad() {
            architectures::no_grad = true;
        }
        ~WithoutGrad() noexcept {
            architectures::no_grad = false;
        }
    };

    // 用来统一各种数据类型的, 但是这样搞, 会引入虚函数多态, 效率对于 forward backward, save_weights 这些影响其实不是很大
    // backward 没法 const, 因为 relu 是就地操作的, 欸
    class Layer {
    public:
        const std::string name;  // 这一层的名字
        std::vector<tensor> output;  // 输出的张量
    public:
        Layer(std::string& _name) : name(std::move(_name)) {}
        virtual std::vector<tensor> forward(const std::vector<tensor>& input) = 0;
        virtual std::vector<tensor> backward(std::vector<tensor>& delta) = 0;
        virtual void update_gradients(const data_type learning_rate=1e-4) {}
        virtual void save_weights(std::ofstream& writer) const {}
        virtual void load_weights(std::ifstream& reader) {}
        virtual std::vector<tensor> get_output() const { return this->output; }
    };


    class Conv2D : public Layer {
    private:
        // 卷积层的固有信息
        std::vector<tensor> weights; // 卷积核的权值参数, out_channels X in_channels X kernel_size X kernel_size
        std::vector<data_type> bias; // 偏置(可以写成 tensor1D)
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
        std::vector<tensor> delta_output; // 存储回传到上一层的梯度
        std::vector<tensor> weights_gradients; // 权值的梯度
        std::vector<data_type> bias_gradients; // bias 的梯度
    public:
        Conv2D(std::string _name, const int _in_channels=3, const int _out_channels=16, const int _kernel_size=3, const int _stride=2);
        // 卷积操作的 forward 过程, batch_num X in_channels X H X W
        std::vector<tensor> forward(const std::vector<tensor>& input);
        // 优化的话, 把一些堆上的数据放到栈区, 局部变量快
        std::vector<tensor> backward(std::vector<tensor>& delta);
        // 更新梯度
        void update_gradients(const data_type learning_rate=1e-4);
        // 保存权值
        void save_weights(std::ofstream& writer) const;
        // 加载权值
        void load_weights(std::ifstream& reader);
        // 获取这一层卷积层的参数值
        int get_params_num() const;
    };


    class MaxPool2D : public Layer {
    private:
        // 这一层的固有属性
        const int kernel_size;
        const int step;
        const int padding; // 暂时不支持
        // 缓冲区, 避免每次重新分配的
        std::vector< std::vector<int> > mask; // 记录哪些位置是有梯度回传的, 第 b 张图, 每张图一个 std::vector<int>
        std::vector<tensor> delta_output; // 返回的 delta
        std::vector<int> offset;  // 偏移量指针, 和之前 Conv2D 的一样
    public:
        MaxPool2D(std::string _name, const int _kernel_size=2, const int _step=2)
                : Layer(_name), kernel_size(_kernel_size), step(_step), padding(0),
                  offset(_kernel_size * _kernel_size, 0) {}
        // 前向
        std::vector<tensor> forward(const std::vector<tensor>& input);
        // 反向传播
        std::vector<tensor> backward(std::vector<tensor>& delta);

    };


    class ReLU : public Layer  {
    public:
        ReLU(std::string _name) : Layer(_name) {}
        std::vector<tensor> forward(const std::vector<tensor>& input);
        std::vector<tensor> backward(std::vector<tensor>& delta);
    };


    // 线性变换层
    class LinearLayer : public Layer {
    private:
        // 线性层的固有信息
        const int in_channels;                // 输入的神经元个数
        const int out_channels;               // 输出的神经元个数
        std::vector<data_type> weights;       // 权值矩阵(这里其实可以改成 Tensor1D, 数据类型可以统一, 但后面的 weights_gradients 不好搞)
        std::vector<data_type> bias;          // 偏置
        // 历史信息
        std::tuple<int, int, int> delta_shape;// 记下来, delta 的形状, 从 1 X 4096 到 128 * 4 * 4 这种
        std::vector<tensor> __input;          // 梯度回传的时候需要输入 Wx + b, 需要保留 x
        // 以下是缓冲区
        std::vector<tensor> delta_output;     // delta 回传到输入的梯度
        std::vector<data_type> weights_gradients; // 缓存区, 权值矩阵的梯度
        std::vector<data_type> bias_gradients;    // bias 的梯度
    public:
        LinearLayer(std::string _name, const int _in_channels, const int _out_channels);
        // 做 Wx + b 矩阵运算
        std::vector<tensor> forward(const std::vector<tensor>& input);
        std::vector<tensor> backward(std::vector<tensor>& delta);
        void update_gradients(const data_type learning_rate=1e-4);
        void save_weights(std::ofstream& writer) const;
        void load_weights(std::ifstream& reader);
    };


    // 这个 BatchNorm 是不同通道做, 不具体实现还真不知道, 后面有时间填坑
    // 目前只考虑 Conv 层的 BN
    class BatchNorm2D : public Layer {
    private:
        // 固有信息
        const int out_channels;
        const data_type eps;
        const data_type momentum;
        // 要学习的参数(这里直接用 vector 就行, 不统一也问题不大)
        std::vector<data_type> gamma;
        std::vector<data_type> beta;
        // 要保留的历史信息
        std::vector<data_type> moving_mean;
        std::vector<data_type> moving_var;
        // 缓冲区, 避免每次重新分配
        std::vector<tensor> normed_input;
        std::vector<data_type> buffer_mean;
        std::vector<data_type> buffer_var;
        // 保留的梯度信息
        std::vector<data_type> gamma_gradients;
        std::vector<data_type> beta_gradients;
        // 临时的梯度信息, 其实也是缓冲区
        tensor norm_gradients;
        // 求梯度需要用的
        std::vector<tensor> __input;
    public:
        BatchNorm2D(std::string _name, const int _out_channels, const data_type _eps=1e-5, const data_type _momentum=0.1);
        std::vector<tensor> forward(const std::vector<tensor>& input);
        std::vector<tensor> backward(std::vector<tensor>& delta);
        void update_gradients(const data_type learning_rate=1e-4);
        void save_weights(std::ofstream& writer) const;
        void load_weights(std::ifstream& reader);
    };


    // 暂时只支持 Conv 层, 但 dropout 一般放在 linear 线性连接层
    // 虽然训练可以正常训练, 但是测试有点垃圾
    class Dropout : public Layer {
    private:
        // 固有属性
        data_type p;
        int selected_num;
        std::vector<int> sequence;
        std::default_random_engine drop;
        // backward 需要用
        std::vector<int> mask;
    public:
        Dropout(std::string _name, const data_type _p=0.5): Layer(_name), p(_p), drop(1314) {}
        std::vector<tensor> forward(const std::vector<tensor>& input);
        std::vector<tensor> backward(std::vector<tensor>& delta);
    };



    // 胡乱写的一个能跑的 CNN 网络结构, 不是真的 AlexNet
    class AlexNet {
    public:
        bool print_info = false;
    private:
        std::list< std::shared_ptr<Layer> > layers_sequence;
    public:
        AlexNet(const int num_classes=3, const bool batch_norm=false);
        // 前向
        std::vector<tensor> forward(const std::vector<tensor>& input);
        // 梯度反传
        void backward(std::vector<tensor>& delta_start);
        // 梯度更新到权值
        void update_gradients(const data_type learning_rate=1e-4);
        // 保存模型
        void save_weights(const std::filesystem::path& save_path) const;
        // 加载模型
        void load_weights(const std::filesystem::path& checkpoint_path);
        // GradCam 可视化
        cv::Mat grad_cam(const std::string& layer_name) const;
    };
}



#endif //CNN_ARCHITECTURES_H
