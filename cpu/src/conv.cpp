//C++
#include <map>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <memory>
#include <iostream>
#include <filesystem>
// self
#include "pipeline.h"


namespace {
    void run(const std::function<void()>& work=[]{}, const std::string message="") {
        auto start = std::chrono::steady_clock::now();
        work();
        auto finish = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
        std::cout << message << " " << duration.count() << " ms" <<  std::endl;
    }

    void cv_info(const cv::Mat& one_image) {
        std::cout << "高  :  " << one_image.rows << "\n宽  :  " << one_image.cols << "\n通道 :  " << one_image.channels() << std::endl;
        std::cout << "步长 :  " << one_image.step << std::endl;
    }

    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }

    cv::Mat cv_resize(cv::Mat& one_image, const int height, const int width, const int _interpolation=cv::INTER_LINEAR) {
		cv::Mat result;
		cv::resize(one_image, result, cv::Size(width, height), 0, 0, _interpolation);
		return result;
	}

	cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
		cv::Mat padded_image;
		cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
		return padded_image;
	}

	cv::Mat cv_concat(std::vector<cv::Mat> sequence) {
        cv::Mat result;
        cv::hconcat(sequence, result);
        return result;
    }
}



namespace {
    // 是否要保留中间变量, 这可以写成一个 RAII 类
    bool no_grad = false;
}



namespace pipeline {

    using list_type = std::vector<std::pair<std::string, int> >;



    using data_type = float;
    class Tensor3D {
    public:
        const int C, H, W;
        data_type* data;
        const std::string name;
        // 分配内存
        Tensor3D(const int _C, const int _H, const int _W, const std::string _name="pipeline")
            : C(_C), H(_H), W(_W), data(new data_type[_C * _H * _W]), name(std::move(_name)) {}
        // 从图像指针中读取
        void read_from_opencv_mat(const uchar* const img_ptr) {
            // 从 img_ptr 数据中获取图像内容
            const int length = H * W;
            const int length_2 = 2 * length;
            for(int i = 0;i < length; ++i) { // OpenCV 的内存排列顺序真恶心
                const int p = 3 * i;
                this->data[i] = img_ptr[p] * 1.f / 255;
                this->data[length + i] = img_ptr[p + 1] * 1.f / 255;
                this->data[length_2 + i] = img_ptr[p + 2] * 1.f / 255;
            }
        }
        void set_zero() {
            const int length = C * H * W;
            for(int i = 0;i < length; ++i) data[i] = 0;  // std::memcpy 会不会快点
        }
        void div(const data_type times) {
            const int length = C * H * W;
            for(int i = 0;i < length; ++i) data[i] /= times;
        }
        cv::Mat opecv_mat() const {
            cv::Mat origin(H, W, CV_8UC3);
            const int length = H * W;
            for(int i = 0;i < length; ++i) {
                const int p = 3 * i;
                origin.data[p] = cv::saturate_cast<uchar>(255 * data[i]);
                origin.data[p + 1] = cv::saturate_cast<uchar>(255 * data[i + length]);
                origin.data[p + 2] = cv::saturate_cast<uchar>(255 * data[i + length + length]);
            }
            return origin;
        }
        int get_length() const {return C * H * W;}
        std::tuple<int, int, int> get_shape() const {
            return std::make_tuple(C, H, W);
        }
        void print_shape() const {
            std::cout << this->name << "  ==>  " << this->C << " x " << this->H << " x " << this->W << "\n";
        }
        void print(const int _C=0) const {
            std::cout << this->name << "  内容是 :\n";
            const int start = _C * H * W;
            for(int i = 0;i < H; ++i) {
                for(int j = 0;j < W; ++j)
                    std::cout << this->data[start + i * W + j] << "   ";
                std::cout << "\n";
            }
        }
        std::shared_ptr<Tensor3D> rot180() const {
            std::shared_ptr<Tensor3D> rot(new Tensor3D(C, H, W, this->name + "_rot180"));
            const int ch_size = H * W;
            for(int c = 0;c < C; ++c) {
                data_type* old_ptr = this->data + c * ch_size;
                data_type* ch_ptr = rot->data + c * ch_size;
                for(int i = 0;i < ch_size; ++i)
                    ch_ptr[i] = old_ptr[ch_size - 1 - i];
            }
            return rot;
        }
        std::shared_ptr<Tensor3D> pad(const int padding=1) const {
            std::shared_ptr<Tensor3D> padded(new Tensor3D(C, H + 2 * padding, W + 2 * padding, this->name + "_rot180"));
            const int new_W = (W + 2 * padding);
            const int ch_size = (H + 2 * padding) * new_W;
            // padded 周围填 0
            std::memset(padded->data, 0, sizeof(data_type) * C * ch_size);
            for(int c = 0;c < C; ++c)
                for(int i = 0;i < H; ++i)
                    std::memcpy(padded->data + c * ch_size + (padding + i) * new_W + padding,
                                this->data + c * H * W + i * W, W * sizeof(data_type));
            return padded;
        }
        void normalize(const std::vector<float>& mean) {}
        ~Tensor3D() noexcept {
            if(this->data != nullptr) {
                delete this->data;
                this->data = nullptr;
                // std::cout << this->name << " 销毁一次\n";
            }
        };
    };
    using tensor = std::shared_ptr<Tensor3D>;

}



namespace architectures {
    using namespace pipeline;

    // 判断形状是否相等
    inline bool equal_shape(const std::tuple<int, int, int>& lhs, const std::tuple<int, int, int>& rhs) {
        return std::get<0>(lhs) == std::get<0>(rhs) and std::get<1>(lhs) == std::get<1>(rhs) and std::get<2>(lhs) == std::get<2>(rhs);
    }

    class Layer {
    public:
        const std::string name;
    };


    class Conv2D {
    public:
        const std::string name;  // 这一层的名字
        std::vector<tensor> weights; // 卷积核的权值参数, out_channels X in_channels X kernel_size X kernel_size
        std::vector<data_type> bias; // 偏置
        const int in_channels;  // 要滤波的特征图有几个通道
        const int out_channels; // 这一层卷积有几个卷积核
        const int kernel_size;  // 卷积核的边长
        const int stride;       // 卷积的步长
        const int padding;      // 是否要 padding, 这个有点麻烦, 会牺牲性能, 最后再说
        const int params_for_one_kernel;   // 一个卷积核的参数个数
        std::default_random_engine seed;   // 初始化的种子
        std::vector<tensor> output; // 输出的张量, 写在这里是为了减少对象的销毁, 类似于缓冲区
        std::vector<int> offset; // 卷积的偏移量, 辅助
        std::vector<tensor> delta_output; // 存储回传到上一层的梯度
        std::vector<tensor> __input; // 求梯度需要, 其实存储的是指针
        std::vector<tensor> weights_gradients; // 权值的梯度
        std::vector<data_type> bias_gradients; // bias 的梯度
    public:
        Conv2D(std::string _name, const int _in_channels=3, const int _out_channels=16, const int _kernel_size=3, const int _stride=2, const int _padding=0)
                : name(std::move(_name)), bias(_out_channels), in_channels(_in_channels), out_channels(_out_channels), kernel_size(_kernel_size), stride(_stride), padding(_padding),
                  params_for_one_kernel(_in_channels * _kernel_size * _kernel_size),
                  offset(_kernel_size * _kernel_size) {
            assert(_kernel_size & 1 and _kernel_size >= 3);
            // 首先给权值矩阵 weights 和偏置 b 分配空间
            this->weights.reserve(out_channels);
            for(int o = 0;o < out_channels; ++o) {
                // 一共有 out_channels 个卷积核, 每个卷积核有 in_channels X kernel_size X kernel_size 个参数
                weights.emplace_back(new Tensor3D(in_channels, kernel_size, kernel_size, this->name + "_" + std::to_string(o)));
            }
            // 随机初始化
            this->seed.seed(212);
            std::normal_distribution<float> engine(0.0, 1.0);
            for(int o = 0;o < out_channels; ++o) bias[o] = engine(this->seed) / 15.f;
            for(int o = 0;o < out_channels; ++o) {
                data_type* data_ptr = this->weights[o]->data;
                for(int i = 0;i < params_for_one_kernel; ++i) data_ptr[i] = engine(this->seed) / 15.f;
            }
        }
        // 卷积操作的 forward 过程, batch_num X in_channels X H X W
        std::vector<tensor> forward(const std::vector<tensor>& input) {
            // 获取输入特征图的信息
            const int batch_size = input.size();
            const int in_size = input[0]->C;
            assert(in_size == this->in_channels and std::string(this->name + "输入的通道数不对").c_str());
            const int H = input[0]->H;
            const int W = input[0]->W;
            const int length = H * W;
            // 计算输出的特征图大小
            int out_H = std::floor((H - kernel_size - 2 * padding) / stride) + 1;
            int out_W = std::floor((W - kernel_size - 2 * padding) / stride) + 1;
            const int out_length = out_H * out_W; // 输出的特征图, 一个通道的输出有多大, 111 X 111, 7 X 7 这种
            // 为卷积做准备
            const int radius = int((kernel_size - 1) / 2);
            const int H_radius = H - radius;
            const int W_radius = W - radius;
            const int window_range = kernel_size * kernel_size;
            // 如果是第一次经过这一层, 分配空间(这里灵活性差点, 如果形状不是一样的, 会崩溃)
            if(this->output.empty()) { //  or not equal_shape(this->output[0]->get_shape(), {out_channels, out_H, out_W})
                // std::cout << this->name << " 第一次分配输出的向量\n";
                // 分配输出的张量
                this->output.reserve(batch_size);
                for(int b = 0;b < batch_size; ++b)  // B X 16 X 111 X 111, 多通道求和
                    this->output.emplace_back(new Tensor3D(out_channels, out_H, out_W, this->name + "_output_" + std::to_string(b)));
                // 辅助变量 offset 只求一遍
                int pos = 0;
                for(int x = -radius;x <= radius; ++x) {
                    for(int y = -radius; y <= radius; ++y) {
                        this->offset[pos] = x * W + y;
                        ++pos;
                    }
                }
            }
            output[0]->print_shape();
            // 记录输入, 如果存在 backward 的话
            this->__input = input;
            // 首先每张图像分开卷积
            for(int b = 0;b < batch_size; ++b) {
                // in_channels X 224 X 224
                data_type* const cur_image_features = input[b]->data;
                // 16 个卷积核的输出, 16 x 111 x 111
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
                                for(int k = 0;k < window_range; ++k) { // 遍历局部窗口
                                    sum_value += cur_image_features[start + offset[k]] * cur_w_ptr[start_w + k];
                                }
                            }
                            sum_value += this->bias[o]; // 别忘记加上 b
                            out_ptr[cnt] = sum_value; // 卷积结果放到输出的 cnt 位置上, 按行优先存储
                            ++cnt;
                        }
                    } // std::cout << "cnt = " << std::sqrt(cnt) << std::endl;
                }
            }
            return this->output;
        }

        // 优化的话, 把一些堆上的数据放到栈区, 局部变量快
        std::vector<tensor> backward(const std::vector<tensor>& delta) {
            // 获取信息  batch_size X out_channels X 2 X 2
            std::cout << "backward of  " << this->name << std::endl;
            const int batch_size = delta.size();
            assert(batch_size > 0 and batch_size == this->__input.size());
            const int out_H = delta[0]->H;
            const int out_W = delta[0]->W;
            const int out_length = out_H * out_W;
            // 获取输入的信息
            const int H = this->__input[0]->H;
            const int W = this->__input[0]->W;
            const int length = H * W;
            // 给缓冲区的梯度分配空间
            if(this->weights_gradients.empty()) {
                // W
                this->weights_gradients.reserve(out_channels);
                for(int o = 0;o < out_channels; ++o)
                    this->weights_gradients.emplace_back(new Tensor3D(in_channels, kernel_size, kernel_size, this->name + "_weights_gradients_" + std::to_string(o)));
                // b
                this->bias_gradients.assign(out_channels, 0);
            }
            for(int o = 0;o < out_channels; ++o) this->weights_gradients[o]->set_zero();
            for(int o = 0;o < out_channels; ++o) this->bias_gradients[o] = 0;
            std::cout << "内存分配成功 " << bias_gradients.size() << std::endl;
            // 先求 W, b 的梯度, 更简单
            // 求 weight, out_channels X in_channels X 3 X 3
            // 对 batch 中的梯度求均值
            const int weight_len = kernel_size * kernel_size;
            for(int b = 0;b < batch_size; ++b) {
                // 首先, 遍历每个卷积核
                for(int o = 0;o < out_channels; ++o) {
                    // 第 b 张梯度第 o 个通道的梯度
                    data_type* o_delta = delta[b]->data + o * out_H * out_W;
                    // 卷积核的每个 in 通道, 分开求
                    for(int i = 0;i < in_channels; ++i) {
                        // 第 b 张第 i 个通道的输入
                        data_type* in_ptr = __input[b]->data + i * H * W;
                        // 现在到了第 o 个卷积核的第 i 个通道
                        data_type* w_ptr = weights_gradients[o]->data + i * kernel_size * kernel_size;
                        // 找到要求的 w 的每个点, 从特征图中找到对应的点
                        for(int k_x = 0; k_x < kernel_size; ++k_x) {
                            for(int k_y = 0;k_y < kernel_size; ++k_y) {
                                data_type sum_value = 0;
                                // w 二维平面现在要求的第 k 个点, 遍历 delta 平面, 去输入中找对应相乘的数
                                for(int x = 0;x < out_H; ++x) {
                                    data_type* delta_ptr = o_delta + x * out_W; // delta 每一行
                                    data_type* input_ptr = in_ptr + (x * stride + k_x) * W; // 对应的输入在第几行
                                    for(int y = 0;y < out_W; ++y) {
                                        // 找到 delta 的值
                                        data_type delta_value = delta_ptr[y];
                                        // 找到输入的点值, 去二维平面中去找
                                        data_type input_value = input_ptr[y * stride + k_y];
                                        // 得到这个 w 权值的梯度
                                        sum_value += input_value * delta_value;
                                        if(i == 0 and k_x == 0 and k_y == 0) {
                                            std::cout << input_value << " * " << delta_value << " ==> " << sum_value << std::endl;
                                        }
                                    }
                                }
                                // 更新到 weight_gradients, 注意除以了 batch_size
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
            std::cout << "开始更新梯度!\n";
            std::cout << "bias_gradients  " << bias_gradients[0] << std::endl;
            weights_gradients[0]->print(0);
            weights_gradients[0]->print(1);
            // 把梯度更新到 W 和 b
            for(int o = 0;o < out_channels; ++o) {
                data_type* w_ptr = weights[o]->data;
                data_type* wg_ptr = weights_gradients[o]->data;
                for(int i = 0;i < params_for_one_kernel; ++i)
                    w_ptr[i] -= 1e-3 * wg_ptr[i];
                bias[o] -= 1e-3 * bias_gradients[o];
            }
            std::cout << "开始计算 delta 的梯度, 反传给输入的梯度!\n";
            // 给 delta_output 分配内存
            if(this->delta_output.empty()) {
                this->delta_output.reserve(batch_size);
                for(int b = 0;b < batch_size; ++b)
                    this->delta_output.emplace_back(new Tensor3D(in_channels, H, W, this->name + "_delta_" + std::to_string(b)));
            }
            // delta 初始化为 0
            for(int o = 0;o < batch_size; ++o)
                this->delta_output[o]->set_zero();
            // 开始计算 delta 的梯度, 传回到输入的
            const int radius = (kernel_size - 1) / 2;
            const int H_radius = H - radius;
            const int W_radius = W - radius;
            const int window_range = kernel_size * kernel_size;
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
                            data_type sum_value = 0.f;
                            const int coord = x * W + y; // 当前点对于这个通道的特征图的位移
                            for(int i = 0;i < in_channels; ++i) { // 每个点有多个通道
                                const int start = i * length + coord; // 输入的第 i 张特征图在 (x, y) 处的位移
                                const int start_w = i * window_range; // 第 o 个卷积核的第 i 个通道
                                for(int k = 0;k < window_range; ++k) { // 遍历局部窗口
                                    sum_value += cur_image_features[start + offset[k]] * cur_w_ptr[start_w + k];
                                    cur_image_features[start + offset[k]] += cur_w_ptr[start_w + k] * out_ptr[cnt];
                                    std::cout << start + offset[k] << "  ===> " << cur_w_ptr[start_w + k] << " * " << out_ptr[cnt] << "\n";
                                }
                            }
                            ++cnt;
                        }
                    }
                }
            }
            // 返回
            return this->delta_output;
        }

        // 获取这一层卷积层的参数值
        int get_params_num() const {
            return (this->params_for_one_kernel + 1) * this->out_channels;
        }
    };
}




int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;


    using namespace architectures;

    // 开始模拟 cnn 的前向跟反向传播

//    // 设定输入的特征图, 2 X 5 X 5
//    std::vector<tensor> input;
//    input.emplace_back(new Tensor3D(2, 5, 5, "conv_in"));
//    data_type* src_ptr = input[0]->data;
//    for(int i = 0;i < 50; ++i) src_ptr[i] = (i + 1) * 0.01f;
//    input[0]->print(0);
//    input[0]->print(1);
//    // 设计一个卷积核
//    Conv2D conv("demo", 2, 1, 3, 2);
//    data_type* w_ptr = conv.weights[0]->data; // 2 X 3 X 3
//    for(int i = 0;i < 18; ++i) w_ptr[i] = (i + 1) * 0.02f;
//    for(int i = 0;i < 1; ++i) conv.bias[i] = (i + 1) * 0.3;
//    conv.weights[0]->print(0);
//    conv.weights[0]->print(1);
//    // 前向传播
//    auto output = conv.forward(input);
//    output[0]->print();
//    output[0]->print_shape();







//    // 设定输入的特征图, 2 X 5 X 5
//    std::vector<tensor> input;
//    input.emplace_back(new Tensor3D(2, 5, 5, "conv_in"));
//    data_type* src_ptr = input[0]->data;
//    for(int i = 0;i < 50; ++i) src_ptr[i] = (i + 1) * 0.01f;
//    input[0]->print(0);
//    input[0]->print(1);
//    // 设计一个卷积核
//    Conv2D conv("demo", 2, 4, 3, 2);
//    for(int o = 0;o < 4; ++o) {
//        data_type* w_ptr = conv.weights[o]->data;
//        for(int i = 0;i < 18; ++i)
//            w_ptr[i] = (i + 1) * 0.02f * (o + 1);
//        conv.bias[o] = (o + 1) * 0.3;
//    }
//    for(int o = 0;o < 4; ++o) {
//        std::cout << "第 " << o << " 个卷积核\n";
//        conv.weights[o]->print(0);
//        conv.weights[o]->print(1);
//    }
//    // 前向传播
//    auto output = conv.forward(input);
//    for(int o = 0;o < 4; ++o) {
//        output[0]->print(o);
//    }




    // 这一次模拟反向传播
    std::vector<tensor> input;
    input.emplace_back(new Tensor3D(2, 5, 5, "conv_in"));
    data_type* src_ptr = input[0]->data;
    for(int i = 0;i < 50; ++i) src_ptr[i] = (i + 1) * 0.01f;
    input[0]->print(0);
    input[0]->print(1);

    // 设计一个卷积核
    Conv2D conv("demo", 2, 1, 3, 2);
    data_type* w_ptr = conv.weights[0]->data; // 2 X 3 X 3
    for(int i = 0;i < 18; ++i) w_ptr[i] = (i + 1) * 0.02f;
    for(int i = 0;i < 1; ++i) conv.bias[i] = (i + 1) * 0.3;
    conv.weights[0]->print(0);
    conv.weights[0]->print(1);

    // 前向传播
    auto output = conv.forward(input);
    output[0]->print();
    output[0]->print_shape();

    // 设计梯度, 大小和 output 一样
    std::vector<tensor> delta;
    delta.emplace_back(new Tensor3D(1, 2, 2, "delta"));
    for(int i = 0;i < 4; ++i) {
        delta[0]->data[i] = -0.10 + i * 0.04;
    }
    delta[0]->print();

    // 开始回传
    auto delta_pre = conv.backward(delta);

    return 0;
}
