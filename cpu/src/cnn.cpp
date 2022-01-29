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

    std::map<std::string, list_type> get_images_for_classification(
            const std::filesystem::path dataset_path, const std::vector<std::string> categories={}, const std::pair<float, float>& ratios={0.8, 0.1}) {
        // 遍历 dataset_path 文件夹下指定的类别
        list_type all_images_list;
        const int categories_num = categories.size();
        for(int i = 0;i < categories_num; ++i) {
            const auto images_dir = dataset_path / categories[i];
            assert(std::filesystem::exists(images_dir) and std::string(images_dir.string() + " 路径不存在!").c_str());
            auto walker = std::filesystem::directory_iterator(images_dir);
            for(const auto& iter : walker)
                all_images_list.emplace_back(iter.path().string(), i);
        }
        // 打乱图像列表
        std::shuffle(all_images_list.begin(), all_images_list.end(), std::default_random_engine(212));
        // 将数据集划分成三部分
        const int total_size = all_images_list.size();
        assert(ratios.first > 0 and ratios.second > 0 and ratios.first + ratios.second < 1);
        const int train_size = int(total_size * ratios.first);
        const int test_size = int(total_size * ratios.second);
        std::map<std::string, list_type> results;
        results.emplace("train", list_type(all_images_list.begin(), all_images_list.begin() + train_size));
        results.emplace("test", list_type(all_images_list.begin() + train_size, all_images_list.begin() + train_size + test_size));
        results.emplace("valid", list_type(all_images_list.begin() + train_size + test_size, all_images_list.end()));
        std::cout << "train  :  " << results["train"].size() << "\n" << "test   :  " << results["test"].size() << "\n" << "valid  :  " << results["valid"].size() << "\n";
        return results;
    }

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

    class DataLoader {
        using batch_type = std::vector< std::pair<tensor, int> >;
    private:
        list_type images_list; // 数据集列表, image <==> label
        int images_num;  // 这个子数据集一共有多少张图像和对应的标签
        const int batch_size; // 每次打包几张图象
        const bool augment; // 是否要做图像增强
        const bool shuffle; // 是否要打乱列表
        const int seed; // 每次随机打乱列表的种子
        int iter = -1;  // 当前采到了第 iter 张图像
        std::vector<tensor> buffer; // batch 缓冲区, 不用每次分配和销毁
        const int H = 256, W = 256, C = 3; // 允许的图像尺寸
    public:
        explicit DataLoader(const list_type& _images_list, const int _bs=1, const bool _aug=false, const bool _shuffle=true, const int _seed=212)
                : images_list(_images_list), batch_size(_bs), augment(_aug), shuffle(_shuffle), seed(_seed) {
            this->images_num = this->images_list.size();
            this->buffer.reserve(this->batch_size);
            for(int i = 0;i < this->batch_size; ++i)
                this->buffer.emplace_back(new Tensor3D(C, H, W));
        }
        int length() const {return this->images_num;}
        batch_type generate_batch() {
            // 我要开始算有几个 batch, 然后将 batch 组合在一起
            batch_type one_batch;
            one_batch.reserve(this->batch_size);
            for(int i = 0;i < this->batch_size; ++i)
                one_batch.emplace_back(this->add_to_buffer(i));
            return one_batch;
        }
    private:
        std::pair<tensor, int> add_to_buffer(const int batch_index) {
            // 获取图像序号
            ++this->iter;
            if(this->iter == this->images_num) {
                this->iter = 0;  // 重新开始取
                if(this->shuffle) { // 然后再一次打乱列表
                    std::shuffle(this->images_list.begin(), this->images_list.end(), std::default_random_engine(this->seed)); // std::cout << this->images_list[0].first << ", " << this->images_list[0].second << std::endl;
                }
            }
            // 读取图像
            const auto& image_path = this->images_list[this->iter].first;
            const int image_label = this->images_list[this->iter].second;
            cv::Mat origin = cv::imread(image_path);
            // resize
            cv::resize(origin, origin, {W, H});
            // 对图像做数据增强
            // 直接对 buffer 进行填充, 将图像转化成 float 数据, 且是 Tensor 形式, C x H x W;
            this->buffer[batch_index]->read_from_opencv_mat(origin.data);
            // 返回图像内容和标签
            return std::pair<tensor, int>(this->buffer[batch_index], image_label);
        }
    };
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
    private:
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
            for(int o = 0;o < out_channels; ++o) bias[o] = engine(this->seed);
            for(int o = 0;o < out_channels; ++o) {
                data_type* data_ptr = this->weights[o]->data;
                for(int i = 0;i < params_for_one_kernel; ++i) data_ptr[i] = engine(this->seed);
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
        // 获取这一层卷积层的参数值
        int get_params_num() const {
            return (this->params_for_one_kernel + 1) * this->out_channels;
        }
    };


    class MaxPool2D {
    private:
        const std::string name;
        const int kernel_size;
        const int step;
        const int padding;
        std::vector<tensor> output;
        std::vector< std::vector<int> > mask;    // 记录哪些位置是有梯度回传的, 第 b 张图, 每张图一个 std::vector<int>
        std::vector<tensor> delta_output;// 返回的 delta
        std::vector<int> offset;  // 偏移量
    public:
        MaxPool2D(std::string _name, const int _kernel_size=2, const int _step=2)
                : name(std::move(_name)), kernel_size(_kernel_size), step(_step), padding(0),
                  offset(_kernel_size * _kernel_size, 0) {}
        std::vector<tensor> forward(const std::vector<tensor>& input) {
            // 获取输入信息
            const int batch_size = input.size();
            const int H = input[0]->H;
            const int W = input[0]->W;
            const int C = input[0]->C;
            // 计算输出的大小
            const int out_H = std::floor(((H - kernel_size + 2 * padding) / step)) + 1;
            const int out_W = std::floor(((W - kernel_size + 2 * padding) / step)) + 1;
            // 第一次经过该池化层
            if(this->output.empty()) {
                // 给输出分配空间
                this->output.reserve(batch_size);
                for(int b = 0;b < batch_size; ++b)
                    this->output.emplace_back(new Tensor3D(C, out_H, out_W, this->name + "_output_" + std::to_string(b)));
                // 给反向传播的 delta 分配空间
                this->delta_output.reserve(batch_size);
                for(int b = 0;b < batch_size; ++b)
                    this->delta_output.emplace_back(new Tensor3D(C, H, W, this->name + "_delta_" + std::to_string(b)));
                // mask 对 batch 的每一张图都分配空间
                this->mask.reserve(batch_size);
                for(int b = 0;b < batch_size; ++b)
                    this->mask.emplace_back(std::vector<int>(C * out_H * out_W, 0));
                // 第一次经过这一层, 根据 kernel_size 计算 offset
                int pos = 0;
                for(int i = 0;i < kernel_size; ++i)
                    for(int j = 0;j < kernel_size; ++j)
                        this->offset[pos++] = i * W + j;
            }
            // 每次 forward 要记得把 mask 全部填充为 0, 很重要 !
            const int out_length = out_H * out_W;
            const int mask_size = C * out_length;
            for(int b = 0;b < batch_size; ++b) {
                int* const mask_ptr = this->mask[b].data();
                for(int i = 0;i < mask_size; ++i)
                    mask_ptr[i] = 0;
            }
            this->output[0]->print_shape();
            // 开始池化
            const int length = H * W;
            const int H_kernel = H - kernel_size;
            const int W_kernel = W - kernel_size;
            const int window_range = kernel_size * kernel_size;
            for(int b = 0;b < batch_size; ++b) { // batch 的每一张图像对应的特征图分开池化
                // 16 X 111 X 111 → 16 X 55 X 55
                for(int i = 0;i < C; ++i) {  // 每个通道
                    // 现在我拿到了第 b 张图的第 i 个通道, 一个 55 X 55 的指针
                    data_type* const cur_image_features = input[b]->data + i * length;
                    data_type* const output_ptr = this->output[b]->data + i * out_length;
                    int* const mask_ptr = this->mask[b].data() + i * out_length; // 累计有 out_length 个有效值, 梯度回传的时候
                    int cnt = 0;
                    for(int x = 0; x <= H_kernel; x += step) {
                        data_type* const row_ptr = cur_image_features + x * W;
                        for(int y = 0; y <= W_kernel; y += step) {
                            // 找到局部的 kernel_size X kernel_size 的区域, 找最大值
                            data_type max_value = row_ptr[y];
                            int max_index = 0; // 记录最大值的位置
                            for(int k = 1; k < window_range; ++k) { // 从 1 开始因为 0 已经比过了
                                data_type comp = row_ptr[y + offset[k]];
                                if(max_value < comp) {
                                    max_value = comp;
                                    max_index = offset[k];
                                }
                            }
                            // 记录局部最大值
                            output_ptr[cnt] = max_value;
                            // 第 i 个通道, i * out_H * out_W 为起点的二维平面, 偏移量 max_index
                            max_index += x * W + y;
                            mask_ptr[cnt] = i * length + max_index;
                            ++cnt;
                        }
                    } // if(this->name == "max_pool_2" and b == 0 and i == 0)
                }
            }
            return this->output;
        }

        // 反向传播
        std::vector<tensor> backward(const std::vector<tensor>& delta) {
            // 获取输入的梯度信息
            const int batch_size = delta.size();
            assert(batch_size == this->delta_output.size());
            assert(equal_shape(this->output[0]->get_shape(), delta[0]->get_shape()));
            // B X 128 X 6 X 6, 先填 0
            for(int b = 0;b < batch_size; ++b)
                this->delta_output[b]->set_zero();
            // batch 每张图像, 根据 mask 标记的位置, 把 delta 中的值填到 delta_output 中去
            const int total_length = delta[0]->get_length();
            for(int b = 0;b < batch_size; ++b) {
                int* mask_ptr = this->mask[b].data();
                data_type* const src_ptr = delta[b]->data;
                data_type* const res_ptr = this->delta_output[b]->data;
                for(int i = 0;i < total_length; ++i) {
                    res_ptr[mask_ptr[i]] = src_ptr[i];
                }
            }
            return this->delta_output;
        }
    };



    class AlexNet {
    private:
        Conv2D conv_layer_1 = Conv2D("conv_layer_1", 3, 16, 3);
        Conv2D conv_layer_2 = Conv2D("conv_layer_2", 16, 32, 3);
        Conv2D conv_layer_3 = Conv2D("conv_layer_3", 32, 64, 3);
        Conv2D conv_layer_4 = Conv2D("conv_layer_4", 64, 128, 3);
        MaxPool2D max_pool_1 = MaxPool2D("max_pool_1", 2, 2);
        MaxPool2D max_pool_2 = MaxPool2D("max_pool_2", 3, 2);
    public:
        std::vector<tensor> forward(const std::vector<tensor>& input) {
            // 对输入的形状做检查
            auto conv_output_1 = this->conv_layer_1.forward(input);
            auto pool_output_1 = this->max_pool_1.forward(conv_output_1);
            auto conv_output_2 = this->conv_layer_2.forward(pool_output_1);
            auto conv_output_3 = this->conv_layer_3.forward(conv_output_2);
            auto conv_output_4 = this->conv_layer_4.forward(conv_output_3);
            auto pool_output_2 = this->max_pool_2.forward(conv_output_4);
            // 在这里模拟 pool 层的反向传播
            const int CH = 30;
            conv_output_4[0]->print(CH);
            pool_output_2[0]->print(CH);
            std::vector<tensor> backward_delta;
            for(int b = 0;b < 4; ++b) {
                tensor one(new Tensor3D(128, 3, 3));
                for(int i = 0;i < 128; ++i) {
                    data_type* ch_ptr = one->data + i * 9;
                    for(int k = 1;k <= 9; ++k) ch_ptr[k - 1] = 0.1f * k;
                }
                backward_delta.emplace_back(one);
            }
            backward_delta[0]->print(CH);
            const auto delta = this->max_pool_2.backward(backward_delta);
            delta[0]->print(CH);
            return pool_output_2;
        }
    };
}




int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    // 指定一些参数
    const int train_batch_size = 4;
    const int valid_batch_size = 1;
    const int test_batch_size = 1;
    const std::tuple<int, int, int> image_size({224, 224, 3});
    const std::filesystem::path dataset_path("../datasets/animals");
    const std::vector<std::string> categories({"dog", "panda", "bird"});

    // 获取图片
    auto dataset = pipeline::get_images_for_classification(dataset_path, categories);

    // 构造数据流
    pipeline::DataLoader train_loader(dataset["train"], train_batch_size, false, true);

    // 定义网络结构
    std::unique_ptr<architectures::AlexNet> network(new architectures::AlexNet());

    // 开始训练
    const int total_iters = 100000; // 训练 batch 的总数
    const float learning_rate = 1e-3; // 学习率
    const int valid_inters = 600; // 验证一次的间隔
    for(int iter = 1; iter <= total_iters; ++iter) {
        // 从训练集中采样一个 batch
        const auto sample = train_loader.generate_batch();
        // 分离出图像指针和标签
        std::vector<pipeline::tensor> images;
        std::vector<int> labels(train_batch_size);
        images.reserve(train_batch_size);
        for(int b = 0;b < train_batch_size; ++b) {
            images.emplace_back(sample[b].first);
            labels[b] = sample[b].second;  // 这里最好改成 pair
        }
        // 送到网络中
        const auto output = network->forward(images);
        if(iter == 2) break;
    }

    // 保存
    const std::filesystem::path checkpoints_dir("./checkpoints/AlexNet");
    if(not std::filesystem::exists(checkpoints_dir))
        std::filesystem::create_directories(checkpoints_dir);

    // 加载模型, 推断
    std::cout << "训练结束!\n";
    return 0;
}
