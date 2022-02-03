//C++
#include <map>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <memory>
#include <iostream>
#include <filesystem>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


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

	cv::Mat cv_concat(std::vector<cv::Mat> sequence) {
        cv::Mat result;
        cv::hconcat(sequence, result);
        return result;
    }

    cv::Mat rotate(cv::Mat& src, double angle) {
        // 抄自 https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
        // 角度最好在正负 15-75 之间, 这个程序还是有问题的
        cv::Point2f center((src.cols-1)/2.0, (src.rows-1)/2.0);
        cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
        rot.at<double>(0,2) += bbox.width/2.0 - src.cols/2.0;
        rot.at<double>(1,2) += bbox.height/2.0 - src.rows/2.0;
        cv::warpAffine(src, src, rot, bbox.size());
        return src;
    }
}



namespace {
    // 全局变量, 是否要 backward, 访问速度上要慢一些
    bool no_grad = false;

    class WithoutGrad final {
    public:
        explicit WithoutGrad() {
            no_grad = true;
        }
        ~WithoutGrad() noexcept {
            no_grad = false;
        }
    };
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
        // 直接传一个 tuple 代表形状
        Tensor3D(const std::tuple<int, int, int>& shape, const std::string _name="pipeline")
            : C(std::get<0>(shape)), H(std::get<1>(shape)), W(std::get<2>(shape)),
              data(new data_type[std::get<0>(shape) * std::get<1>(shape) * std::get<2>(shape)]),
              name(std::move(_name)) {}
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
            this->normalize();
        }
        void normalize(const std::vector<data_type> mean={0.406, 0.456, 0.485}, const std::vector<data_type> std_div={0.225, 0.224, 0.229}) {
            if(C != 3) return;
            const int ch_size = H * W;
            for(int ch = 0;ch < C; ++ch) {
                data_type* const ch_ptr = this->data + ch * ch_size;
                for(int i = 0;i < ch_size; ++i)
                    ch_ptr[i] = (ch_ptr[i] - mean[ch]) / std_div[ch];
            }
        }
        cv::Mat opecv_mat() const {
            // 只针对没有进行 normalize 的 Tensor 可以取出数据查看, 坑不填了, 懒得
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
        using batch_type = std::pair< std::vector<tensor>, std::vector<int> >; // batch 是一个 pair
    private:
        list_type images_list; // 数据集列表, image <==> label
        int images_num;        // 这个子数据集一共有多少张图像和对应的标签
        const int batch_size;  // 每次打包几张图象
        const bool augment;    // 是否要做图像增强
        const bool shuffle;    // 是否要打乱列表
        const int seed;        // 每次随机打乱列表的种子
        int iter = -1;         // 当前采到了第 iter 张图像
        std::vector<tensor> buffer; // batch 缓冲区, 用来从图像生成 tensor 的, 不用每次要读取图像时分配和销毁
        const int H, W, C;     // 允许的图像尺寸
    public:
        explicit DataLoader(const list_type& _images_list, const int _bs=1, const bool _aug=false, const bool _shuffle=true, const std::tuple<int, int, int> image_size={224, 224, 3},  const int _seed=212)
                : images_list(_images_list),
                  batch_size(_bs),
                  augment(_aug),
                  shuffle(_shuffle),
                  H(std::get<0>(image_size)),
                  W(std::get<1>(image_size)),
                  C(std::get<2>(image_size)),
                  seed(_seed) {
            this->images_num = this->images_list.size();  // 总共有多少张图象
            this->buffer.reserve(this->batch_size);  // 预留空间
            for(int i = 0;i < this->batch_size; ++i) // 每次图像
                this->buffer.emplace_back(new Tensor3D(C, H, W));
        }
        int length() const {return this->images_num;}
        batch_type generate_batch() {
            // 我要开始算有几个 batch, 然后将 batch 组合在一起
            std::vector<tensor> images;
            std::vector<int> labels;
            images.reserve(this->batch_size);
            labels.reserve(this->batch_size);
            for(int i = 0;i < this->batch_size; ++i) {
                auto sample = this->add_to_buffer(i);
                images.emplace_back(sample.first);
                labels.emplace_back(sample.second);
            }
            return std::make_pair(std::move(images), std::move(labels));
        }
    private:
        // 获取第 batch_index 的图像信息, 填充成 tensor
        std::pair<tensor, int> add_to_buffer(const int batch_index) {
            // 获取图像序号
            ++this->iter;
            if(this->iter == this->images_num) { // 取到头了, 重新开始
                this->iter = 0;  // 下标置为 0
                if(this->shuffle) { // 然后再一次打乱列表
                    std::shuffle(this->images_list.begin(), this->images_list.end(), std::default_random_engine(this->seed)); // std::cout << this->images_list[0].first << ", " << this->images_list[0].second << std::endl;
                }
            }
            // 读取图像
            const auto& image_path = this->images_list[this->iter].first;
            const int image_label = this->images_list[this->iter].second;
            cv::Mat origin = cv::imread(image_path);
            // 对图像做数据增强
            if(this->augment) this->augmentor.make_augment(origin);
            // resize, 必须在数据增强之后
            cv::resize(origin, origin, {W, H});
            // 直接对 buffer 进行填充, 将图像转化成 float 数据, 且是 Tensor 形式, C x H x W;
            this->buffer[batch_index]->read_from_opencv_mat(origin.data);
            // 返回图像内容和标签
            return std::pair<tensor, int>(this->buffer[batch_index], image_label);
        }
        class ImageAugmentor {
        private:
            std::default_random_engine e, l, c, r; // e 用来获得操作的概率; l 用来打乱操作列表; c 用来得到裁剪需要的概率; r 用来得到旋转的概率
            std::uniform_real_distribution<float> engine;
            std::uniform_real_distribution<float> crop_engine;
            std::uniform_real_distribution<float> rotate_engine;
            std::uniform_int_distribution<int> minus_engine;
            std::vector<std::pair<std::string, float> > ops;
        public:
            ImageAugmentor() : e(212), l(826), c(320), r(520),
                engine(0.0, 1.0), crop_engine(0.0, 0.25), rotate_engine(15, 75), minus_engine(1, 10),
                ops({{"hflip", 0.5}, {"vflip", 0.2}, {"crop", 0.7}, {"rotate", 0.5}}) {}

            void make_augment(cv::Mat& origin) {
                // 首先打乱操作的顺序
                std::shuffle(ops.begin(), ops.end(), this->l);
                // 遍历这个操作
                for(const auto& item : ops) {
                    // 获得一个随机概率
                    const float prob = engine(e);
                    // 如果概率足够大, 执行操作
                    if(prob >= 1.0 - item.second) {
                        // 镜像翻转
                        if(item.first == "hflip")
                            cv::flip(origin, origin, 1);
                        else if(item.first == "vflip")
                            cv::flip(origin, origin, 0);
                        else if(item.first == "crop") {
                             // 获取图像信息
                            const int H = origin.rows;
                            const int W = origin.cols;
                            // 获取截取的比例
                            float crop_ratio = 0.7f + crop_engine(c);
                            const int _H = int(H * crop_ratio);
                            const int _W = int(W * crop_ratio);
                            // 获取截取的位置
                            std::uniform_int_distribution<int> _H_pos(0, H - _H);
                            std::uniform_int_distribution<int> _W_pos(0, W - _W);
                            // 开始截取图像
                            origin = origin(cv::Rect(_W_pos(c), _H_pos(c), _W, _H)).clone();
                        }
                        else if(item.first == "rotate") {
                            // 获取一个随即角度
                            float angle = rotate_engine(r);
                            if(minus_engine(r) & 1) angle = -angle;
                            origin = rotate(origin, angle);
                        }
                    }
                }
            }
        };
        ImageAugmentor augmentor;
    };
}



namespace architectures {
    using namespace pipeline;

    // 随机初始化用的, C++ 这个生成的数字过大, softmax 之前的都好几百, 直接爆了, 坑爹
    const data_type random_times = 10.f;


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
        Conv2D(std::string _name, const int _in_channels=3, const int _out_channels=16, const int _kernel_size=3, const int _stride=2)
                : name(std::move(_name)), bias(_out_channels), in_channels(_in_channels), out_channels(_out_channels), kernel_size(_kernel_size), stride(_stride),
                  params_for_one_kernel(_in_channels * _kernel_size * _kernel_size),
                  offset(_kernel_size * _kernel_size) {
            assert(_kernel_size & 1 and _kernel_size >= 3 and "卷积核的大小必须是正奇数 !");
            // 首先给权值矩阵 weights 和偏置 b 分配空间
            this->weights.reserve(out_channels);
            for(int o = 0;o < out_channels; ++o) {
                // 一共有 out_channels 个卷积核, 每个卷积核有 in_channels X kernel_size X kernel_size 个参数
                weights.emplace_back(new Tensor3D(in_channels, kernel_size, kernel_size, this->name + "_" + std::to_string(o)));
            }
            // 随机初始化, 这里用的是正态分布初始化
            this->seed.seed(212);
            std::normal_distribution<float> engine(0.0, 1.0);
            for(int o = 0;o < out_channels; ++o) bias[o] = engine(this->seed) / random_times;
            for(int o = 0;o < out_channels; ++o) {
                data_type* data_ptr = this->weights[o]->data;
                for(int i = 0;i < params_for_one_kernel; ++i)
                    data_ptr[i] = engine(this->seed) / random_times;
            }
        }

        // 卷积操作的 forward 过程, batch_num X in_channels X H X W
        std::vector<tensor> forward(const std::vector<tensor>& input) {
            // 获取输入特征图的信息
            const int batch_size = input.size();
            const int in_size = input[0]->C;
            const int H = input[0]->H;
            const int W = input[0]->W;
            const int length = H * W; // 一个二维特征图的大小, 用来算偏移量的
            // 计算输出的特征图大小
            const int out_H = std::floor((H - kernel_size - 2 * padding) / stride) + 1;
            const int out_W = std::floor((W - kernel_size - 2 * padding) / stride) + 1;
            const int out_length = out_H * out_W; // 输出的特征图, 一个通道的输出有多大, 111 X 111, 7 X 7 这种
            // 为卷积做准备
            const int radius = int((kernel_size - 1) / 2);
            // 如果是第一次经过这一层, 分配空间(这里灵活性差点, 如果形状不是一样的, 可能会崩溃, 要重新分配, 暂时不搞了)
            if(this->output.empty()) {
                // std::cout << this->name << " 第一次分配输出的向量\n";
                // 分配输出的张量
                this->output.reserve(batch_size);
                for(int b = 0;b < batch_size; ++b)  // B X 16 X 111 X 111
                    this->output.emplace_back(new Tensor3D(out_channels, out_H, out_W, this->name + "_output_" + std::to_string(b)));
                // 辅助变量 offset 只求一遍, 虽然方便, 但没有局部变量快
                int pos = 0;
                for(int x = -radius;x <= radius; ++x)
                    for(int y = -radius; y <= radius; ++y) {
                        this->offset[pos] = x * W + y;
                        ++pos;
                    }
            }
            // 记录输入, 如果存在 backward 的话, 后面 backward 算 w 的梯度要用
            if(not no_grad) this->__input = input;
            // 为卷积做准备
            const int H_radius = H - radius; // 避免每次循环重新计算 H - radius
            const int W_radius = W - radius;
            const int window_range = kernel_size * kernel_size; // 卷积核一个二维平面的大小, 用来算偏移的
            const int* const __offset = this->offset.data(); // 获取偏移量指针
            // 首先每张图像分开卷积
            for(int b = 0;b < batch_size; ++b) {
                // 获取第 b 张图像的起始地址, in_channels X 224 X 224
                data_type* const cur_image_features = input[b]->data;
                // B 个卷积核的输出, B x 111 x 111
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
                                for(int k = 0;k < window_range; ++k)// 遍历局部窗口
                                    sum_value += cur_image_features[start + __offset[k]] * cur_w_ptr[start_w + k];
                            }
                            sum_value += this->bias[o]; // 别忘记加上 b
                            out_ptr[cnt] = sum_value;   // 卷积结果放到输出的 cnt 位置上, 按行优先存储
                            ++cnt;  // 存放的位置 + 1
                        }
                    } // std::cout << "cnt = " << std::sqrt(cnt) << std::endl;
                }
            }
            return this->output;  // 返回卷积结果, 在上面的 out_ptr 被更新
        }

        // 优化的话, 把一些堆上的数据放到栈区, 局部变量快
        std::vector<tensor> backward(const std::vector<tensor>& delta) {
            // 获取回传的梯度信息, 之前 forward 输出是多大, delta 就是多大(不考虑异常输入)
            const int batch_size = delta.size();
            const int out_H = delta[0]->H;
            const int out_W = delta[0]->W;
            const int out_length = out_H * out_W;
            // 获取之前 forward 的输入特征图信息
            const int H = this->__input[0]->H;
            const int W = this->__input[0]->W;
            const int length = H * W;
            // 第一次经过这里, 给缓冲区的梯度分配空间
            if(this->weights_gradients.empty()) {
                // weights
                this->weights_gradients.reserve(out_channels);
                for(int o = 0;o < out_channels; ++o)
                    this->weights_gradients.emplace_back(new Tensor3D(in_channels, kernel_size, kernel_size, this->name + "_weights_gradients_" + std::to_string(o)));
                // bias
                this->bias_gradients.assign(out_channels, 0);
            }
            // 这里默认不记录梯度的历史信息, W, b 之前的梯度全部清空
            for(int o = 0;o < out_channels; ++o) this->weights_gradients[o]->set_zero();
            for(int o = 0;o < out_channels; ++o) this->bias_gradients[o] = 0;
            // 先求 weights, bias 的梯度
            for(int b = 0;b < batch_size; ++b) { // 这个 batch 每张图像对应一个梯度, 多个梯度取平均
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
                                        // 得到这个 w 权值的梯度, 由参与计算的输入和返回的梯度
                                        sum_value += input_ptr[y * stride + k_y] * delta_ptr[y];
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
            // 接下来求输出到输入的梯度 delta_output
            // 第一次经过 backward 给 delta_output 分配内存
            if(this->delta_output.empty()) {
                this->delta_output.reserve(batch_size);
                for(int b = 0;b < batch_size; ++b)
                    this->delta_output.emplace_back(new Tensor3D(in_channels, H, W, this->name + "_delta_" + std::to_string(b)));
            }
            // delta_output 初始化为 0, 这一步不是多余的
            for(int o = 0;o < batch_size; ++o) this->delta_output[o]->set_zero();
            // 翻转 180, padding 那个太难了, 下面直接采用最笨的方法, 用卷积来定位每个输入对应的参与计算的权值 w
            const int radius = (kernel_size - 1) / 2;
            const int H_radius = H - radius;
            const int W_radius = W - radius;
            const int window_range = kernel_size * kernel_size;
            // 多个 batch 的梯度分开算
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
                                }
                            }
                            ++cnt; // 用来定位当前是输出的第几个数
                        }
                    }
                }
            }
            // 返回
            return this->delta_output;
        }

        // 更新梯度
        void update_gradients(const data_type learning_rate=1e-4) {
            assert(not this->weights_gradients.empty());
            // 把梯度更新到 W 和 b
            for(int o = 0;o < out_channels; ++o) {
                data_type* w_ptr = weights[o]->data;
                data_type* wg_ptr = weights_gradients[o]->data;
                // 逐个卷积核做权值更新
                for(int i = 0;i < params_for_one_kernel; ++i)
                    w_ptr[i] -= learning_rate *  wg_ptr[i];
                // 更新 bias
                bias[o] -= learning_rate *  bias_gradients[o];
            }
        }

        // 获取这一层卷积层的参数值
        int get_params_num() const {
            return (this->params_for_one_kernel + 1) * this->out_channels;
        }
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

        std::vector<tensor> forward(const std::vector<tensor>& input) {
            // 获取输入信息
            const int batch_size = input.size();
            const int H = input[0]->H;
            const int W = input[0]->W;
            const int C = input[0]->C;
            // 计算输出的大小
            const int out_H = std::floor(((H - kernel_size + 2 * padding) / step)) + 1;
            const int out_W = std::floor(((W - kernel_size + 2 * padding) / step)) + 1;
            // 第一次经过该池化层(同样 batch_size 如果变得更大, 这个会出问题, 要重新申请)
            if(this->output.empty()) {
                // 给输出分配空间
                this->output.reserve(batch_size);
                for(int b = 0;b < batch_size; ++b)
                    this->output.emplace_back(new Tensor3D(C, out_H, out_W, this->name + "_output_" + std::to_string(b)));
                // 给反向传播的 delta 分配空间
                if(not no_grad) {
                    this->delta_output.reserve(batch_size);
                    for(int b = 0;b < batch_size; ++b)
                        this->delta_output.emplace_back(new Tensor3D(C, H, W, this->name + "_delta_" + std::to_string(b)));
                    // mask 对 batch 的每一张图都分配空间
                    this->mask.reserve(batch_size);
                    for(int b = 0;b < batch_size; ++b)
                        this->mask.emplace_back(std::vector<int>(C * out_H * out_W, 0));
                }
                // 第一次经过这一层, 根据 kernel_size 计算 offset
                int pos = 0;
                for(int i = 0;i < kernel_size; ++i)
                    for(int j = 0;j < kernel_size; ++j)
                        this->offset[pos++] = i * W + j;
            }
            // 如果存在 backward, 每次 forward 要记得把 mask 全部填充为 0
            const int out_length = out_H * out_W;
            int* mask_ptr = nullptr;
            if(not no_grad) {
                const int mask_size = C * out_length;
                for(int b = 0;b < batch_size; ++b) {
                    int* const mask_ptr = this->mask[b].data();
                    for(int i = 0;i < mask_size; ++i) mask_ptr[i] = 0;
                }
            }
            // 开始池化
            const int length = H * W;
            const int H_kernel = H - kernel_size;
            const int W_kernel = W - kernel_size;
            const int window_range = kernel_size * kernel_size;
            for(int b = 0;b < batch_size; ++b) { // batch 的每一张图像对应的特征图分开池化
                // 16 X 111 X 111 → 16 X 55 X 55
                for(int i = 0;i < C; ++i) {  // 每个通道
                    // 现在我拿到了第 b 张图的第 i 个通道, 一个指向内容大小 55 X 55 的指针
                    data_type* const cur_image_features = input[b]->data + i * length;
                    // 第 b 个输出的第 i 个通道的, 同样是指向内容大小 55 X 55 的指针
                    data_type* const output_ptr = this->output[b]->data + i * out_length;
                    // 记录第 b 个输出, 记录有效点在 111 X 111 这个图上的位置, 一共有 55 X 55 个值
                    if(not no_grad) mask_ptr = this->mask[b].data() + i * out_length;
                    int cnt = 0;  // 当前池化输出的位置
                    for(int x = 0; x <= H_kernel; x += step) {
                        data_type* const row_ptr = cur_image_features + x * W; // 获取这个通道图像的第 x 行指针
                        for(int y = 0; y <= W_kernel; y += step) {
                            // 找到局部的 kernel_size X kernel_size 的区域, 找最大值
                            data_type max_value = row_ptr[y];
                            int max_index = 0; // 记录最大值的位置
                            for(int k = 1; k < window_range; ++k) { // 从 1 开始因为 0 已经比过了, max_value = row_ptr[y]
                                data_type comp = row_ptr[y + offset[k]];
                                if(max_value < comp) {
                                    max_value = comp;
                                    max_index = offset[k];
                                }
                            }
                            // 局部最大值填到输出的对应位置上
                            output_ptr[cnt] = max_value;
                            // 如果后面要 backward, 记录 mask
                            if(not no_grad) {
                                max_index += x * W + y; // 第 i 个通道, i * out_H * out_W 为起点的二维平面, 偏移量 max_index
                                mask_ptr[cnt] = i * length + max_index;
                            }
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
            // B X 128 X 6 X 6, 先填 0
            for(int b = 0;b < batch_size; ++b) this->delta_output[b]->set_zero();
            // batch 每张图像, 根据 mask 标记的位置, 把 delta 中的值填到 delta_output 中去
            const int total_length = delta[0]->get_length();
            for(int b = 0;b < batch_size; ++b) {
                int* mask_ptr = this->mask[b].data();
                // 获取 delta 第 b 张输出传回来的梯度起始地址
                data_type* const src_ptr = delta[b]->data;
                // 获取返回到输入的梯度, 第 b 张梯度的起始地址
                data_type* const res_ptr = this->delta_output[b]->data;
                for(int i = 0;i < total_length; ++i)
                    res_ptr[mask_ptr[i]] = src_ptr[i]; // res_ptr 在有效位置 mask_ptr[i] 上填“输出返回来的梯度” src_ptr[i]
            }
            return this->delta_output;
        }
    };


    class ReLU {
    private:
        // 固有属性
        std::string name;
        // 缓冲区, 避免每次重新申请
        std::vector<tensor> output;
    public:
        ReLU(std::string _name) : name(std::move(_name)) {}

        std::vector<tensor> forward(const std::vector<tensor>& input) {
            // 获取图像信息
            const int batch_size = input.size();
            // 如果是第一次经过这一层
            if(output.empty()) {
                // 给输出分配空间
                this->output.reserve(batch_size);
                for(int b = 0;b < batch_size; ++b)
                    this->output.emplace_back(new Tensor3D(input[0]->C, input[0]->H, input[0]->W, this->name + "_" + std::to_string(b)));
            }
            // 只保留 > 0 的部分
            const int total_length = input[0]->get_length();
            for(int b = 0;b < batch_size; ++b) {
                data_type* const src_ptr = input[b]->data;
                data_type* const out_ptr = this->output[b]->data;
                for(int i = 0;i < total_length; ++i)
                    out_ptr[i] = src_ptr[i] >= 0 ? src_ptr[i] : 0;
            }
            return this->output;
        }

        std::vector<tensor> backward(std::vector<tensor>& delta) { // 这个没有 delta_output, 因为形状一模一样, 可以减少一些空间使用
            // 获取信息
            const int batch_size = delta.size();
            // 从这一层的输出中,  < 0 的部分过滤掉
            const int total_length = delta[0]->get_length();
            for(int b = 0;b < batch_size; ++b) {
                data_type* src_ptr = delta[b]->data;
                data_type* out_ptr = this->output[b]->data;
                for(int i = 0;i < total_length; ++i)
                    src_ptr[i] = out_ptr[i] <= 0 ? 0 : src_ptr[i]; // 输出 > 0 的才有梯度从输出 src_ptr 传到输入
            }
            return delta;
        }
    };


    // 一维向量
    class Tensor1D {
    public:
        const int length;   // 向量长度
        data_type* data = nullptr; // 数据指针
    public:
        Tensor1D(const int len) : length(len), data(new data_type[len]) {}
        ~Tensor1D() {
            if(this->data != nullptr) {
                delete this->data;
                this->data = nullptr;
            }
        }
        void print(const std::string message = "") const {
            std::cout << message << "";
            for(int i = 0;i < length; ++i)
                std::cout << data[i] << "  ";
            std::cout << "\n";
        }
        // 找到这个一维向量的最大值
        data_type max() const {
            return this->data[argmax()];
        }
        // 找到这个一维向量最大值的位置
        int argmax() const {
            if(data == nullptr) return 0;
            data_type max_value = this->data[0];
            int max_index = 0;
            for(int i = 1;i < length; ++i)
                if(this->data[i] > max_value) {
                    max_value = this->data[i];
                    max_index = i;
                }
            return max_index;
        }
    };
    using tensor1D = std::shared_ptr<Tensor1D>;

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
        LinearLayer(std::string _name, const int _in_channels, const int _out_channels)
                : in_channels(_in_channels), out_channels(_out_channels),
                  weights(_in_channels * _out_channels, 0),
                  bias(_out_channels) {
            // 随机种子初始化
            std::default_random_engine e(1998);
            std::normal_distribution<float> engine(0.0, 1.0);
            for(int i = 0;i < _out_channels; ++i) bias[i] = engine(e) / random_times;
            const int length = _in_channels * _out_channels;
            for(int i = 0;i < length; ++i) weights[i] = engine(e) / random_times;
        }

        // 做 Wx + b 矩阵运算
        std::vector<tensor1D> forward(const std::vector<tensor>& input) {
            // 获取输入信息
            const int batch_size = input.size();
            const int in_size = input[0]->get_length();
            this->delta_shape = input[0]->get_shape();
            // 清空之前的结果, 重新开始
            std::vector<tensor1D>().swap(this->output);
            for(int b = 0;b < batch_size; ++b)
                this->output.emplace_back(new Tensor1D(out_channels));
            // 记录输入
            if(not no_grad) this->__input = input;
            // batch 每个图象分开算
            for(int b = 0;b < batch_size; ++b) {
                // 矩阵相乘,   dot
                data_type* src_ptr = input[b]->data; // 1 X 4096
                data_type* res_ptr = this->output[b]->data; // 1 X 10
                for(int i = 0;i < out_channels; ++i) {
                    data_type sum_value = 0;
                    for(int j = 0;j < in_channels; ++j)
                        sum_value += src_ptr[j] * this->weights[j * out_channels + i];
                    res_ptr[i] = sum_value + bias[i];
                }
            }
            return this->output;
        }

        std::vector<tensor> backward(const std::vector<tensor1D>& delta) {
            // 获取 delta 信息
            const int batch_size = delta.size();
                        // 第一次回传, 给缓冲区的梯度 W, b 分配空间
            if(this->weights_gradients.empty()) {
                this->weights_gradients.assign(in_channels * out_channels, 0);
                this->bias_gradients.assign(out_channels, 0);
            }
            // 计算 W 的梯度
            for(int i = 0;i < in_channels; ++i) {
                data_type* w_ptr = this->weights_gradients.data() + i * out_channels;
                for(int j = 0;j < out_channels; ++j) {
                    data_type sum_value = 0;
                    for(int b = 0;b < batch_size; ++b)
                        sum_value += this->__input[b]->data[i] * delta[b]->data[j];
                    w_ptr[j] = sum_value / batch_size;
                }
            }
            // 计算 bias 的梯度
            for(int i = 0;i < out_channels; ++i) {
                data_type sum_value = 0;
                for(int b = 0;b < batch_size; ++b)
                    sum_value += delta[b]->data[i];
                this->bias_gradients[i] = sum_value / batch_size;
            }
            // 如果是第一次回传
            if(this->delta_output.empty()) {
                // 分配空间
                this->delta_output.reserve(batch_size);
                for(int b = 0;b < batch_size; ++b)
                    this->delta_output.emplace_back(new Tensor3D(delta_shape, "linear_delta_" + std::to_string(b)));
            }
            // 计算返回的梯度, 大小和 __input 一致
            for(int b = 0;b < batch_size; ++b) {  // 每个 batch
                data_type* src_ptr = delta[b]->data;
                data_type* res_ptr = this->delta_output[b]->data;
                for(int i = 0;i < in_channels; ++i) {  // 每个输入神经元
                    data_type sum_value = 0;
                    data_type* w_ptr = this->weights.data() + i * out_channels;
                    for(int j = 0;j < out_channels; ++j)  // 每个输出都由第 i 个神经元参与计算得到
                        sum_value += src_ptr[j] * w_ptr[j];
                    res_ptr[i] = sum_value;
                }
            }
            // 返回到上一层给的梯度
            return this->delta_output;
        }

        void update_gradients(const data_type learning_rate=1e-4) {
            // 这里要判断一下, 是否空的
            assert(not this->weights_gradients.empty());
            // 梯度更新到权值
            const int total_length = in_channels * out_channels;
            for(int i = 0;i < total_length; ++i) this->weights[i] -= learning_rate *  this->weights_gradients[i];
            for(int i = 0;i < out_channels; ++i) this->bias[i] -= learning_rate *  this->bias_gradients[i];
        }
    };



    inline data_type __exp(const data_type x) {
        if(x >= 88) return FLT_MAX; // 直接返回 float 的最大值, 如果 data_type 换成 double 这个还得改
        else if(x <= -50) return 0.f;
        return std::exp(x);
    }

    // 给 batch_size 个向量, 每个向量 softmax 成多类别的概率
    std::vector<tensor1D> softmax(const std::vector<tensor1D>& input) {
        const int batch_size = input.size();
        const int num_classes = input[0]->length;
        std::vector<tensor1D> output;
        output.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b) {
            tensor1D probs(new Tensor1D(num_classes));
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
            output.emplace_back(probs);
        }
        return output;
    }

    // batch_size 个样本, 每个样本 0, 1, 2 这种, 例如  1 就得到 [0.0, 1.0, 0.0, 0.0]
    inline std::vector<tensor1D> one_hot(const std::vector<int>& labels, const int num_classes) {
        const int batch_size = labels.size();
        std::vector<tensor1D> one_hot_code;
        one_hot_code.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b) {
            tensor1D sample(new Tensor1D(num_classes));
            for(int i = 0;i < num_classes; ++i)
                sample->data[i] = 0;
            assert(labels[b] >= 0 and labels[b] < num_classes);
            sample->data[labels[b]] = 1.0;
            one_hot_code.emplace_back(sample);
        }
        return one_hot_code;
    }

    // 给输出概率 probs, 和标签 label 计算交叉熵损失, 返回损失值和回传的梯度
    std::pair<data_type, std::vector<tensor1D> > cross_entroy_backward(
            const std::vector<tensor1D>& probs, const std::vector<tensor1D>& labels) {
        const int batch_size = labels.size();
        const int num_classes = probs[0]->length;
        std::vector<tensor1D> delta;
        delta.reserve(batch_size);
        data_type loss_value = 0;
        for(int b = 0;b < batch_size; ++b) {
            tensor1D piece(new Tensor1D(num_classes));
            for(int i = 0;i < num_classes; ++i) {
                piece->data[i] = probs[b]->data[i] - labels[b]->data[i];
                loss_value += std::log(probs[b]->data[i]) * labels[b]->data[i];
            }
            delta.emplace_back(piece);
        }
        loss_value = loss_value * (-1.0) / batch_size;
        return std::make_pair(loss_value, delta);
    }


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
        AlexNet(const int num_classes=3)
            : classifier(LinearLayer("linear_1", 6 * 6 * 128, num_classes)) {}

        std::vector<tensor1D> forward(const std::vector<tensor>& input) {
            // 对输入的形状做检查
            assert(input.size() > 0);
            // batch_size X 3 X 224 X 224
            auto conv_output_1 = this->conv_layer_1.forward(input);
            auto relu_output_1 = this->relu_layer_1.forward(conv_output_1);
            // batch_size X 16 X 111 X 111
            auto pool_output_1 = this->max_pool_1.forward(relu_output_1);
            // batch_size X 16 X 55 X 55
            auto conv_output_2 = this->conv_layer_2.forward(pool_output_1);
            auto relu_output_2 = this->relu_layer_2.forward(conv_output_2);
            // batch_size X 32 X 27 X 27
            auto conv_output_3 = this->conv_layer_3.forward(relu_output_2);
            auto relu_output_3 = this->relu_layer_3.forward(conv_output_3);
            // batch_size X 64 X 13 X 13
            auto conv_output_4 = this->conv_layer_4.forward(relu_output_3);
            auto relu_output_4 = this->relu_layer_4.forward(conv_output_4);
            // batch_size X 128 X 6 X 6
            auto output = this->classifier.forward(relu_output_4);
            // batch_size X num_classes
            return output;
        }

        // 梯度反传
        void backward(const std::vector<tensor1D>& delta_start) {
            auto delta = this->classifier.backward(delta_start);
            delta = this->relu_layer_4.backward(delta);
            delta = this->conv_layer_4.backward(delta);
            delta = this->relu_layer_3.backward(delta);
            delta = this->conv_layer_3.backward(delta);
            delta = this->relu_layer_2.backward(delta);
            delta = this->conv_layer_2.backward(delta);
            delta = this->max_pool_1.backward(delta);
            delta = this->relu_layer_1.backward(delta);
            delta = this->conv_layer_1.backward(delta);
        }

        // 这种写法灵活性差点, 新添加一层要改动很多; 后面可以考虑用多态, 存储指针试试
        void update_gradients(const data_type learning_rate=1e-4) {
            this->classifier.update_gradients(learning_rate);
            this->conv_layer_4.update_gradients(learning_rate);
            this->conv_layer_3.update_gradients(learning_rate);
            this->conv_layer_2.update_gradients(learning_rate);
            this->conv_layer_1.update_gradients(learning_rate);
        }
    };


    class ClassificationEvaluator {
    private:
        int correct_num = 0;  // 当前累计的判断正确的样本数目
        int sample_num = 0;   // 当前累计的样本数目
    public:
        ClassificationEvaluator() {}
        // 这一个 batch 猜对了几个
        void compute(const std::vector<int>& predict, const std::vector<int>& labels) {
            const int batch_size = labels.size();  // 这里不能是 predict 的 size, 程序设计问题, 没办法
            for(int b = 0;b < batch_size; ++b)
                if(predict[b] == labels[b])
                    ++this->correct_num;
            this->sample_num += batch_size;
        }
        // 查看累计的正确率
        data_type get() const {
            return this->correct_num * 1.f / this->sample_num;
        }
        // 重新开始统计
        void clear() {
            this->correct_num = this->sample_num = 0;
        }
    };


    // 这个 BatchNorm 是不同通道做, 不具体实现还真不知道, 后面有时间填坑
    class BatchNorm2D {
    private:
        data_type gamma;
    };
}



// 完全可复现, 随机种子定了
// 还需要实现的功能
// 1. 模型参数的存储和加载
// 2. 动量, Adam 这些, 暂时没想到优雅的解决办法
// 3. batch norm 的实现
// 4. dropout 的实现
// 5. 网络结构有点差劲, 虽然可以跑, 凑合用
// 6. 自动求导, 重头戏, 有时间再说


int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    using namespace architectures;

    // 指定一些参数
    const int train_batch_size = 8;
    const int valid_batch_size = 1;
    const int test_batch_size = 1;
    assert(train_batch_size >= valid_batch_size and train_batch_size >= test_batch_size); // 设计问题, train 的 batch 必须更大
    assert(valid_batch_size == 1 and test_batch_size == 1); // 设计问题, 暂时只支持这个
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
    AlexNet network(num_classes);

    // 开始训练
    const int total_iters = 100000;   // 训练 batch 的总数
    const float learning_rate = 1e-4; // 学习率
    const int valid_inters = 1000;     // 验证一次的间隔
    int iters_one_epoch = 500;        // 每个 epoch 多少个 batch, 更新
    float mean_loss = 0.f;            // 平均损失
    float cur_iter = 0;               // 计算平均损失用的
    ClassificationEvaluator train_evaluator;  // 计算累计的准确率
    std::vector<int> predict(train_batch_size, -1); // 存储每个 batch 的预测结果, 和 labels 算准确率用的
    // 开始训练
    for(int iter = 1; iter <= total_iters; ++iter) {
        // 从训练集中采样一个 batch
        const auto sample = train_loader.generate_batch();
        // 送到网络中
        const auto output = network.forward(sample.first);
        // 网络输出经过 softmax 转化成概率
        const auto probs = softmax(output);
        // 输出概率和标签计算交叉熵损失, 返回损失项和梯度
        const auto loss_delta = cross_entroy_backward(probs, one_hot(sample.second, num_classes));
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
        if(iter % iters_one_epoch == 0) {
            cur_iter = 0;
            mean_loss = 0.f;
            train_evaluator.clear();
            printf("\n");
        }
        if(iter % valid_inters == 0) {
            printf("\n开始验证.....\n");
            WithoutGrad guard;
            float mean_valid_loss = 0.f;
            ClassificationEvaluator valid_evaluator;
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
        }
    }

    // 保存
    const std::filesystem::path checkpoints_dir("../checkpoints/AlexNet");
    if(not std::filesystem::exists(checkpoints_dir))
        std::filesystem::create_directories(checkpoints_dir);

    // 加载模型, 推断
    std::cout << "训练结束!\n";
    return 0;
}
