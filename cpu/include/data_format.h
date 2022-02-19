#ifndef CNN_DATA_FORMAT_H
#define CNN_DATA_FORMAT_H



// OpenCV
#include <opencv2/core.hpp>


using data_type = float;
class Tensor3D {
public:
    const int C, H, W;
    data_type* data;
    std::string name;
    // 形状 C x H x W, 分配内存
    Tensor3D(const int _C, const int _H, const int _W, const std::string _name="pipeline")
        : C(_C), H(_H), W(_W), data(new data_type[_C * _H * _W]), name(std::move(_name)) {}
    // 形状 C x H x W, 分配内存
    Tensor3D(const std::tuple<int, int, int>& shape, const std::string _name="pipeline")
        : C(std::get<0>(shape)), H(std::get<1>(shape)), W(std::get<2>(shape)),
          data(new data_type[std::get<0>(shape) * std::get<1>(shape) * std::get<2>(shape)]),
          name(std::move(_name)) {}
    // 形状 length x 1 x 1, 此时length = C, 全连接层用得到
    Tensor3D(const int length, const std::string _name="pipeline")
        : C(length), H(1), W(1), data(new data_type[length]), name(std::move(_name)) {}
    // 从图像指针中读取内容
    void read_from_opencv_mat(const uchar* const img_ptr);
    // 清零
    void set_zero();
    // 找最大值
    data_type max() const;
    int argmax() const;
    // 找最小值
    data_type min() const;
    int argmin() const;
    void div(const data_type times);
    void normalize(const std::vector<data_type> mean={0.406, 0.456, 0.485}, const std::vector<data_type> std_div={0.225, 0.224, 0.229});
    // 从 tensor 恢复成图像
    cv::Mat opecv_mat(const int CH=3) const;
    // 获取这个 Tensor 的内容长度
    int get_length() const;
    // 获取这个 Tensor 的形状
    std::tuple<int, int, int> get_shape() const;
    // 打印这个 Tensor 的形状
    void print_shape() const;
    // 打印这个 Tensor 在第 _C 个通道的内容
    void print(const int _C=0) const;
    std::shared_ptr<Tensor3D> rot180() const;
    std::shared_ptr<Tensor3D> pad(const int padding=1) const;
    ~Tensor3D() noexcept;
};
using tensor = std::shared_ptr<Tensor3D>;





#endif //CNN_DATA_FORMAT_H
