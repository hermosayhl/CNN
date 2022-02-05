#ifndef CNN_DATA_FORMAT_H
#define CNN_DATA_FORMAT_H



// OpenCV
#include <opencv2/core.hpp>


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
    Tensor3D(const int length, const std::string _name="pipeline")
        : C(length), H(1), W(1), data(new data_type[length]), name(std::move(_name)) {}
    // 从图像指针中读取
    void read_from_opencv_mat(const uchar* const img_ptr);
    void set_zero();
    data_type max() const;
    int argmax() const;
    void div(const data_type times);
    void normalize(const std::vector<data_type> mean={0.406, 0.456, 0.485}, const std::vector<data_type> std_div={0.225, 0.224, 0.229});
    cv::Mat opecv_mat() const;
    int get_length() const;
    std::tuple<int, int, int> get_shape() const;
    void print_shape() const;
    void print(const int _C=0) const;
    std::shared_ptr<Tensor3D> rot180() const;
    std::shared_ptr<Tensor3D> pad(const int padding=1) const;
    ~Tensor3D() noexcept;
};
using tensor = std::shared_ptr<Tensor3D>;





#endif //CNN_DATA_FORMAT_H
