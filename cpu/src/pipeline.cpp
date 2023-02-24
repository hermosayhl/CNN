// C++
#include <iostream>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// self
#include "pipeline.h"



namespace {
    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
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


using namespace pipeline;


void ImageAugmentor::make_augment(cv::Mat& origin, const bool show) {
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
            if(show == true) cv_show(origin);
        }
    }
}



std::map<std::string, pipeline::list_type> pipeline::get_images_for_classification(
        const std::filesystem::path dataset_path,
            const std::vector<std::string> categories,
            const std::pair<float, float> ratios) {
    // 遍历 dataset_path 文件夹下指定的类别
    list_type all_images_list;
    const int categories_num = categories.size();
    for(int i = 0;i < categories_num; ++i) {
        const auto images_dir = dataset_path / categories[i];
        assert(std::filesystem::exists(images_dir) && std::string(images_dir.string() + " 路径不存在!").c_str());
        auto walker = std::filesystem::directory_iterator(images_dir);
        for(const auto& iter : walker)
            all_images_list.emplace_back(iter.path().string(), i);
    }
    // 打乱图像列表
    std::shuffle(all_images_list.begin(), all_images_list.end(), std::default_random_engine(212));
    // 将数据集划分成三部分
    const int total_size = all_images_list.size();
    assert(ratios.first > 0 && ratios.second > 0 && ratios.first + ratios.second < 1);
    const int train_size = int(total_size * ratios.first);
    const int test_size = int(total_size * ratios.second);
    std::map<std::string, list_type> results;
    results.emplace("train", list_type(all_images_list.begin(), all_images_list.begin() + train_size));
    results.emplace("test", list_type(all_images_list.begin() + train_size, all_images_list.begin() + train_size + test_size));
    results.emplace("valid", list_type(all_images_list.begin() + train_size + test_size, all_images_list.end()));
    std::cout << "train  :  " << results["train"].size() << "\n" << "test   :  " << results["test"].size() << "\n" << "valid  :  " << results["valid"].size() << "\n";
    return results;
}



DataLoader::DataLoader(const list_type& _images_list, const int _bs, const bool _aug, const bool _shuffle, const std::tuple<int, int, int> image_size,  const int _seed)
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

int DataLoader::length() const {return this->images_num;}

DataLoader::batch_type DataLoader::generate_batch() {
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

// 获取第 batch_index 的图像信息, 填充成 tensor
std::pair<tensor, int> DataLoader::add_to_buffer(const int batch_index) {
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

