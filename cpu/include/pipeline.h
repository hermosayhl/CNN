#ifndef CNN_PIPELINE_H
#define CNN_PIPELINE_H



// C++
#include <map>
#include <random>
#include <filesystem>
// self
#include "data_format.h"


namespace pipeline {

    using list_type = std::vector<std::pair<std::string, int> >;
    // 从文件夹 dataset_path, 按照 categories 得到不同类别的图像列表
    std::map<std::string, list_type> get_images_for_classification(
            const std::filesystem::path dataset_path,
            const std::vector<std::string> categories={},
            const std::pair<float, float> ratios={0.8, 0.1});

    // 这个特别慢 !!!!, 如果有数据增强的话, 速度降低为原来的 1/4
    class ImageAugmentor {
    private:
        std::default_random_engine e, l, c, r; // e 用来获得操作的概率; l 用来打乱操作列表; c 用来得到裁剪需要的概率; r 用来得到旋转的概率
        std::uniform_real_distribution<float> engine;
        std::uniform_real_distribution<float> crop_engine;
        std::uniform_real_distribution<float> rotate_engine;
        std::uniform_int_distribution<int> minus_engine;
        std::vector<std::pair<std::string, float> > ops;
    public:
        ImageAugmentor(const std::vector<std::pair<std::string, float> >& _ops={{"hflip", 0.5}, {"vflip", 0.2}, {"crop", 0.7}, {"rotate", 0.5}})
            : e(212), l(826), c(320), r(520),
            engine(0.0, 1.0), crop_engine(0.0, 0.25), rotate_engine(15, 75), minus_engine(1, 10),
            ops(std::move(_ops)) {}
        void make_augment(cv::Mat& origin, const bool show=false);
    };

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
        explicit DataLoader(const list_type& _images_list, const int _bs=1, const bool _aug=false, const bool _shuffle=true, const std::tuple<int, int, int> image_size={224, 224, 3},  const int _seed=212);
        int length() const;
        batch_type generate_batch();
    private:
        // 获取第 batch_index 的图像信息, 填充成 tensor
        std::pair<tensor, int> add_to_buffer(const int batch_index);
        // 图像增强
        ImageAugmentor augmentor;
    };
}


#endif //CNN_PIPELINE_H
