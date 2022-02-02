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
        Tensor3D() = default;
        Tensor3D(uchar* const img_ptr, const int _C, const int _H, const int _W)
            : C(_C), H(_H), W(_W), data(new data_type[_C * _H * _W]) {
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
        void print_shape() const {
            std::cout << this->C << " x " << this->H << " x " << this->W << "\n";
        }
        void normalize(const std::vector<float>& mean) {}
        ~Tensor3D() noexcept {
            if(this->data != nullptr) {
                delete this->data;
                this->data = nullptr;
                std::cout << "销毁一次\n";
            }
        };
    };
    using tensor = std::shared_ptr<Tensor3D>;

    class DataLoader {
        using batch_type = std::vector< std::pair<tensor, int> >;
    private:
        list_type images_list;
        int images_num;
        const int batch_size;
        const bool augment;
        const bool shuffle;
        const int seed;
        int iter = -1;
        // 缓冲区, 不用每次分配
    public:
        explicit DataLoader(const list_type& _images_list, const int _bs=1, const bool _aug=false, const bool _shuffle=true, const int _seed=212)
                : images_list(_images_list), batch_size(_bs), augment(_aug), shuffle(_shuffle), seed(_seed) {
            this->images_num = this->images_list.size();
        }
        int length() const {return this->images_num;}
        batch_type generate_batch() {
            // 我要开始算有几个 batch, 然后将 batch 组合在一起
            batch_type one_batch;
            one_batch.reserve(this->batch_size);
            for(int i = 0;i < this->batch_size; ++i)
                one_batch.emplace_back(this->__getitem__());
            return one_batch;
        }
    private:
        std::pair<tensor, int> __getitem__() {
            // 获取图像序号
            ++this->iter;
            if(this->iter == this->images_num) {
                this->iter = 0; // 重新开始取, 然后再一次打乱列表
                if(this->shuffle) std::shuffle(this->images_list.begin(), this->images_list.end(), std::default_random_engine(this->seed));
            }
            // 读取图像
            const auto& image_path = this->images_list[this->iter].first;
            const int image_label = this->images_list[this->iter].second;
            cv::Mat origin = cv::imread(image_path);
            // resize
            cv::resize(origin, origin, {224, 224});
            // 对图像做数据增强
            // 将图像转化成 float 数据, 且是 Tensor 形式, C x H x W;
            // 这种方式还是差了一点, 因为每个 batch 都要销毁 tensor, 效率上还是慢了
            return std::pair<tensor, int>(new Tensor3D(origin.data, origin.channels(), origin.rows, origin.cols), image_label);
        }
    };
}



int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    // 指定类别
    std::vector<std::string> categories({"dog", "panda", "bird"});

    // 获取图片
    auto dataset = pipeline::get_images_for_classification(
            "../datasets/animals", categories);

    // 构造数据流
    pipeline::DataLoader train_loader(dataset["train"], 4, false, true);
    for(int i = 0;i < 1000; ++i) {
        auto sample = train_loader.generate_batch();
        for(const auto & it : sample) {
            std::cout << categories[it.second] << std::endl;
            cv_show(it.first->opecv_mat());
        }
        std::cout << i << "...\n";
    }
    return 0;
}


/*
 * //C++
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
        Tensor3D() = default;
        // 分配内存
        Tensor3D(const int _C, const int _H, const int _W)
            : C(_C), H(_H), W(_W), data(new data_type[_C * _H * _W]) {}
        // 从图像指针中读取
        void read_from_opencv_mat(uchar* const img_ptr) {
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
        void print_shape() const {
            std::cout << this->C << " x " << this->H << " x " << this->W << "\n";
        }
        void normalize(const std::vector<float>& mean) {}
        ~Tensor3D() noexcept {
            if(this->data != nullptr) {
                delete this->data;
                this->data = nullptr;
                std::cout << "销毁一次\n";
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
        const int H = 224, W = 224, C = 3; // 允许的图像尺寸
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



int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    // 指定类别
    std::vector<std::string> categories({"dog", "panda", "bird"});

    // 获取图片
    auto dataset = pipeline::get_images_for_classification(
            "../datasets/animals", categories);

    // 构造数据流
    pipeline::DataLoader train_loader(dataset["train"], 4, false, true);
    for(int i = 0;i < 1300; ++i) {
        auto sample = train_loader.generate_batch();
        for(const auto & it : sample) {
            // std::cout << categories[it.second] << std::endl;
            // cv_show(it.first->opecv_mat());
        }
        // std::cout << i << "...\n";
    }
    return 0;
}

 */











/*
 * //C++
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
        void print_shape() const {
            std::cout << this->C << " x " << this->H << " x " << this->W << "\n";
        }
        void normalize(const std::vector<float>& mean) {}
        ~Tensor3D() noexcept {
            if(this->data != nullptr) {
                delete this->data;
                this->data = nullptr;
                std::cout << this->name << " 销毁一次\n";
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
        const int H = 224, W = 224, C = 3; // 允许的图像尺寸
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

    class Conv2D {
    private:
        const std::string name;
        std::vector<tensor> weights; // 卷积核的权值参数, out_channels X in_channels X kernel_size X kernel_size
        std::vector<data_type> b; // 偏置
        const int in_channels;  // 要滤波的特征图有几个通道
        const int out_channels; // 这一层卷积有几个卷积核
        const int kernel_size;  // 卷积核的边长
        const int stride;       // 卷积的步长
        const int padding;      // 是否要 padding, 这个有点麻烦
        const int params_for_one_kernel;   // 参数个数
        std::default_random_engine seed;   // 初始化的种子
    public:
        Conv2D(std::string _name, const int _in_channels=3, const int _out_channels=16, const int _kernel_size=3, const int _stride=2, const int _padding=0)
                : name(std::move(_name)), b(_out_channels), in_channels(_in_channels), out_channels(_out_channels), kernel_size(_kernel_size), stride(_stride), padding(_padding),
                  params_for_one_kernel(_in_channels * _kernel_size * _kernel_size) {
            // 首先给权值矩阵 weights 和偏置 b 分配空间
            this->weights.reserve(out_channels);
            for(int o = 0;o < out_channels; ++o) {
                // 一共有 out_channels 个卷积核, 每个卷积核有 in_channels X kernel_size X kernel_size 个参数
                weights.emplace_back(new Tensor3D(in_channels, kernel_size, kernel_size, this->name + "_" + std::to_string(o)));
            }
            // 随机初始化
            this->seed.seed(212);
            std::normal_distribution<float> engine(0.0, 1.0);
            for(int o = 0;o < out_channels; ++o)
                b[o] = engine(this->seed);
            for(int o = 0;o < out_channels; ++o) {
                data_type* data_ptr = this->weights[o]->data;
                for(int i = 0;i < params_for_one_kernel; ++i) data_ptr[i] = engine(this->seed);
                // std::cout << data_ptr[0] << std::endl;
            }
        }
        // 卷积操作的 forward 过程
        std::vector<tensor> forward(const std::vector<tensor>& input) {
            std::vector<tensor> output;
            return output;
        }
        // 获取这一层卷积层的参数值
        int get_params_num() const {
            return (this->params_for_one_kernel + 1) * this->out_channels;
        }
    };

    class AlexNet {
    private:
         Conv2D conv_layer_1 = Conv2D("conv_layer_1", 3, 16, 3);
    public:
        std::vector<tensor> forward(const std::vector<tensor>& input) {
            auto conv_outout_1 = this->conv_layer_1.forward(input);
            return conv_outout_1;
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
        break;
    }

    // 保存
    const std::filesystem::path checkpoints_dir("./checkpoints/AlexNet");
    if(not std::filesystem::exists(checkpoints_dir))
        std::filesystem::create_directories(checkpoints_dir);

    // 加载模型, 推断
    return 0;
}

 */


































/*
 * 池化层测试完成
 * //C++
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
        const int H = 224, W = 224, C = 3; // 允许的图像尺寸
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
        MaxPool2D max_pool_2 = MaxPool2D("max_pool_2", 2, 2);
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

 */


































/*  ReLU 测试通过
 *
 *
 * //C++
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
            // assert(batch_size == this->delta_output.size());
            // assert(equal_shape(this->output[0]->get_shape(), delta[0]->get_shape()));
            // B X 128 X 6 X 6, 先填 0s
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



    class ReLU {
    private:
        std::string name;
        std::vector<tensor> output;
        std::vector<tensor> delta_output;
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
                // 如果要反向求导, 也给 delta 分配空间
                this->delta_output.reserve(batch_size);
                for(int b = 0;b < batch_size; ++b)
                    this->delta_output.emplace_back(new Tensor3D(input[0]->C, input[0]->H, input[0]->W, this->name + "_delta_" + std::to_string(b)));
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

        std::vector<tensor> backward(const std::vector<tensor>& delta) { // 这个 delta 不必是 const
            // 获取信息
            const int batch_size = delta.size();
            // assert 就算了, 影响性能
            // 从这一层的输出中,  < 0 的部分过滤掉
            const int total_length = delta[0]->get_length();
            for(int b = 0;b < batch_size; ++b) {
                data_type* src_ptr = delta[b]->data;
                data_type* res_ptr = this->delta_output[b]->data;
                data_type* out_ptr = this->output[b]->data;
                for(int i = 0;i < total_length; ++i)
                    res_ptr[i] = out_ptr[i] <= 0 ? 0 : src_ptr[i];
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
        ReLU relu_layer_1 = ReLU("relu_layer_1");
        ReLU relu_layer_2 = ReLU("relu_layer_2");
        ReLU relu_layer_3 = ReLU("relu_layer_3");
        ReLU relu_layer_4 = ReLU("relu_layer_4");
    public:
        std::vector<tensor> forward(const std::vector<tensor>& input) {
            // 对输入的形状做检查
            auto conv_output_1 = this->conv_layer_1.forward(input);
            auto relu_output_1 = this->relu_layer_1.forward(conv_output_1);

            auto pool_output_1 = this->max_pool_1.forward(relu_output_1);

            auto conv_output_2 = this->conv_layer_2.forward(pool_output_1);
            auto relu_output_2 = this->relu_layer_2.forward(conv_output_2);

            auto conv_output_3 = this->conv_layer_3.forward(relu_output_2);
            auto relu_output_3 = this->relu_layer_3.forward(conv_output_3);

            auto conv_output_4 = this->conv_layer_4.forward(relu_output_3);
            auto relu_output_4 = this->relu_layer_4.forward(conv_output_4);

            auto pool_output_2 = this->max_pool_2.forward(relu_output_4);
            // 在这里模拟 pool 层的反向传播
            const int CH = 0;
            conv_output_4[0]->print(CH);
            relu_output_4[0]->print(CH);
            std::vector<tensor> delta;
            for(int b = 0;b < 4; ++b) {
                tensor one(new Tensor3D(128, 7, 7));
                for(int i = 0;i < 128; ++i) {
                    data_type* const res_ptr = one->data + i * 49;
                    for(int k = 1;k <= 49; ++k)
                        res_ptr[k - 1] = 0.1f * k;
                }
                delta.emplace_back(one);
            }
            auto result = this->relu_layer_4.backward(delta);
            result[0]->print(CH);
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

 *
 *
 *
 */



























/*  Linear 线性层完成, 验证了是正确的
 * //C++
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
            // assert(batch_size == this->delta_output.size());
            // assert(equal_shape(this->output[0]->get_shape(), delta[0]->get_shape()));
            // B X 128 X 6 X 6, 先填 0s
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



    class ReLU {
    private:
        std::string name;
        std::vector<tensor> output;
        std::vector<tensor> delta_output;
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
                // 如果要反向求导, 也给 delta 分配空间
                this->delta_output.reserve(batch_size);
                for(int b = 0;b < batch_size; ++b)
                    this->delta_output.emplace_back(new Tensor3D(input[0]->C, input[0]->H, input[0]->W, this->name + "_delta_" + std::to_string(b)));
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

        std::vector<tensor> backward(const std::vector<tensor>& delta) { // 这个 delta 不必是 const
            // 获取信息
            const int batch_size = delta.size();
            // assert 就算了, 影响性能
            // 从这一层的输出中,  < 0 的部分过滤掉
            const int total_length = delta[0]->get_length();
            for(int b = 0;b < batch_size; ++b) {
                data_type* src_ptr = delta[b]->data;
                data_type* res_ptr = this->delta_output[b]->data;
                data_type* out_ptr = this->output[b]->data;
                for(int i = 0;i < total_length; ++i)
                    res_ptr[i] = out_ptr[i] <= 0 ? 0 : src_ptr[i];
            }
            return this->delta_output;
        }
    };


    class Tensor1D {
    public:
        const int length;
        data_type* data = nullptr;
    public:
        Tensor1D(const int len)
            : length(len), data(new data_type[len]) {}
        ~Tensor1D() {
            if(this->data != nullptr) {
                delete this->data;
                this->data = nullptr;
            }
        }
        void print() const {
            for(int i = 0;i < length; ++i)
                std::cout << data[i] << "  ";
            std::cout << "\n";
        }
    };

    using tensor1D = std::shared_ptr<Tensor1D>;

    class LinearLayer {
    public:
        const int in_channels;   // 输入的神经元个数
        const int out_channels;  // 输出的神经元个数
        std::vector<data_type> weights;       // 权值矩阵
        std::vector<data_type> bias;          // 偏置
        // 历史信息
        std::tuple<int, int, int> delta_shape;// 记下来, delta 的形状, 从 1 X 4096 到 128 * 4 * 4 这种
        std::vector<tensor> __input;          // 梯度回传的时候需要输入 Wx + b, 需要保留 x
        // 以下是缓冲区
        std::vector<tensor1D> output;         // 记录输出
        std::vector<tensor> delta_output;     // delta 回传
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
            for(int i = 0;i < _out_channels; ++i) bias[i] = engine(e);
            const int length = _in_channels * _out_channels;
            for(int i = 0;i < length; ++i) weights[i] = engine(e);
        }

        // 这里千万要注意, train 跟 valid, test 的不一样, 真的坑爹, 不能直接判断 empty, 然后分配空间
        std::vector<tensor1D> forward(const std::vector<tensor>& input) {
            // 获取输入信息
            const int batch_size = input.size();
            const int in_size = input[0]->get_length();
            assert(in_size == in_channels);
            this->delta_shape = input[0]->get_shape();
            // 清空之前的结果, 重新开始
            std::vector<tensor1D>().swap(this->output);
            for(int b = 0;b < batch_size; ++b)
                this->output.emplace_back(new Tensor1D(out_channels));
            // 记录输入
            this->__input = input;
            // batch 每个图象分开算
            std::cout << "batch_size  " << batch_size << "\n";
            for(int b = 0;b < batch_size; ++b) {
                // 矩阵相乘,   dot
                data_type* src_ptr = input[b]->data; // 1 X 4096
                data_type* res_ptr = this->output[b]->data; // 1 X 10
                for(int i = 0;i < out_channels; ++i) {
                    data_type* w_ptr = this->weights.data() + i * in_channels; // 4096 * 10
                    data_type sum_value = 0;
                    for(int j = 0;j < in_channels; ++j) {
                        sum_value += src_ptr[j] * this->weights[j * out_channels + i];
                        if(b == 0 and i == 0) {
                            std::cout << src_ptr[j] << " * " << this->weights[j * out_channels + i] << std::endl;
                        }
                    }
                    res_ptr[i] = sum_value + bias[i];
                }
            }
            return this->output;
        }

        std::vector<tensor> backward(const std::vector<tensor1D>& delta) {
            // 获取 delta 信息
            const int batch_size = delta.size();
            // 如果是第一次回传
            if(this->delta_output.empty()) {
                // 分配空间
                this->delta_output.reserve(batch_size);
                for(int b = 0;b < batch_size; ++b)
                    this->delta_output.emplace_back(new Tensor3D(std::get<0>(delta_shape), std::get<1>(delta_shape), std::get<2>(delta_shape), "linear_delta_" + std::to_string(b)));
            }
            // 计算返回的梯度
            // 4 X 10, 10 X 4096, 但这个 4096 的排列不大对
            for(int b = 0;b < batch_size; ++b) {
                data_type* src_ptr = delta[b]->data;
                data_type* res_ptr = this->delta_output[b]->data;
                for(int i = 0;i < in_channels; ++i) {
                    data_type sum_value = 0;
                    data_type* w_ptr = this->weights.data() + i * out_channels;
                    for(int j = 0;j < out_channels; ++j)
                        sum_value += src_ptr[j] * w_ptr[j];
                    res_ptr[i] = sum_value;
                }
            }
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
            for(int i = 0;i < out_channels; ++i) {
                data_type sum_value = 0;
                for(int b = 0;b < batch_size; ++b)
                    sum_value += delta[b]->data[i];
                this->bias_gradients[i] = sum_value / batch_size;
            }
            for(int i = 0;i < in_channels; ++i) {
                for(int j = 0;j < out_channels; ++j)
                    std::cout << this->weights_gradients[i * out_channels + j] << "  ";
                std::cout << "\n";
            }
            for(int i = 0;i < out_channels; ++i) std::cout << bias_gradients[i] << "  ";
            std::cout << "\n";
            // 梯度更新到权值
            const int total_length = in_channels * out_channels;
            for(int i = 0;i < total_length; ++i) this->weights[i] -= 1e-3 * this->weights_gradients[i];
            for(int i = 0;i < out_channels; ++i) this->bias[i] -= 1e-3 * this->bias_gradients[i];
            // 返回到上一层给的梯度
            return this->delta_output;
        }
    };


    // 记得最后开 O1 优化
    class AlexNet {
    private:
        Conv2D conv_layer_1 = Conv2D("conv_layer_1", 3, 16, 3);
        Conv2D conv_layer_2 = Conv2D("conv_layer_2", 16, 32, 3);
        Conv2D conv_layer_3 = Conv2D("conv_layer_3", 32, 64, 3);
        Conv2D conv_layer_4 = Conv2D("conv_layer_4", 64, 128, 3);
        MaxPool2D max_pool_1 = MaxPool2D("max_pool_1", 2, 2);
        MaxPool2D max_pool_2 = MaxPool2D("max_pool_2", 3, 2);
        ReLU relu_layer_1 = ReLU("relu_layer_1");
        ReLU relu_layer_2 = ReLU("relu_layer_2");
        ReLU relu_layer_3 = ReLU("relu_layer_3");
        ReLU relu_layer_4 = ReLU("relu_layer_4");
        LinearLayer classifier;
    public:
        AlexNet(const int num_classes=3)
            : classifier(LinearLayer("linear_1", 9 * 128, num_classes)) {}

        std::vector<tensor1D> forward(const std::vector<tensor>& input) {
            // 对输入的形状做检查
            auto conv_output_1 = this->conv_layer_1.forward(input);
            auto relu_output_1 = this->relu_layer_1.forward(conv_output_1);

            auto pool_output_1 = this->max_pool_1.forward(relu_output_1);

            auto conv_output_2 = this->conv_layer_2.forward(pool_output_1);
            auto relu_output_2 = this->relu_layer_2.forward(conv_output_2);

            auto conv_output_3 = this->conv_layer_3.forward(relu_output_2);
            auto relu_output_3 = this->relu_layer_3.forward(conv_output_3);

            auto conv_output_4 = this->conv_layer_4.forward(relu_output_3);
            auto relu_output_4 = this->relu_layer_4.forward(conv_output_4);

            auto pool_output_2 = this->max_pool_2.forward(relu_output_4);

//            auto output = this->classifier.forward(pool_output_2);

            // 模拟前向和后向传播
            LinearLayer one("linear_2", 4, 3);
            for(int i = 0;i < 12; ++i)
                one.weights[i] = (i + 1) * 0.01;
            for(int i = 0;i < 4; ++i) {
                for(int j = 0;j < 3; ++j)
                    std::cout << one.weights[i * 3 + j] << "  ";
                std::cout << "\n";
            }
            for(int i = 0;i < 3; ++i)
                one.bias[i] = -(i + 1) * 0.05;
            for(int i = 0;i < 3; ++i) std::cout << one.bias[i] << "  ";
            std::cout << "\n";
            // 前向
            std::vector<tensor> demo_in;
            demo_in.emplace_back(new Tensor3D(1, 2, 2));
            for(int i = 0;i < 4; ++i)
                demo_in[0]->data[i] = (i - 1) * 0.3;
            demo_in[0]->print();
            auto demo_out = one.forward(demo_in);
            std::cout << "线性层输出 \n";
            demo_out[0]->print();
            // 反向
            std::vector<tensor1D> delta;
            delta.emplace_back(new Tensor1D(3));
            delta[0]->data[0] = 0.212;
            delta[0]->data[1] = 1.998;
            delta[0]->data[2] = 0.229;
            auto delta_output = one.backward(delta);
            delta_output[0]->print();

            return std::vector<tensor1D>();
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
    const int num_classes = categories.size();
    std::unique_ptr<architectures::AlexNet> network(new architectures::AlexNet(num_classes));

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
        if(iter == 1) break;
    }

    // 保存
    const std::filesystem::path checkpoints_dir("./checkpoints/AlexNet");
    if(not std::filesystem::exists(checkpoints_dir))
        std::filesystem::create_directories(checkpoints_dir);

    // 加载模型, 推断
    std::cout << "训练结束!\n";
    return 0;
}


*/




/*
 *
 * Softmax 层和 CrossEntroy 应该是写对了
 */













/*
 *
 *  Rot180 和  padding 都写好了
 *  //C++
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
        std::vector<tensor> delta_output; // 存储回传到上一层的梯度
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

        std::vector<tensor> backward(const std::vector<tensor>& delta) {
            return this->delta_output;
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
            // assert(batch_size == this->delta_output.size());
            // assert(equal_shape(this->output[0]->get_shape(), delta[0]->get_shape()));
            // B X 128 X 6 X 6, 先填 0s
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



    class ReLU {
    private:
        std::string name;
        std::vector<tensor> output;
        std::vector<tensor> delta_output;
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
                // 如果要反向求导, 也给 delta 分配空间
                this->delta_output.reserve(batch_size);
                for(int b = 0;b < batch_size; ++b)
                    this->delta_output.emplace_back(new Tensor3D(input[0]->C, input[0]->H, input[0]->W, this->name + "_delta_" + std::to_string(b)));
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

        std::vector<tensor> backward(const std::vector<tensor>& delta) { // 这个 delta 不必是 const
            // 获取信息
            const int batch_size = delta.size();
            // assert 就算了, 影响性能
            // 从这一层的输出中,  < 0 的部分过滤掉
            const int total_length = delta[0]->get_length();
            for(int b = 0;b < batch_size; ++b) {
                data_type* src_ptr = delta[b]->data;
                data_type* res_ptr = this->delta_output[b]->data;
                data_type* out_ptr = this->output[b]->data;
                for(int i = 0;i < total_length; ++i)
                    res_ptr[i] = out_ptr[i] <= 0 ? 0 : src_ptr[i];
            }
            return this->delta_output;
        }  // 这里的 ReLU 其实完全可以重新分配一个 delta
    };


    class Tensor1D {
    public:
        const int length;
        data_type* data = nullptr;
    public:
        Tensor1D(const int len)
            : length(len), data(new data_type[len]) {}
        ~Tensor1D() {
            if(this->data != nullptr) {
                delete this->data;
                this->data = nullptr;
            }
        }
        void print(const std::string message = "") const {
            std::cout << message << "===> ";
            for(int i = 0;i < length; ++i)
                std::cout << data[i] << "  ";
            std::cout << "\n";
        }
    };

    using tensor1D = std::shared_ptr<Tensor1D>;

    class LinearLayer {
    private:
        const int in_channels;   // 输入的神经元个数
        const int out_channels;  // 输出的神经元个数
        std::vector<data_type> weights;       // 权值矩阵
        std::vector<data_type> bias;          // 偏置
        // 历史信息
        std::tuple<int, int, int> delta_shape;// 记下来, delta 的形状, 从 1 X 4096 到 128 * 4 * 4 这种
        std::vector<tensor> __input;          // 梯度回传的时候需要输入 Wx + b, 需要保留 x
        // 以下是缓冲区
        std::vector<tensor1D> output;         // 记录输出
        std::vector<tensor> delta_output;     // delta 回传
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
            for(int i = 0;i < _out_channels; ++i) bias[i] = engine(e);
            const int length = _in_channels * _out_channels;
            for(int i = 0;i < length; ++i) weights[i] = engine(e);
        }

        // 这里千万要注意, train 跟 valid, test 的不一样, 真的坑爹, 不能直接判断 empty, 然后分配空间
        std::vector<tensor1D> forward(const std::vector<tensor>& input) {
            // 获取输入信息
            const int batch_size = input.size();
            const int in_size = input[0]->get_length();
            assert(in_size == in_channels);
            this->delta_shape = input[0]->get_shape();
            // 清空之前的结果, 重新开始
            std::vector<tensor1D>().swap(this->output);
            for(int b = 0;b < batch_size; ++b)
                this->output.emplace_back(new Tensor1D(out_channels));
            // 记录输入
            this->__input = input;
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
            // 如果是第一次回传
            if(this->delta_output.empty()) {
                // 分配空间
                this->delta_output.reserve(batch_size);
                for(int b = 0;b < batch_size; ++b)
                    this->delta_output.emplace_back(new Tensor3D(std::get<0>(delta_shape), std::get<1>(delta_shape), std::get<2>(delta_shape), "linear_delta_" + std::to_string(b)));
            }
            // 计算返回的梯度
            // 4 X 10, 10 X 4096, 但这个 4096 的排列不大对
            for(int b = 0;b < batch_size; ++b) {
                data_type* src_ptr = delta[b]->data;
                data_type* res_ptr = this->delta_output[b]->data;
                for(int i = 0;i < in_channels; ++i) {
                    data_type sum_value = 0;
                    data_type* w_ptr = this->weights.data() + i * out_channels;
                    for(int j = 0;j < out_channels; ++j)
                        sum_value += src_ptr[j] * w_ptr[j];
                    res_ptr[i] = sum_value;
                }
            }
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
            for(int i = 0;i < out_channels; ++i) {
                data_type sum_value = 0;
                for(int b = 0;b < batch_size; ++b)
                    sum_value += delta[b]->data[i];
                this->bias_gradients[i] = sum_value / batch_size;
            }
            // 梯度更新到权值
            const int total_length = in_channels * out_channels;
            for(int i = 0;i < total_length; ++i) this->weights[i] -= 1e-3 * this->weights_gradients[i];
            for(int i = 0;i < out_channels; ++i) this->bias[i] -= 1e-3 * this->bias_gradients[i];
            // 返回到上一层给的梯度
            return this->delta_output;
        }
    };



    std::vector<tensor1D> softmax(const std::vector<tensor1D>& input) {
        const int batch_size = input.size();
        const int num_classes = input[0]->length;
        std::vector<tensor1D> output;
        output.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b) {
            tensor1D probs(new Tensor1D(num_classes));
            // 首先算出输出的最大值, 防止溢出, 还是改变不了什么, 大于 -37 直接等于 1
            data_type max_value = input[b]->data[0];
            for(int i = 1;i < num_classes; ++i)   // 这里可以写一个 .max() 函数
                if(input[b]->data[i] > max_value)
                    max_value = input[b]->data[i];
            data_type sum_value = 0;
            for(int i = 0;i < num_classes; ++i) {
                probs->data[i] = std::exp(input[b]->data[i] - max_value);
                sum_value += probs->data[i];
            }
            for(int i = 0;i < num_classes; ++i)
                probs->data[i] /= sum_value;
            output.emplace_back(probs);
        }
        return output;
    }

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


    std::pair<data_type, std::vector<tensor1D> > cross_entroy_backward(
            const std::vector<tensor1D>& probs, const std::vector<tensor1D>& label) {
        const int batch_size = probs.size();
        const int num_classes = probs[0]->length;
        assert(batch_size == label.size() and num_classes == label[0]->length);
        std::vector<tensor1D> delta;
        delta.reserve(batch_size);
        data_type loss_value = 0;
        for(int b = 0;b < batch_size; ++b) {
            tensor1D piece(new Tensor1D(num_classes));
            for(int i = 0;i < num_classes; ++i) {
                piece->data[i] = probs[b]->data[i] - label[b]->data[i];
                loss_value += std::log(probs[b]->data[i]) * label[b]->data[i];
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
        MaxPool2D max_pool_2 = MaxPool2D("max_pool_2", 3, 2);
        ReLU relu_layer_1 = ReLU("relu_layer_1");
        ReLU relu_layer_2 = ReLU("relu_layer_2");
        ReLU relu_layer_3 = ReLU("relu_layer_3");
        ReLU relu_layer_4 = ReLU("relu_layer_4");
        LinearLayer classifier;
    public:
        AlexNet(const int num_classes=3)
            : classifier(LinearLayer("linear_1", 9 * 128, num_classes)) {}

        std::vector<tensor1D> forward(const std::vector<tensor>& input) {
            // 对输入的形状做检查
            auto conv_output_1 = this->conv_layer_1.forward(input);
            auto relu_output_1 = this->relu_layer_1.forward(conv_output_1);

            auto pool_output_1 = this->max_pool_1.forward(relu_output_1);

            auto conv_output_2 = this->conv_layer_2.forward(pool_output_1);
            auto relu_output_2 = this->relu_layer_2.forward(conv_output_2);

            auto conv_output_3 = this->conv_layer_3.forward(relu_output_2);
            auto relu_output_3 = this->relu_layer_3.forward(conv_output_3);

            auto conv_output_4 = this->conv_layer_4.forward(relu_output_3);
            auto relu_output_4 = this->relu_layer_4.forward(conv_output_4);

            auto pool_output_2 = this->max_pool_2.forward(relu_output_4);
            // 4 X 128 * 3 * 3 ===> 4 X 3
            auto output = this->classifier.forward(pool_output_2);
            // 4 X 3
            return softmax(output);
        }

        void backward(const std::vector<tensor1D>& delta_start) {
            // 直接从 softmax 之前开始
            auto delta = this->classifier.backward(delta_start);
            delta = this->max_pool_2.backward(delta);
            delta = this->relu_layer_4.backward(delta);
            // ......
        }
    };
}




int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;


    using namespace architectures;

    // 先写一个旋转 180°的, 然后写一个 padding 空白值的, 速度上会有很大的损伤
    tensor demo(new Tensor3D(2, 5, 5, "demo"));
    const int length = demo->get_length();
    std::cout << length << std::endl;
    for(int i = 0;i < length; ++i)
        demo->data[i] = (i + 1) * 0.01;
    demo->print(1);
    // 开始旋转 180
    auto padded = demo->pad(3);
    padded->print(0);
    padded->print(1);

    return 0;

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
    const int num_classes = categories.size();
    std::unique_ptr<AlexNet> network(new AlexNet(num_classes));

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
            labels[b] = sample[b].second;  // 这里最好改成 pair, 最后再说, 改的漂亮一点
        }
        // 送到网络中
        const auto output = network->forward(images);
        // 根据 labels 设计成 one_hot
        const auto loss_delta = cross_entroy_backward(output, one_hot(labels, num_classes));
        // 计算梯度
        network->backward(loss_delta.second);
        if(iter == 1) break;
    }

    // 保存
    const std::filesystem::path checkpoints_dir("./checkpoints/AlexNet");
    if(not std::filesystem::exists(checkpoints_dir))
        std::filesystem::create_directories(checkpoints_dir);

    // 加载模型, 推断
    std::cout << "训练结束!\n";
    return 0;
}

 *
 *
 */

















































/* CNN 的反向传播跟网上博客都不一样, 不知道我自己算的对不对
 * //C++
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
        std::vector<tensor> delta_output; // 存储回传到上一层的梯度
        std::vector<tensor> input; // 求梯度需要, 其实存储的是指针
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
            this->input = input;
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
            const int batch_size = delta.size();
            assert(batch_size > 0 and batch_size == this->input.size());
            const int out_H = delta[0]->H;
            const int out_W = delta[0]->W;
            const int out_length = out_H * out_W;
            // 获取输入的信息
            const int H = this->input[0]->H;
            const int W = this->input[0]->W;
            // 给缓冲区的梯度分配空间
            if(this->weights_gradients.empty()) {
                // W
                this->weights_gradients.reserve(out_channels);
                for(int o = 0;o < out_channels; ++o)
                    this->weights_gradients.emplace_back(new Tensor3D(in_channels, kernel_size, kernel_size, this->name + "_weights_gradients_" + std::to_string(o)));
                for(int o = 0;o < out_channels; ++o) this->weights_gradients[o]->set_zero();
                // b
                this->bias_gradients.assign(out_channels, 0);
            }
            std::cout << "内存分配成功 " << bias_gradients.size() << std::endl;
            // 先求 W, b 的梯度, 更简单
            // 求 weight, out_channels X in_channels X 3 X 3
            // 对 batch 中的梯度求均值
            const int weight_len = kernel_size * kernel_size;
            for(int b = 0;b < batch_size; ++b) {
                // 首先, 遍历每个卷积核
                for(int o = 0;o < out_channels; ++o) {
//                    std::cout << "o = " << o << std::endl;
                    // 第 b 张梯度第 o 个通道的梯度
                    data_type* o_delta = delta[b]->data + o * out_H * out_W;
                    // 卷积核的每个 in 通道, 分开求
                    for(int i = 0;i < in_channels; ++i) {
                        // 第 b 张第 i 个通道的输入
                        data_type* in_ptr = input[b]->data + i * H * W;
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
                                    }
                                }
                                // 更新到 weight_gradients, 注意除以了 batch_size
//                                std::cout << k_x << " * " << kernel_size << " + " << k_y << " = " << k_x * kernel_size + k_y << std::endl;
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
            // 把梯度更新到 W 和 b
            for(int o = 0;o < out_channels; ++o) {
                data_type* w_ptr = weights[o]->data;
                data_type* wg_ptr = weights_gradients[o]->data;
                for(int i = 0;i < params_for_one_kernel; ++i)
                    w_ptr[i] -= 1e-3 * wg_ptr[i];
                bias[o] -= 1e-3 * bias_gradients[o];
            }
            // 返回
            std::cout << "梯度计算完成 !\n";
            return this->delta_output;
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
            // assert(batch_size == this->delta_output.size());
            // assert(equal_shape(this->output[0]->get_shape(), delta[0]->get_shape()));
            // B X 128 X 6 X 6, 先填 0s
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



    class ReLU {
    private:
        std::string name;
        std::vector<tensor> output;
        std::vector<tensor> delta_output;
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
                // 如果要反向求导, 也给 delta 分配空间
                this->delta_output.reserve(batch_size);
                for(int b = 0;b < batch_size; ++b)
                    this->delta_output.emplace_back(new Tensor3D(input[0]->C, input[0]->H, input[0]->W, this->name + "_delta_" + std::to_string(b)));
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

        std::vector<tensor> backward(const std::vector<tensor>& delta) { // 这个 delta 不必是 const
            // 获取信息
            const int batch_size = delta.size();
            // assert 就算了, 影响性能
            // 从这一层的输出中,  < 0 的部分过滤掉
            const int total_length = delta[0]->get_length();
            for(int b = 0;b < batch_size; ++b) {
                data_type* src_ptr = delta[b]->data;
                data_type* res_ptr = this->delta_output[b]->data;
                data_type* out_ptr = this->output[b]->data;
                for(int i = 0;i < total_length; ++i)
                    res_ptr[i] = out_ptr[i] <= 0 ? 0 : src_ptr[i];
            }
            return this->delta_output;
        }  // 这里的 ReLU 其实完全可以重新分配一个 delta
    };


    class Tensor1D {
    public:
        const int length;
        data_type* data = nullptr;
    public:
        Tensor1D(const int len)
            : length(len), data(new data_type[len]) {}
        ~Tensor1D() {
            if(this->data != nullptr) {
                delete this->data;
                this->data = nullptr;
            }
        }
        void print(const std::string message = "") const {
            std::cout << message << "===> ";
            for(int i = 0;i < length; ++i)
                std::cout << data[i] << "  ";
            std::cout << "\n";
        }
        data_type max() const {
            if(data == nullptr) return 0;
            data_type max_value = this->data[0];
            for(int i = 1;i < length; ++i)
                if(this->data[i] > max_value)
                    max_value = this->data[i];
            return max_value;
        }
    };

    using tensor1D = std::shared_ptr<Tensor1D>;

    class LinearLayer {
    private:
        const int in_channels;   // 输入的神经元个数
        const int out_channels;  // 输出的神经元个数
        std::vector<data_type> weights;       // 权值矩阵
        std::vector<data_type> bias;          // 偏置
        // 历史信息
        std::tuple<int, int, int> delta_shape;// 记下来, delta 的形状, 从 1 X 4096 到 128 * 4 * 4 这种
        std::vector<tensor> __input;          // 梯度回传的时候需要输入 Wx + b, 需要保留 x
        // 以下是缓冲区
        std::vector<tensor1D> output;         // 记录输出
        std::vector<tensor> delta_output;     // delta 回传
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
            for(int i = 0;i < _out_channels; ++i) bias[i] = engine(e);
            const int length = _in_channels * _out_channels;
            for(int i = 0;i < length; ++i) weights[i] = engine(e);
        }

        // 这里千万要注意, train 跟 valid, test 的不一样, 真的坑爹, 不能直接判断 empty, 然后分配空间
        std::vector<tensor1D> forward(const std::vector<tensor>& input) {
            // 获取输入信息
            const int batch_size = input.size();
            const int in_size = input[0]->get_length();
            assert(in_size == in_channels);
            this->delta_shape = input[0]->get_shape();
            // 清空之前的结果, 重新开始
            std::vector<tensor1D>().swap(this->output);
            for(int b = 0;b < batch_size; ++b)
                this->output.emplace_back(new Tensor1D(out_channels));
            // 记录输入
            this->__input = input;
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
            // 如果是第一次回传
            if(this->delta_output.empty()) {
                // 分配空间
                this->delta_output.reserve(batch_size);
                for(int b = 0;b < batch_size; ++b)
                    this->delta_output.emplace_back(new Tensor3D(std::get<0>(delta_shape), std::get<1>(delta_shape), std::get<2>(delta_shape), "linear_delta_" + std::to_string(b)));
            }
            // 计算返回的梯度
            // 4 X 10, 10 X 4096, 但这个 4096 的排列不大对
            for(int b = 0;b < batch_size; ++b) {
                data_type* src_ptr = delta[b]->data;
                data_type* res_ptr = this->delta_output[b]->data;
                for(int i = 0;i < in_channels; ++i) {
                    data_type sum_value = 0;
                    data_type* w_ptr = this->weights.data() + i * out_channels;
                    for(int j = 0;j < out_channels; ++j)
                        sum_value += src_ptr[j] * w_ptr[j];
                    res_ptr[i] = sum_value;
                }
            }
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
            for(int i = 0;i < out_channels; ++i) {
                data_type sum_value = 0;
                for(int b = 0;b < batch_size; ++b)
                    sum_value += delta[b]->data[i];
                this->bias_gradients[i] = sum_value / batch_size;
            }
            // 梯度更新到权值
            const int total_length = in_channels * out_channels;
            for(int i = 0;i < total_length; ++i) this->weights[i] -= 1e-3 * this->weights_gradients[i];
            for(int i = 0;i < out_channels; ++i) this->bias[i] -= 1e-3 * this->bias_gradients[i];
            // 返回到上一层给的梯度
            return this->delta_output;
        }
    };



    std::vector<tensor1D> softmax(const std::vector<tensor1D>& input) {
        const int batch_size = input.size();
        const int num_classes = input[0]->length;
        std::vector<tensor1D> output;
        output.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b) {
            tensor1D probs(new Tensor1D(num_classes));
            // 首先算出输出的最大值, 防止溢出, 还是改变不了什么, 大于 -37 直接等于 1
            data_type max_value = input[b]->data[0];
            for(int i = 1;i < num_classes; ++i)   // 这里可以写一个 .max() 函数
                if(input[b]->data[i] > max_value)
                    max_value = input[b]->data[i];
            data_type sum_value = 0;
            for(int i = 0;i < num_classes; ++i) {
                probs->data[i] = std::exp(input[b]->data[i] - max_value);
                sum_value += probs->data[i];
            }
            for(int i = 0;i < num_classes; ++i)
                probs->data[i] /= sum_value;
            output.emplace_back(probs);
        }
        return output;
    }

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


    std::pair<data_type, std::vector<tensor1D> > cross_entroy_backward(
            const std::vector<tensor1D>& probs, const std::vector<tensor1D>& label) {
        const int batch_size = probs.size();
        const int num_classes = probs[0]->length;
        assert(batch_size == label.size() and num_classes == label[0]->length);
        std::vector<tensor1D> delta;
        delta.reserve(batch_size);
        data_type loss_value = 0;
        for(int b = 0;b < batch_size; ++b) {
            tensor1D piece(new Tensor1D(num_classes));
            for(int i = 0;i < num_classes; ++i) {
                piece->data[i] = probs[b]->data[i] - label[b]->data[i];
                loss_value += std::log(probs[b]->data[i]) * label[b]->data[i];
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
        MaxPool2D max_pool_2 = MaxPool2D("max_pool_2", 3, 2);
        ReLU relu_layer_1 = ReLU("relu_layer_1");
        ReLU relu_layer_2 = ReLU("relu_layer_2");
        ReLU relu_layer_3 = ReLU("relu_layer_3");
        ReLU relu_layer_4 = ReLU("relu_layer_4");
        LinearLayer classifier;
    public:
        AlexNet(const int num_classes=3)
            : classifier(LinearLayer("linear_1", 9 * 128, num_classes)) {}

        std::vector<tensor1D> forward(const std::vector<tensor>& input) {
            // 对输入的形状做检查
            auto conv_output_1 = this->conv_layer_1.forward(input);
            auto relu_output_1 = this->relu_layer_1.forward(conv_output_1);

            auto pool_output_1 = this->max_pool_1.forward(relu_output_1);

            auto conv_output_2 = this->conv_layer_2.forward(pool_output_1);
            auto relu_output_2 = this->relu_layer_2.forward(conv_output_2);

            auto conv_output_3 = this->conv_layer_3.forward(relu_output_2);
            auto relu_output_3 = this->relu_layer_3.forward(conv_output_3);

            auto conv_output_4 = this->conv_layer_4.forward(relu_output_3);
            auto relu_output_4 = this->relu_layer_4.forward(conv_output_4);

            auto pool_output_2 = this->max_pool_2.forward(relu_output_4);
            // 4 X 128 * 3 * 3 ===> 4 X 3
            auto output = this->classifier.forward(pool_output_2);
            // 4 X 3
            return softmax(output);
        }

        void backward(const std::vector<tensor1D>& delta_start) {
            // 直接从 softmax 之前开始
            auto delta = this->classifier.backward(delta_start);
            delta = this->max_pool_2.backward(delta);
            delta = this->relu_layer_4.backward(delta);
            delta[0]->print_shape();
            delta = this->conv_layer_4.backward(delta);
            // ......
        }
    };
}




int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;


    using namespace architectures;

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
    const int num_classes = categories.size();
    std::unique_ptr<AlexNet> network(new AlexNet(num_classes));

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
            labels[b] = sample[b].second;  // 这里最好改成 pair, 最后再说, 改的漂亮一点
        }
        // 送到网络中
        const auto output = network->forward(images);
        // 根据 labels 设计成 one_hot
        const auto loss_delta = cross_entroy_backward(output, one_hot(labels, num_classes));
        // 计算梯度
        network->backward(loss_delta.second);
        if(iter == 1) break;
    }

    // 保存
    const std::filesystem::path checkpoints_dir("./checkpoints/AlexNet");
    if(not std::filesystem::exists(checkpoints_dir))
        std::filesystem::create_directories(checkpoints_dir);

    // 加载模型, 推断
    std::cout << "训练结束!\n";
    return 0;
}
*/