# 简介

2022年初，写的一个简单的 CNN 卷积神经网络训练的 demo，当前版本只是为了验证神经网络前向和反向传播，加深个人理解，暂不涉及高性能优化及模型部署等，目前也只写了 CPU 上的简单实现，后续有计划推出 CPU 优化版本以及 GPU 版本（基于 CUDA）。

知乎博客：https://zhuanlan.zhihu.com/p/468100301

![image-20230208205425138](./imgs/image-20230208205425138.png)

主要实现了一些基础层

- Conv2d 卷积层，不支持 padding
- 最大池化层
- ReLU 层
- Linear 全连接层
- Batch Normalization 层（验证时性能很差，暂未解决）
- Dropout（验证时性能很差，暂未解决）

还有训练流程

- 交叉熵
- 随机梯度下降

还有张量 Tensor 结构的定义、借助 gradCAM 原理可视化等。

# 环境

- Windows 10
- C++17（[TDM-GCC](https://link.zhihu.com/?target=https%3A//jmeubank.github.io/tdm-gcc/download/)）
- CMake 3.17
- OpenCV 4.5
- [数据集](https://github.com/hermosayhl/CNN/tree/main/datasets) 从[cat-dog-panda](https://link.zhihu.com/?target=https%3A//www.kaggle.com/ashishsaxena2209/animal-image-datasetdog-cat-and-panda) 数据集剔除 cat（cat 和 dog 的分类相对较难），然后又从 [CUB-200 bird](https://link.zhihu.com/?target=http%3A//www.vision.caltech.edu/visipedia/CUB-200.html) 数据集中随机抽出 1000 张鸟类图像，凑成三分类的小型数据集。train : valid : test 比例 8:1:1。

# Start

## **Cmake**

进入 [cpu](https://github.com/hermosayhl/CNN/tree/main/cpu) 目录，创建构建目录、生成目录

```bash
mkdir build_cmake
mkdir bin
cd build
```

执行 CMake，生成 Makefiles

```bash
cmake .. -G "MinGW Makefiles"
```

添加 `-G  "MinGW Makefiles"` 是因为我电脑上有 MSVC，会自动优先使用 MSVC，但我更喜欢用 GCC，因此需要加这一条，如果在 linux 上，可以不加这条。

编译构建

```bash
# Windows + GCC
mingw32-make -j4
# Linux + GCC
make -j4
```

我执行的是上一条命令，在 bin 目录下即可看到生成的三个文件

![image-20230208212620325](./imgs/image-20230208212620325.png)

点击 train.exe 即可训练模型

![image-20230208213513653](imgs/train.gif)

点击 inference.exe ，加载训练好的模型，并对单张图像做推理

![image-20230208213627060](imgs/image-20230208213627060.png)

点击 gradCAM.exe，尝试 CNN 可视化效果

![image-20230208213713076](imgs/image-20230208213713076.png)



## Xmake

[Xmake](https://github.com/xmake-io/xmake) 是最近接触的一款很方便的构建工具，构建速度也很快，十分模块化，推荐使用。

先创建 build 目录

```bash
mkdir build
```

指定用 mingw

```bash
xmake f -p mingw
```

构建目标 `cnn_train`

```bash
xmake build cnn_train
```

运行目标 `cnn_train`

```bash
xmake run cnn_train
```



对于 inference 和 gradCAM 同理。