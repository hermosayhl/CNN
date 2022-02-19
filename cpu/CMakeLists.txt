cmake_minimum_required(VERSION 3.19)
project(cnn)

# 添加编译选项
SET(CMAKE_CXX_FLAGS "-std=c++17 -O2")

set(PROJECT_SOURCE_DIR ./)

# 生成文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# include 目录
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 添加可执行文件
add_executable(cnn
                   src/cnn.cpp
#                   src/grad_cam.cpp
#                   src/test/test.cpp
                   src/pipeline.cpp
                   src/data_format.cpp
                   src/relu.cpp
                   src/linear.cpp
                   src/conv2d.cpp
                   src/func.cpp
                   src/pool2d.cpp
                   src/batchnorm2d.cpp
                   src/metrics.cpp
                   src/architectures.cpp
                   src/alexnet.cpp
                   src/dropout.cpp
        )

set(OpenCV_DIR "D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/")
# 寻找 opencv 的库
find_package(OpenCV COMPONENTS core highgui imgproc imgcodecs)
# 添加 opencv 的头文件
include_directories(${OpenCV_INCLUDE_DIRS} )
# 链接 opencv 动态库路径
link_directories(${OpenCV_LIBRARY_DIRS} )

## 设置库文件
target_link_libraries(cnn ${OpenCV_LIBS})