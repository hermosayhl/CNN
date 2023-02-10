-- 设置工程名
set_project("cnn")
-- 设置工程版本
set_version("0.0.1")
-- 设置 xmake 版本
set_xmakever("2.1.0")
-- 设置支持的平台
set_allowedplats("windows", "mingw", "linux")
-- 指定 build 目录
-- set_targetdir("./build_xmake")
-- set_objectdir("./build_xmake/.objs") 



-- 编译共有的一些文件, 避免每个目标都重新编译
target("cnn_layers")
    -- 生成类型, 共享库
    set_kind("shared")
    -- 设置 C/C++ 标准
    set_languages("c99", "cxx17")
    -- 开启优化
    if is_mode("release") then
        set_optimize("fastest")
    end
    -- 添加源文件, 除了 inference.cpp 和 grad_cam.cpp
    add_files("$(projectdir)/src/*.cpp|cnn.cpp|inference.cpp|grad_cam.cpp")
    -- 添加 include 自己的代码
    add_includedirs("$(projectdir)/include/")
    -- 设置第三方库
    --  1. 添加 OpenCV
    if is_os("windows") then
        opencv_root    = "F:/liuchang/environments/OpenCV/4.5.5/opencv-4.5.5/build/install/"
        opencv_version = "455"
    end
    add_includedirs(opencv_root .. "include")
    add_linkdirs(opencv_root .. "x64/mingw/bin")
    add_links(
        "libopencv_core" .. opencv_version, 
        "libopencv_highgui" .. opencv_version, 
        "libopencv_imgproc" .. opencv_version,  
        "libopencv_imgcodecs" .. opencv_version
    )
target_end()



-- 训练的目标
target("cnn_train")
    -- 设置生成类型, 可执行文件
    set_kind("binary")
    -- 开启警告
    set_warnings("all")
    -- 设置 C/C++ 标准
    set_languages("c99", "cxx17")
    -- 添加源文件, 含有 main 函数入口
    add_files("$(projectdir)/src/cnn.cpp")
    -- 添加 include 自己的代码
    add_includedirs("$(projectdir)/include/")
    --  1. 添加 OpenCV
    if is_os("windows") then
        opencv_root    = "F:/liuchang/environments/OpenCV/4.5.5/opencv-4.5.5/build/install/"
    end
    add_includedirs(opencv_root .. "include")
    -- 这里要再链接一次的原因是, cnn_layers 动态库中只有 OpenCV 动态库的入口, 不是完整的拷贝????
    add_linkdirs(opencv_root .. "x64/mingw/bin")
    add_links(
        "libopencv_core" .. opencv_version, 
        "libopencv_highgui" .. opencv_version, 
        "libopencv_imgproc" .. opencv_version,  
        "libopencv_imgcodecs" .. opencv_version
    )
    -- 加载动态库, 链接到可执行文件 
    add_deps("cnn_layers")
    -- 设置目标工作目录
    set_rundir("$(projectdir)/bin")
-- 结束 cnn_train
target_end()





-- 推理的目标
target("cnn_infer")
    -- 设置生成类型, 可执行文件
    set_kind("binary")
    -- 开启警告
    set_warnings("all")
    -- 设置 C/C++ 标准
    set_languages("c99", "cxx17")
    -- 添加源文件
    add_files("$(projectdir)/src/inference.cpp")
    -- 添加 include 自己的代码
    add_includedirs("$(projectdir)/include/")
    --  1. 添加 OpenCV
    if is_os("windows") then
        opencv_root    = "F:/liuchang/environments/OpenCV/4.5.5/opencv-4.5.5/build/install/"
    end
    add_includedirs(opencv_root .. "include")
    add_linkdirs(opencv_root .. "x64/mingw/bin")
    add_links(
        "libopencv_core" .. opencv_version, 
        "libopencv_highgui" .. opencv_version, 
        "libopencv_imgproc" .. opencv_version,  
        "libopencv_imgcodecs" .. opencv_version
    )
    -- 加载动态库, 链接到可执行文件 
    add_deps("cnn_layers")
    -- 设置目标工作目录
    set_rundir("$(projectdir)/bin")
-- 结束 cnn_infer
target_end()





-- 可视化的目标
target("cnn_visualize")
    -- 设置生成类型, 可执行文件
    set_kind("binary")
    -- 开启警告
    set_warnings("all")
    -- 设置 C/C++ 标准
    set_languages("c99", "cxx17")
    -- 添加源文件
    add_files("$(projectdir)/src/grad_cam.cpp")
    -- 添加 include 自己的代码
    add_includedirs("$(projectdir)/include/")
    --  1. 添加 OpenCV
    if is_os("windows") then
        opencv_root    = "F:/liuchang/environments/OpenCV/4.5.5/opencv-4.5.5/build/install/"
    end
    add_includedirs(opencv_root .. "include")
    add_linkdirs(opencv_root .. "x64/mingw/bin")
    add_links(
        "libopencv_core" .. opencv_version, 
        "libopencv_highgui" .. opencv_version, 
        "libopencv_imgproc" .. opencv_version,  
        "libopencv_imgcodecs" .. opencv_version
    )
    -- 加载动态库, 链接到可执行文件 
    add_deps("cnn_layers")
    -- 设置目标工作目录
    set_rundir("$(projectdir)/bin")
-- 结束 cnn_visualize
target_end()