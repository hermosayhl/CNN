-- 设置工程名
set_project("cnn")
-- 设置工程版本
set_version("0.0.1")
-- 设置 xmake 版本
set_xmakever("2.4.0")
-- 设置支持的平台
set_allowedplats("windows", "mingw", "linux", "other")
-- 指定 build 目录
-- set_targetdir("./build_xmake")
-- set_objectdir("./build_xmake/.objs") 



-- 添加 opencv 支持
function add_opencv_support()
    opencv_root    = ""
    opencv_version = ""
    -- 如果是 windows 平台
    if is_plat("windows") then
        opencv_root    = "F:/liuchang/environments/OpenCV/4.7.0-win/build"
        opencv_version = "470"
        add_linkdirs(opencv_root .. "/x64/vc16/lib")
        add_includedirs(opencv_root .. "/include")
        print("windows using MSVC")
    else
        -- 如果是 linux 平台
        if is_plat("linux") then
            opencv_root    = "/home/dx/usrs/liuchang/tools/opencv/build/install"
            opencv_version = "452"
            add_includedirs(opencv_root .. "/include/opencv4")
            add_linkdirs(opencv_root .. "/lib")
            print("linux using GCC")
        end
        -- 如果是 MinGW 平台
        if is_plat("mingw") then
            opencv_root    = "F:/liuchang/environments/OpenCV/4.5.5/opencv-4.5.5/build/install"
            opencv_version = "455"
            add_linkdirs(opencv_root .. "/x64/mingw/bin")
            add_includedirs(opencv_root .. "/include")
            print("windows using MinGW")
        end
    end
        
    -- 这里要再链接一次的原因是, cnn_layers 动态库中只有 OpenCV 动态库的入口, 不是完整的拷贝
    if is_plat("windows") then
        add_links("opencv_world" .. opencv_version)
    else
        add_links(
            "libopencv_core" .. opencv_version, 
            "libopencv_highgui" .. opencv_version, 
            "libopencv_imgproc" .. opencv_version,  
            "libopencv_imgcodecs" .. opencv_version
        )
    end
end




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
    add_opencv_support()
target_end()




-- 使用一些相同的操作
function use_default_config()
    -- 设置生成类型, 可执行文件
    set_kind("binary")
    -- 开启警告
    set_warnings("all")
    -- 设置 C/C++ 标准
    set_languages("c99", "cxx17")
    -- 添加 include 自己的代码
    add_includedirs("$(projectdir)/include/")
    -- 添加 OpenCV 支持
    add_opencv_support()
    -- 加载动态库, 链接到可执行文件 
    add_deps("cnn_layers")
    -- 设置目标工作目录
    set_rundir("$(projectdir)/bin")
end




-- 训练的目标
target("cnn_train")
    
    -- 添加源文件, 含有 main 函数入口
    add_files("$(projectdir)/src/cnn.cpp")

    -- 执行相同的操作
    use_default_config()
    
-- 结束 cnn_train
target_end()





-- 推理的目标
target("cnn_infer")

    -- 添加源文件
    add_files("$(projectdir)/src/inference.cpp")

    -- 执行相同的操作
    use_default_config()

-- 结束 cnn_infer
target_end()




-- 可视化的目标
target("cnn_visualize")

    -- 添加源文件
    add_files("$(projectdir)/src/grad_cam.cpp")

    -- 执行相同的操作
    use_default_config()

-- 结束 cnn_visualize
target_end()