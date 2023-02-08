

target("cnn_train")
	set_kind("binary")
	-- 添加跟 possion_editing 有关的 cpp 源文件
	add_files("$(projectdir)/src/*.cpp|inference.cpp|grad_cam.cpp")
	-- 设置编译期跟链接器
	set_toolset("cxx", "g++")
    set_toolset("ld", "g++")
    -- 设置 C/C++ 标准
    set_languages("c99", "cxx17")
    -- 开启警告
    set_warnings("all")
    -- 开启优化
    set_optimize("fastest")
    -- 添加 include 自己的代码
    add_includedirs("$(projectdir)/include/")
    -- 设置第三方库
    -- 	1. 添加 OpenCV
    local opencv_root = "F:/liuchang/environments/OpenCV/4.5.5/opencv-4.5.5/build/install/"
    add_includedirs(opencv_root .. "include")
    add_linkdirs(opencv_root .. "x64/mingw/bin")
    add_links("libopencv_core455", "libopencv_highgui455", "libopencv_imgproc455", "libopencv_imgcodecs455")
    -- 设置目标工作目录
    set_rundir("$(projectdir)/bin")
