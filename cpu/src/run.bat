chcp 65001
set exe_file=run.exe
set INCLUDE=D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/include
set LIBRAIY=D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/x64/mingw/bin
set DLL=-llibopencv_highgui452 -llibopencv_core452 -llibopencv_imgcodecs452  -llibopencv_imgproc452 -llibopencv_dnn452
set ARGS=-std=c++17 -lpthread -O1
del %exe_file%
g++ %ARGS%  -I%INCLUDE% -ID:/environments/C++/3rdparty/Eigen3/eigen-3.3.9/installed/include/eigen3 -I../include/ -L %LIBRAIY% cnn.cpp  %DLL%  -o %exe_file%
%exe_file%
pause