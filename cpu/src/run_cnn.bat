chcp 65001
set exe_file=backup.exe
set INCLUDE=D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/include
set LIBRAIY=D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/x64/mingw/bin
set DLL=-llibopencv_highgui452 -llibopencv_core452 -llibopencv_imgcodecs452  -llibopencv_imgproc452 -llibopencv_dnn452
set ARGS=-std=c++17 -lpthread -O1
del %exe_file%
g++ %ARGS%  -I%INCLUDE%  -I../include/ -L %LIBRAIY% ./backup/cnn.cpp  %DLL%  -o %exe_file%
%exe_file%
pause