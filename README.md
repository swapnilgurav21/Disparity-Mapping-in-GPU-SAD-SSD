# Disparity-Mapping-in-GPU-SAD-SSD

Instructions to run thsi code in Visual Studio Community 2019
1. Download NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
2. Download Boost library from https://sourceforge.net/projects/boost/files/boost-binaries/1.76.0_b1/
	Select boost_1_76_0_b1-msvc-14.2-64.exe from the list and download it
3. Do not forget to memorize the paths of boost and CUDA folders created after installation. It will be
        required in overwriting the BOOST and CUDA lib paths in CMakeLists.txt
5. Extract the Opencl-ex1.zip into Opencl-ex1 folder
6. Open Visual Studio-> Choose "Open a local folder" -> and select "Opencl-ex1" folder
7. Go to CMakeLists.txt which is in "Opencl-ex1" folder, do not confuse it with one in the "out" folder!
8. IN CMakeLists.txt change the Boost include and lib path if it is not the same like in my path.
9. In Opencl-ex1.cpp file change the path to cl file.
10. Run the program in Both Debug and Release mode.
11. In case there is no Release mode in the list you can create one, Manage Configurations-> Press "+" sign
    and choose x64-Release

