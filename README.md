# Disparity-Mapping-in-GPU-SAD-SSD

Instructions to run this code in Visual Studio Community 2019
1.  Download NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
2.  Download Boost library from https://sourceforge.net/projects/boost/files/boost-binaries/1.76.0_b1/
	Select boost_1_76_0_b1-msvc-14.2-64.exe from the list and download it
3.  Do not forget to memorize the paths of boost and CUDA folders created after installation. It will be
        required in overwriting the BOOST and CUDA lib paths in CMakeLists.txt
5.  Download the project from our git repsository 'Disparity-Mapping-in-GPU-SAD-SSD'.
6.  Extract the out.zip file and add it to the "Opencl-ex1" folder.
7.  Open Visual Studio-> Choose "Open a local folder" -> and select "Opencl-ex1" folder.
8.  In CMakeLists.txt change the Boost include and lib path as per your file locations.
9.  In project.cpp file change the path to cl file and the path to the input images file to the correct
	path according to your file location.
10. The output should look like in the result section below.

# Input Images

![Bowling_L](https://user-images.githubusercontent.com/65502010/126905013-951a25dd-c281-49a0-96d8-b56c6830669f.jpg)
![Bowling_R](https://user-images.githubusercontent.com/65502010/126905025-82559f67-ae0f-4480-9b64-3b4a8da4698e.jpg)
	
 	Image 1 : Bowling Left Image, Image 2 : Bowling Right Image
# Result

![output_disparity_cpuSAD](https://user-images.githubusercontent.com/65502010/126905037-66c35eb1-cbd8-4822-849b-2bf0e1a3857b.jpg)
![output_disparity_cpuSSD](https://user-images.githubusercontent.com/65502010/126905041-11d142cd-c758-4eb0-85a3-8b6d5dd1bc25.jpg)

	Image 1 :Output CPU SAD	, Image 2 :Output CPU SSD
![output_disparity_gpu_1](https://user-images.githubusercontent.com/65502010/126905043-4efb934a-261c-48b4-9452-a043c2ae6580.jpg)
![output_disparity_gpu_2](https://user-images.githubusercontent.com/65502010/126905045-b00527ef-e7f4-43c1-8851-63dfb87a2c3c.jpg)

	Image 1 :Output GPU SAD	, Image 2 :Output GPU SSD
<img width="347" alt="Bowling window size 13" src="https://user-images.githubusercontent.com/65502010/126905049-4090de7d-8bde-4cb4-b27d-3b0fa0ea8751.PNG">

	Output Print statement of perfromance analysis between CPU and GPU implementation




