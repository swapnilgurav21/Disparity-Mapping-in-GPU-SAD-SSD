
// includes
#include <stdio.h>
#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

#include <boost/lexical_cast.hpp>

////////////////////////////////////////////////////////////////////////////// // CPU implementation //////////////////////////////////////////////////////////////////////////////
int getIndexGlobal(std::size_t countX, int i, int j)
{
		return j * countX + i;
}
// Read value from global array a, return 0 if outside image
float getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j)
{
	if (i < 0 || (size_t) i >= countX || j < 0 || (size_t) j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}

void disparityMapCalc(const std::vector<float>& img1, const std::vector<float>& img2, std::vector<float>& disparity, size_t width, size_t height, size_t countX, size_t countY)
{
int window_size = 11;
float SSD_min = 99999;
float SSD = 0;
float disp = 0;
float pixel_x = 0.0;
float final_disparity = 0.0;
int disparity_range = 100;
int x = 0;
int y = 0;
int w = 0;
int h = 0;

	for (x = 0; x < height; x++)
	{

		for(y=0; y < width; y++)
		{
			SSD_min = 99999.0;
			for (disp = 0; disp <= disparity_range; disp++)
			{
				SSD = 0;
				for(w = 0; w < window_size; w++)
				{
					for(h = 0; h < window_size; h++)
					{
						pixel_x = abs (getValueGlobal(img1,countX,countY,w+x, h+y) - getValueGlobal(img2,countX,countY, w+x-disp, h+y));
						pixel_x = pixel_x*pixel_x;
						SSD = SSD + pixel_x;
					}
				}

				if (SSD_min > SSD)
				{
					SSD_min = SSD;
					final_disparity = (float)disp;
					final_disparity = (float)(final_disparity)/disparity_range;

				}
			}

			disparity[width*y + x] = final_disparity;


		}

	}
}

//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	// Create a context	
	//cl::Context context(CL_DEVICE_TYPE_GPU);
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}

cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[platformId] (), 0, 0 };
std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
cl::Context context(CL_DEVICE_TYPE_GPU, prop);

// Declare necessary constants
std::size_t wgSizeX = 16; // Number of work items per work group in X direction
std::size_t wgSizeY = 16;
std::size_t countX = wgSizeX * 24; // Overall number of work items in X direction = Number of elements in X direction
std::size_t countY = wgSizeY * 18;

std::size_t count = countX * countY; // Overall number of elements
std::size_t size = count * sizeof (float); // Size of data in bytes

//Allocate space for input and output data from CPU and GPU.
std::vector<float> h_input1 (count);
std::vector<float> h_input2 (count);
std::vector<float> h_outputGpu (count);
std::vector<float> h_outputCpu (count);

// Get a device of the context
int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
std::cout << "Using device " << deviceNr << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
ASSERT (deviceNr > 0);
ASSERT ((size_t) deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
std::vector<cl::Device> devices;
devices.push_back(device);
OpenCL::printDeviceInfo(std::cout, device);

// Create a command queue
cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

// Load the source code
cl::Program program = OpenCL::loadProgramSource(context, "D:/College/2nd sem/GPU_LAB/exercise/Opencl-ex1 (1)/Opencl-ex1/Opencl-ex1/OpenCLExercise_DisparityMap.cl");
// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
OpenCL::buildProgram(program, devices);

// Creation of Image - inputs
cl::Image2D image1(context,CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT),countX, countY);
cl::Image2D image2(context,CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT),countX, countY);

// Buffer creations
cl::Buffer output_matching_cost1(context, CL_MEM_READ_WRITE, (countX*countY)* sizeof(int) );
cl::Buffer output_disparity(context, CL_MEM_READ_WRITE, (countX*countY)*sizeof(int));
cl::Buffer d_outputGpu(context, CL_MEM_READ_WRITE, (countX*countY)*sizeof(int));

cl::size_t<3> origin;
origin[0] = 0;
origin[1] = 0;
origin[2] = 0;
cl::size_t<3> region;
region[0] = countX;
region[1] = countY;
region[2] = 1;

memset(h_input1.data(), 255, size);
memset(h_input2.data(), 255, size);
memset(h_outputGpu.data(), 255, size);
memset(h_outputCpu.data(), 255, size);

//////////////////////////////////////////////////////////////////////////////////
//		Read input images and set the data in respective buffers.
//////////////////////////////////////////////////////////////////////////////////

std::vector<float> inputImage0Data;
std::vector<float> inputImage1Data;
std::size_t inputWidth0, inputHeight0, inputWidth1, inputHeight1;
Core::readImagePGM("D:/College/2nd sem/GPU_LAB/exercise/Opencl-ex1 (1)/Opencl-ex1/Opencl-ex1/images/Teddy_L.pgm", inputImage0Data, inputWidth0, inputHeight0);
Core::readImagePGM("D:/College/2nd sem/GPU_LAB/exercise/Opencl-ex1 (1)/Opencl-ex1/Opencl-ex1/images/Teddy_R.pgm", inputImage1Data, inputWidth1, inputHeight1);

for (size_t j = 0; j < countY; j++)
{
    for (size_t i = 0; i < countX; i++)
    {
        h_input1[i + countX * j] = inputImage0Data[(i % inputWidth0) + inputWidth0 * (j % inputHeight0)];
        h_input2[i + countX * j] = inputImage1Data[(i % inputWidth1) + inputWidth1 * (j % inputHeight1)];

    }
}

//Implement the disparity map on the CPU.
std::cout<<"-------CPU Execution----------"<<std::endl;
Core::TimeSpan startTime = Core::getCurrentTime();
//disparityMapCalc(h_input1,h_input2,h_outputCpu,inputWidth0,inputHeight0,countX, countY);
Core::TimeSpan endTime = Core::getCurrentTime();
Core::TimeSpan cpuTime = endTime - startTime;
Core::writeImagePGM("Disparity_map_cpu_scene_result.pgm", h_outputCpu, countX, countY);
std::cout<<"-------CPU Execution Done----------"<<std::endl;

//Enqueue the images to the kernel.
cl::Event imageevent;
queue.enqueueWriteImage(image1, true, origin, region, countX*(sizeof(float)), 0, h_input1.data(), NULL, &imageevent);
queue.enqueueWriteImage(image2, true, origin, region, countX*(sizeof(float)), 0, h_input2.data(), NULL, &imageevent);

// Create kernel object.
cl::Kernel matchingCostFunction(program, "matchingCostFunction");

// Set Kernel Arguments.
matchingCostFunction.setArg<cl::Image2D>(0, image1);
matchingCostFunction.setArg<cl::Image2D>(1, image2);
matchingCostFunction.setArg<cl::Buffer>(2, output_disparity);
matchingCostFunction.setArg<cl_uint>(3, countX);
matchingCostFunction.setArg<cl_uint>(4, countY);


std::cout<<"-------Launching the matchingCostFunction kernel---------"<<std::endl;

//Launch the Kernel in X and Y direction.
cl::Event kernelLaunchEvent;
queue.enqueueNDRangeKernel(matchingCostFunction, 0, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &kernelLaunchEvent);

std::cout<<"----------Kernel successfully executed-------------------"<<std::endl;

//Read the output from GPU back to the host buffer.
queue.enqueueReadBuffer(output_disparity, true, 0, count*sizeof(int), h_outputGpu.data(),NULL,NULL);

//Print the Performance Parameters.
Core::TimeSpan gpuTime = OpenCL::getElapsedTime(kernelLaunchEvent);
std::cout<<"--------Performance Parameters-------------------"<<std::endl;
std::cout<<"1. CPU execution time: "<<cpuTime<<std::endl;
std::cout<<"2. GPU execution time: "<<gpuTime<<std::endl;
double speedUp = (cpuTime.getSeconds())/(gpuTime.getSeconds());
std::cout<<"3. SpeedUp(GPU/CPU)  : " << speedUp <<std::endl;
std::cout<<"--------Performance Parameters End---------------"<<std::endl;
//Generate the output image.
Core::writeImagePGM("Disparity_map_scene_w11.pgm", h_outputGpu, countX, countY);

std::cout<<"Disparity image result generated"<<std::endl;

}
