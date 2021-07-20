
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
	if (i < 0 || (size_t)i >= countX || j < 0 || (size_t)j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}

void disparityMappingSAD(const std::vector<float>& img1, const std::vector<float>& img2, std::vector<float>& h_outputSAD, size_t countX, size_t countY)
{
	int windowSize = 11;
	int windowRange = windowSize / 2;
	float SAD = 0.0;
	float SADmin = 10000.0;
	float disparity = 0.0;
	int disparityMax = 100;


	for (int i = 0; i < (int)countX; i++)
	{
		for (int j = 0; j < (int)countY; j++)
		{
			SADmin = 10000.0;
			for (int d = 0; d <= disparityMax; d++)
			{
				SAD = 0;
				for (int width = 1- windowRange; width < windowRange; width++)
				{
					for (int height = 1- windowRange; height < windowRange; height++)
					{
						SAD = SAD + abs(getValueGlobal(img1, countX, countY, width + i, height + j) - getValueGlobal(img2, countX, countY, width + i - d, height + j));
					}
				}
				if (SADmin > SAD)
				{
					SADmin = SAD;
					disparity = (float)d;
					disparity = (float)(disparity) / disparityMax;
				}
			}
			h_outputSAD[getIndexGlobal(countX, i, j)] = disparity;
		}

	}
}
void disparityMappingSSD(const std::vector<float>& img1, const std::vector<float>& img2, std::vector<float>& h_outputSSD, size_t countX, size_t countY)
{
	int windowSize = 11;
	int windowRange = windowSize / 2;
	float SSD = 0.0;
	float SSDmin = 10000.0;
	float disparity = 0.0;
	int disparityMax = 100;
	float temp = 0;

	for (int i = 0; i < (int)countX; i++)
	{
		for (int j = 0; j < (int)countY; j++)
		{
			SSDmin = 10000.0;
			for (int d = 0; d <= disparityMax; d++)
			{
				SSD = 0;
				for (int width = 1 - windowRange; width < windowRange; width++)
				{
					for (int height = 1 - windowRange; height < windowRange; height++)
					{
						temp = abs(getValueGlobal(img1, countX, countY, width + i, height + j) - getValueGlobal(img2, countX, countY, width + i - d, height + j));
						SSD = SSD + (temp * temp);
					}
				}
				if (SSDmin > SSD)
				{
					SSDmin = SSD;
					disparity = (float)d;
					disparity = (float)(disparity) / disparityMax;
				}
			}
			h_outputSSD[getIndexGlobal(countX, i, j)] = disparity;
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
	//platformId = 1;

	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[platformId](), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);

	// Declare necessary constants
	std::size_t wgSizeX = 16; // Number of work items per work group in X direction
	std::size_t wgSizeY = 16;
	std::size_t countX = wgSizeX * 24; // Overall number of work items in X direction = Number of elements in X direction
	std::size_t countY = wgSizeY * 18;

	std::size_t count = countX * countY; // Overall number of elements
	std::size_t size = count * sizeof(float); // Size of data in bytes

	//Allocate space for input and output data from CPU and GPU.
	std::vector<float> h_inputL(count);
	std::vector<float> h_inputR(count);
	std::vector<float> h_outputGpu(count);
	std::vector<float> h_outputCpuSAD(count);
	std::vector<float> h_outputCpuSSD(count);

	// Get a device of the context
	int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
	std::cout << "Using device " << deviceNr << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
	ASSERT(deviceNr > 0);
	ASSERT((size_t)deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "D:/PG_MS/Sem3/GPU_Lab/OpenCLExercise1_Basics/Opencl-ex1/Opencl-ex1/Opencl-ex1/src/Project.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Creation of Image - inputs
	cl::Image2D imageL(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);
	cl::Image2D imageR(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);

	// Buffer creations
	cl::Buffer d_output(context, CL_MEM_READ_WRITE, (countX * countY) * sizeof(int));

	cl::size_t<3> origin;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;
	cl::size_t<3> region;
	region[0] = countX;
	region[1] = countY;
	region[2] = 1;

	memset(h_inputL.data(), 255, size);
	memset(h_inputR.data(), 255, size);
	memset(h_outputGpu.data(), 255, size);
	memset(h_outputCpuSAD.data(), 255, size);
	memset(h_outputCpuSSD.data(), 255, size);

	//////////////////////////////////////////////////////////////////////////////////
	//		Read input images and set the data in respective buffers.
	//////////////////////////////////////////////////////////////////////////////////

	std::vector<float> inputDataL;
	std::vector<float>inputDataR;
	std::size_t inputWidthL, inputHeightL, inputWidthR, inputHeightR;
	Core::readImagePGM("D:/PG_MS/Sem3/GPU_Lab/OpenCLExercise1_Basics/Opencl-ex1/Opencl-ex1/Opencl-ex1/Teddy_L.pgm", inputDataL, inputWidthL, inputHeightL);
	Core::readImagePGM("D:/PG_MS/Sem3/GPU_Lab/OpenCLExercise1_Basics/Opencl-ex1/Opencl-ex1/Opencl-ex1/Teddy_R.pgm", inputDataR, inputWidthR, inputHeightR);

	for (size_t j = 0; j < countY; j++)
	{
		for (size_t i = 0; i < countX; i++)
		{
			h_inputL[i + countX * j] = inputDataL[(i % inputWidthL) + inputWidthL * (j % inputHeightL)];
			h_inputR[i + countX * j] = inputDataR[(i % inputWidthR) + inputWidthR * (j % inputHeightR)];
		}
	}

	//Implement the disparity map on the CPU.
	std::cout << "-------CPU Execution----------" << std::endl;
	Core::TimeSpan startTime = Core::getCurrentTime();
	disparityMappingSAD(h_inputL,h_inputR,h_outputCpuSAD,countX, countY);
	disparityMappingSSD(h_inputL, h_inputR, h_outputCpuSSD, countX, countY);
	Core::TimeSpan endTime = Core::getCurrentTime();
	Core::TimeSpan cpuTime = endTime - startTime;
	Core::writeImagePGM("output_disparity_cpuSAD.pgm", h_outputCpuSAD, countX, countY);
	Core::writeImagePGM("output_disparity_cpuSSD.pgm", h_outputCpuSSD, countX, countY);
	std::cout << "-------CPU Execution Done----------" << std::endl;

	for (int impl = 1; impl <= 2; impl++)
	{
		std::cout << "Implementation #" << impl << ":" << std::endl;

		// Reinitialize output memory to 0xff
		memset(h_outputGpu.data(), 255, size);
		//TODO: GPU
		queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data());


		//Enqueue the images to the kernel.
		cl::Event copy1;
		queue.enqueueWriteImage(imageL, true, origin, region, countX * (sizeof(float)), 0, h_inputL.data(), NULL, &copy1);
		queue.enqueueWriteImage(imageR, true, origin, region, countX * (sizeof(float)), 0, h_inputR.data(), NULL, &copy1);

		// Create kernel object.
		std::string kernelName = "disparityMapping" + boost::lexical_cast<std::string> (impl);
		cl::Kernel disparityMapping(program, kernelName.c_str());

		// Set Kernel Arguments.
		cl::Event execution;
		disparityMapping.setArg<cl::Image2D>(0, imageL);
		disparityMapping.setArg<cl::Image2D>(1, imageR);
		disparityMapping.setArg<cl::Buffer>(2, d_output);

		queue.enqueueNDRangeKernel(disparityMapping, 0, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &execution);

		std::cout << "----------Kernel successfully executed-------------------" << std::endl;

		//Read the output from GPU back to the host buffer.
		cl::Event copy2;
		queue.enqueueReadBuffer(d_output, true, 0, count * sizeof(int), h_outputGpu.data(), NULL, &copy2);


		Core::TimeSpan gpuTime = OpenCL::getElapsedTime(execution);
		Core::TimeSpan copyTime = OpenCL::getElapsedTime(copy1) + OpenCL::getElapsedTime(copy2);
		Core::TimeSpan overallGpuTime = gpuTime + copyTime;
		std::cout << "CPU Time: " << cpuTime.toString() << ", " << (count / cpuTime.getSeconds() / 1e6) << " MPixel/s" << std::endl;;
		std::cout << "Memory copy Time: " << copyTime.toString() << std::endl;
		std::cout << "GPU Time w/o memory copy: " << gpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / gpuTime.getSeconds()) << ", " << (count / gpuTime.getSeconds() / 1e6) << " MPixel/s)" << std::endl;
		std::cout << "GPU Time with memory copy: " << overallGpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / overallGpuTime.getSeconds()) << ", " << (count / overallGpuTime.getSeconds() / 1e6) << " MPixel/s)" << std::endl;

		//Generate the output image.
		Core::writeImagePGM("output_disparity_gpu_" + boost::lexical_cast<std::string> (impl) + ".pgm", h_outputGpu, countX, countY);

		std::cout << "Success" << std::endl;
	}

}
