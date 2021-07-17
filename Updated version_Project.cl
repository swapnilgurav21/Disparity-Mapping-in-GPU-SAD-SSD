#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp>
#endif

int getIndexGlobal(size_t countX, int i, int j) 
{
	return j * countX + i;
}

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

float getValueImage(__read_only image2d_t a, int i, int j)
{
	//if (i < 0 || i >= countX || j < 0 || j >= countY)
	//return 0;
	return read_imagef(a, sampler, (int2) { i, j }).x;
}

float abs_diff(float a, float b)
{
	float diff;
	diff = a - b;
	if (diff < 0)
		return (-1 * diff);
	else
		return diff;
}
float square_diff(float a, float b)
{
	float diff;
	diff = a - b;
	return diff * diff;
}
__kernel void disparityMapping1(__read_only image2d_t d_inputL, __read_only image2d_t d_inputR, __global float* d_output)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	int imageWidth = get_image_width(d_inputL);
	int imageHeight = get_image_height(d_inputL);

	int windowSize = 11;
	int windowRange = windowSize/2 ;
	float SAD = 0.0;
	float SADmin = 10000.0;
	float disparity = 0.0;
	int disparityMax = 100;


	for (int d = 0; d <= disparityMax; d++)
	{
		SAD = 0;
		for (imageWidth = 1- windowRange; imageWidth < windowRange; imageWidth++)
		{
			for (imageHeight = 1 - windowRange; imageHeight < windowRange; imageHeight++)
			{

				SAD = SAD + abs_diff(getValueImage(d_inputL, imageWidth + i, imageHeight + j), getValueImage(d_inputR, imageWidth + i - d, imageHeight + j));

			}
		}

		if (SAD < SADmin)
		{
			SADmin = SAD;
			disparity = (float)d;
			disparity = (float)(disparity) / disparityMax;

		}

	}
	d_output[getIndexGlobal(countX, i, j)] = disparity;
}
__kernel void disparityMapping2(__read_only image2d_t d_inputL, __read_only image2d_t d_inputR, __global float* d_output)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	int imageWidth = get_image_width(d_inputL);
	int imageHeight = get_image_height(d_inputL);

	int windowSize = 11;
	int windowRange = windowSize / 2;
	float SSD = 0.0;
	float SSDmin = 10000.0;
	float disparity = 0.0;
	int disparityMax = 100;
	float temp;

	for (int d = 0; d <= disparityMax; d++)
	{
		SSD = 0;
		for (imageWidth = 1 - windowRange; imageWidth < windowRange; imageWidth++)
		{
			for (imageHeight = 1 - windowRange; imageHeight < windowRange; imageHeight++)
			{
				SSD = SSD + square_diff(getValueImage(d_inputL, imageWidth + i, imageHeight + j), getValueImage(d_inputR, imageWidth + i - d, imageHeight + j));

			}
		}

		if (SSD < SSDmin)
		{
			SSDmin = SSD;
			disparity = (float)d;
			disparity = (float)(disparity) / disparityMax;

		}

	}
	d_output[getIndexGlobal(countX, i, j)] = disparity;
}
