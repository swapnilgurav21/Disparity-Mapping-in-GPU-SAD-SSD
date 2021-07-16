#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp>
#endif

// *******************************************************************************
//  # Function    : getAbsolute
//  # Description : Function to calculate the absolute value of a float number.
//  #
//  # Input       : float value
//  # Output      : NA
//  # Return      : float absolute value
//# *******************************************************************************/

float getAbsolute(float value)
{
	if (value < 0)
		return (-1 * value);

	return value;
}

// *******************************************************************************
//  # Function    : matchingCostFunction
//  # Description : Reads 2 input images, calculates the depth by SSD (Sum of Squared
//					Differences).
//  #
//  # Input       : __read_only image2d_t img1, __read_only image2d_t img2,
//					__global float* disparity,  __global float* cost, int width,
//					int height
//  # Output      : Buffer containing disparity values. __global float* disparity
//  # Return      : NA
//# *******************************************************************************/

__kernel void matchingCostFunction(__read_only image2d_t img1,
	__read_only image2d_t img2,
	__global float* disparity,
	int width, int height)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

	int w = get_image_width(img1);
	int h = get_image_height(img1);

	int window_size = 11;
	float SSD_min = 99999.0;
	float SSD = 0.0;
	int disp = 0;
	float final_disparity = 0;
	int disparity_range = 100;

	float pixel_x = 0.0;

	int x = get_global_id(0);
	int y = get_global_id(1);

	for (disp = 0; disp <= disparity_range; disp++)
	{
		SSD = 0.0;
		for (w = 0; w < window_size; w++)
		{
			for (h = 0; h < window_size; h++)
			{
				//SSD calculation.
				pixel_x = getAbsolute(((read_imagef(img1, sampler, (int2)(w + x, h + y)).x - read_imagef(img2, sampler, (int2)(w + x - disp, h + y))).x));

				pixel_x = pixel_x * pixel_x;

				SSD = SSD + (pixel_x);
			}
		}

		if (SSD_min > SSD)
		{
			SSD_min = SSD;
			final_disparity = (float)disp;
			final_disparity = (float)(final_disparity) / disparity_range;

		}

	}
	disparity[y * width + x] = final_disparity;
	printf("%f", final_disparity);
}
