#pragma once
#include <CL/cl.h>
#include <string>
#include <vector>
#include "PlatformUtil.h"
#include "Buffer.h"

class Kernel
{
private:
	cl_kernel _kernel;
	cl_event ev;

	std::vector<size_t> globalSize, localSize;
	size_t dim;
	double runTime;

public:
	Kernel(cl_program program, const std::string& kernelName) : dim(1), runTime(-1.0)
	{
		cl_int err;
		_kernel = clCreateKernel(program, kernelName.c_str(), &err);
		PlatformUtil::checkError(err);
	}

	~Kernel()
	{
		clReleaseKernel(_kernel);
	}

	void SetArg(size_t idx, int value)
	{
		clSetKernelArg(_kernel, idx, sizeof(value), &value);
	}

	void SetLocalArg(size_t idx, size_t size)
	{
		PlatformUtil::checkError(clSetKernelArg(_kernel, idx, size, NULL));
	}

	void SetArg(size_t idx, Buffer& buff, bool write = false)
	{
		if (write && buff.getFlags() != CL_MEM_WRITE_ONLY)
			buff.write();

		cl_mem mem = buff.getMem();
		PlatformUtil::checkError(clSetKernelArg(_kernel, idx, sizeof(cl_mem), &mem));
	}

	void setDim(size_t dimension)
	{
		if (dimension > 3 || !dim)
		{
			std::cout << "Number of dimensions has to be at least 1 and less than or equal to 3" << std::endl;
			return;
		}

		dim = dimension;
	}

	template<typename ...DimSizes>
	void setGlobalSize(DimSizes... gs)
	{
		int arr[] = { gs... };
		setGlobalSize(std::vector<size_t>(arr, arr + sizeof...(gs)));
	}

	template<typename ...DimSizes>
	void setLocalSize(DimSizes... ls)
	{
		int arr[] = { ls... };
		setLocalSize(std::vector<size_t>(arr, arr + sizeof...(ls)));
	}

	void setGlobalSize(std::vector<size_t> gs)
	{
		if (gs.size() != dim)
		{
			std::cout << "Global size has wrong number of dimensions" << std::endl;
			return;
		}

		globalSize = gs;
 	}

	void setLocalSize(std::vector<size_t> ls)
	{
		if (ls.size() != dim)
		{
			std::cout << "Local size has wrong number of dimensions" << std::endl;
			return;
		}

		localSize = ls;
	}

	double getRuntime()
	{
		clWaitForEvents(1, &ev);

		cl_ulong time_start, time_end;

		clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		return (time_end - time_start) / 1e6;
	}

	void execute()
	{	
		for(int d = 0; d < dim; ++d)
		{
			if (localSize[d] <= 0 || globalSize[d] % localSize[d] != 0)
			{
				std::cout << "localSize doesnt divide globalSize, aborting" << std::endl;
				return;
			}
		}

		PlatformUtil::checkError(clEnqueueNDRangeKernel(PlatformUtil::getCommandQueue(), _kernel, dim, NULL, globalSize.data(), localSize.data(), 0, NULL, &ev));
	}
};

