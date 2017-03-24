#pragma once
#include <CL/cl.h>
#include <string>
#include <vector>
#include "PlatformUtil.h"

class Kernel
{
private:
	cl_kernel _kernel;

	std::vector<size_t> globalSize, localSize;
	size_t dim;
	double runTime;

	cl_mem setArg(size_t idx, void* hostptr, size_t size, cl_mem_flags flags)
	{
		cl_mem mem = PlatformUtil::createBuffer(flags, size);

		if (hostptr)
		{
			if (flags != CL_MEM_WRITE_ONLY)
				PlatformUtil::checkError(clEnqueueWriteBuffer(PlatformUtil::getCommandQueue(), mem, CL_TRUE, 0, size, hostptr, 0, NULL, NULL));

			PlatformUtil::checkError(clSetKernelArg(_kernel, idx, sizeof(cl_mem), &mem));
		}
		else
		{
			PlatformUtil::checkError(clSetKernelArg(_kernel, idx, size, NULL));
		}
	}

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

	void SetInputArg(size_t idx, int value)
	{
		clSetKernelArg(_kernel, idx, sizeof(value), &value);
	}

	cl_mem SetInputArg(size_t idx, void* data, size_t size, bool needMem = false)
	{
		cl_mem mem = setArg(idx, data, size, CL_MEM_READ_ONLY);
		if (!needMem)
		{
			clReleaseMemObject(mem);
			mem = NULL;
		}
		return mem;
	}

	cl_mem SetOutputArg(size_t idx, void* data, size_t size, bool needMem = false)
	{
		cl_mem mem = setArg(idx, data, size, CL_MEM_WRITE_ONLY);
		if (!needMem)
		{
			clReleaseMemObject(mem);
			mem = NULL;
		}
		return mem;
	}

	cl_mem SetInputOutputArg(size_t idx, void* data, size_t size, bool needMem = false)
	{
		cl_mem mem = setArg(idx, data, size, CL_MEM_READ_WRITE);
		if (!needMem)
		{
			clReleaseMemObject(mem);
			mem = NULL;
		}
		return mem;
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
		size_t arr[] = { gs... };
		setGlobalSize(std::vector<size_t>(arr, arr + sizeof...(gs)));
	}

	template<typename ...DimSizes>
	void setLocalSize(DimSizes... ls)
	{
		size_t arr[] = { ls... };
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
		return runTime;
	}

	void readResult(cl_mem mem, size_t size, void* hostptr)
	{
		PlatformUtil::checkError(clEnqueueReadBuffer(PlatformUtil::getCommandQueue(), mem, CL_TRUE, 0, size, hostptr, 0, NULL, NULL));
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

		cl_event ev;
		PlatformUtil::checkError(clEnqueueNDRangeKernel(PlatformUtil::getCommandQueue(), _kernel, dim, NULL, globalSize.data(), localSize.data(), 0, NULL, &ev));
		clWaitForEvents(1, &ev);
		clFinish(PlatformUtil::getCommandQueue());

		cl_ulong time_start, time_end;

		double total_time;
		clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		total_time = (time_end - time_start) / 1e6;

		runTime = total_time;

		clFinish(PlatformUtil::getCommandQueue());
	}
};

