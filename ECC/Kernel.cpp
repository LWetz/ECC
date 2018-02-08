#include "Kernel.hpp"

Kernel::Kernel(cl_program program, const std::string& kernelName) : dim(1), runTime(-1.0), ev(NULL)
{
	cl_int err;
	_kernel = clCreateKernel(program, kernelName.c_str(), &err);
	PlatformUtil::checkError(err);
}

Kernel::~Kernel()
{
	clReleaseEvent(ev);
	clReleaseKernel(_kernel);
}

void Kernel::SetArg(size_t idx, int value)
{
	PlatformUtil::checkError(clSetKernelArg(_kernel, idx, sizeof(value), &value));
}

void Kernel::Kernel::SetLocalArg(size_t idx, size_t size)
{
	PlatformUtil::checkError(clSetKernelArg(_kernel, idx, size, NULL));
}

void Kernel::SetArg(size_t idx, Buffer& buff)
{
	cl_mem mem = buff.getMem();
	PlatformUtil::checkError(clSetKernelArg(_kernel, idx, sizeof(cl_mem), &mem));
}

void Kernel::setDim(size_t dimension)
{
	if (dimension > 3 || !dim)
	{
		std::cout << "Number of dimensions has to be at least 1 and less than or equal to 3" << std::endl;
		return;
	}

	dim = dimension;
}

void Kernel::setGlobalSize(std::vector<size_t> gs)
{
	if (gs.size() != dim)
	{
		std::cout << "Global size has wrong number of dimensions" << std::endl;
		return;
	}

	globalSize = gs;
}

void Kernel::setLocalSize(std::vector<size_t> ls)
{
	if (ls.size() != dim)
	{
		std::cout << "Local size has wrong number of dimensions" << std::endl;
		return;
	}

	localSize = ls;
}

size_t Kernel::getRuntime()
{
	clWaitForEvents(1, &ev);

	cl_ulong time_start, time_end;

	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	return time_end - time_start;
}

void Kernel::execute()
{
	for (int d = 0; d < dim; ++d)
	{
		if (localSize[d] <= 0 || globalSize[d] % localSize[d] != 0)
		{
			std::cout << "localSize doesnt divide globalSize, aborting" << std::endl;
			return;
		}
	}

	if (!ev) clReleaseEvent(ev);

	PlatformUtil::checkError(clEnqueueNDRangeKernel(PlatformUtil::getCommandQueue(), _kernel, dim, NULL, globalSize.data(), localSize.data(), 0, NULL, &ev));
}
