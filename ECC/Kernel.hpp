#pragma once
#include <CL/cl.h>
#include <string>
#include <vector>
#include "PlatformUtil.hpp"
#include "Buffer.hpp"

class Kernel
{
private:
	cl_kernel _kernel;
	cl_event ev;

	std::vector<size_t> globalSize, localSize;
	size_t dim;
	size_t runTime;

public:
	Kernel(cl_program program, const std::string& kernelName);

	~Kernel();
	void SetArg(size_t idx, int value);
	void SetLocalArg(size_t idx, size_t size);
	void SetArg(size_t idx, Buffer& buff);
	void setDim(size_t dimension);

	template<typename ...DimSizes>
	void setGlobalSize(DimSizes... gs)
	{
		int arr[] = { static_cast<size_t>(gs...) };
		setGlobalSize(std::vector<size_t>(arr, arr + sizeof...(gs)));
	}

	template<typename ...DimSizes>
	void setLocalSize(DimSizes... ls)
	{
		int arr[] = { static_cast<size_t>(ls...) };
		setLocalSize(std::vector<size_t>(arr, arr + sizeof...(ls)));
	}

	void setGlobalSize(std::vector<size_t> gs);
	void setLocalSize(std::vector<size_t> ls);

	size_t getRuntime();

	void execute();
};

