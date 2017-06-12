#pragma once

#include "PlatformUtil.h"
#include <cstring>

class Buffer
{
private:
	uint8_t* data;
	size_t size;
	cl_mem memObj;
	cl_mem_flags flags;

public:

	Buffer(size_t _size) : data(new uint8_t[_size]), size(_size), memObj(NULL), flags(0)
	{
		memset(data, 0, size);
	}

	Buffer(size_t _size, cl_mem_flags _flags) 
		: data(new uint8_t[_size]), size(_size), memObj(NULL), flags(0)
	{
		memset(data, 0, size);
		buildMemObj(_flags);
	}

	void buildMemObj(cl_mem_flags flags)
	{
		flags = flags;
		memObj = PlatformUtil::createBuffer(flags, size);
	}

	void write()
	{
		PlatformUtil::checkError(clEnqueueWriteBuffer(PlatformUtil::getCommandQueue(), memObj, CL_TRUE, 0, size, data, 0, NULL, NULL));
	}

	void read()
	{
		PlatformUtil::checkError(clEnqueueReadBuffer(PlatformUtil::getCommandQueue(), memObj, CL_TRUE, 0, size, data, 0, NULL, NULL));
	}

	cl_mem_flags getFlags() const
	{
		return flags;
	}

	void* getData() const
	{
		return (void*)data;
	}

	size_t getSize() const
	{
		return size;
	}

	cl_mem getMem() const
	{
		return memObj;
	}

	void clear()
	{
		delete[] data;
		if (memObj != NULL)
			clReleaseMemObject(memObj);
	}
};

class ConstantBuffer : public Buffer
{
public:
	ConstantBuffer(int constant) : Buffer(sizeof(constant), CL_MEM_READ_ONLY)
	{
		memcpy(getData(), &constant, sizeof(constant));
	}
};