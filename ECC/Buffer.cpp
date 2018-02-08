#pragma once

#include "Buffer.hpp"

Buffer::Buffer() : size(0), memObj(NULL), flags(0)
{
}

Buffer::Buffer(size_t _size) : size(_size), memObj(NULL), flags(0)
{
}

Buffer::Buffer(size_t _size, cl_mem_flags _flags)
	: size(_size), memObj(NULL), flags(0)
{
	buildMemObj(_flags);
}

void Buffer::buildMemObj(cl_mem_flags flags)
{
	flags = flags;
	memObj = PlatformUtil::createBuffer(flags, size);
}

void Buffer::writeFrom(void* buffer, size_t buffSize)
{
	PlatformUtil::checkError(clEnqueueWriteBuffer(PlatformUtil::getCommandQueue(), memObj, CL_TRUE, 0, buffSize, buffer, 0, NULL, &ev));
}

void Buffer::readTo(void* buffer, size_t buffSize)
{
	PlatformUtil::checkError(clEnqueueReadBuffer(PlatformUtil::getCommandQueue(), memObj, CL_TRUE, 0, buffSize, buffer, 0, NULL, &ev));
}

cl_mem_flags Buffer::getFlags() const
{
	return flags;
}

size_t Buffer::getSize() const
{
	return size;
}

cl_mem Buffer::getMem() const
{
	return memObj;
}

void Buffer::zero()
{
	int pattern = 0;
	PlatformUtil::checkError(clEnqueueFillBuffer(PlatformUtil::getCommandQueue(), memObj, &pattern, sizeof(pattern), 0, size, 0, NULL, &ev));
}

void Buffer::clear()
{
	if (memObj != NULL)
	{
		PlatformUtil::checkError(clReleaseMemObject(memObj));
	}
}

size_t Buffer::getTransferTime()
{
	clWaitForEvents(1, &ev);

	cl_ulong time_start, time_end;

	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	return time_end - time_start;
}

ConstantBuffer::ConstantBuffer(int value) : Buffer(sizeof(int), CL_MEM_READ_ONLY)
{
	write(value);
}

void ConstantBuffer::write(int value)
{
	writeFrom(&value, sizeof(int));
}
