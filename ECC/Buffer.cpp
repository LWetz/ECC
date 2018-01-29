#pragma once

#include "Buffer.hpp"

Buffer::Buffer() : data(NULL), size(0), memObj(NULL), flags(0)
{
}

Buffer::Buffer(size_t _size) : data(new uint8_t[_size]), size(_size), memObj(NULL), flags(0)
{
}

Buffer::Buffer(size_t _size, cl_mem_flags _flags)
	: data(new uint8_t[_size]), size(_size), memObj(NULL), flags(0)
{
	buildMemObj(_flags);
}

void Buffer::buildMemObj(cl_mem_flags flags)
{
	flags = flags;
	memObj = PlatformUtil::createBuffer(flags, size);
}

void Buffer::write()
{
	PlatformUtil::checkError(clEnqueueWriteBuffer(PlatformUtil::getCommandQueue(), memObj, CL_TRUE, 0, size, data, 0, NULL, &ev));
}

void Buffer::writeFrom(void* buffer, size_t buffSize)
{
	PlatformUtil::checkError(clEnqueueWriteBuffer(PlatformUtil::getCommandQueue(), memObj, CL_TRUE, 0, buffSize, buffer, 0, NULL, &ev));
}

void Buffer::read()
{
	PlatformUtil::checkError(clEnqueueReadBuffer(PlatformUtil::getCommandQueue(), memObj, CL_TRUE, 0, size, data, 0, NULL, &ev));
}

void Buffer::readTo(void* buffer, size_t buffSize)
{
	PlatformUtil::checkError(clEnqueueReadBuffer(PlatformUtil::getCommandQueue(), memObj, CL_TRUE, 0, buffSize, buffer, 0, NULL, &ev));
}

cl_mem_flags Buffer::getFlags() const
{
	return flags;
}

void* Buffer::getData() const
{
	return (void*)data;
}

size_t Buffer::getSize() const
{
	return size;
}

cl_mem Buffer::getMem() const
{
	return memObj;
}

void Buffer::clear()
{
	delete[] data;
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

ConstantBuffer::ConstantBuffer(int constant) : Buffer(sizeof(constant), CL_MEM_READ_ONLY)
{
	memcpy(getData(), &constant, sizeof(constant));
}

