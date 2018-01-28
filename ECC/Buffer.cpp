#pragma once

#include "Buffer.hpp"

Buffer::Buffer() : data(NULL), size(0), memObj(NULL), flags(0)
{

}

Buffer::Buffer(size_t _size) : data(new uint8_t[_size]), size(_size), memObj(NULL), flags(0)
{
	memset(data, 0, size);
}

Buffer::Buffer(size_t _size, cl_mem_flags _flags)
	: data(new uint8_t[_size]), size(_size), memObj(NULL), flags(0)
{
	memset(data, 0, size);
	buildMemObj(_flags);
}

void Buffer::buildMemObj(cl_mem_flags flags)
{
	flags = flags;
	memObj = PlatformUtil::createBuffer(flags, size);
}

void Buffer::write()
{
	PlatformUtil::checkError(clEnqueueWriteBuffer(PlatformUtil::getCommandQueue(), memObj, CL_TRUE, 0, size, data, 0, NULL, NULL));
}

void Buffer::writeFrom(void* buffer, size_t buffSize)
{
	PlatformUtil::checkError(clEnqueueWriteBuffer(PlatformUtil::getCommandQueue(), memObj, CL_TRUE, 0, buffSize, buffer, 0, NULL, NULL));
}

void Buffer::read()
{
	PlatformUtil::checkError(clEnqueueReadBuffer(PlatformUtil::getCommandQueue(), memObj, CL_TRUE, 0, size, data, 0, NULL, NULL));
}

void Buffer::readTo(void* buffer, size_t buffSize)
{
	PlatformUtil::checkError(clEnqueueReadBuffer(PlatformUtil::getCommandQueue(), memObj, CL_TRUE, 0, buffSize, buffer, 0, NULL, NULL));
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


ConstantBuffer::ConstantBuffer(int constant) : Buffer(sizeof(constant), CL_MEM_READ_ONLY)
{
	memcpy(getData(), &constant, sizeof(constant));
}

