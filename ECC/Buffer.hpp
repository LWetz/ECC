#pragma once

#include "PlatformUtil.hpp"
#include <cstring>

class Buffer
{
private:
	size_t size;
	cl_mem memObj;
	cl_mem_flags flags;
	cl_event ev;

public:
	Buffer();
	Buffer(size_t _size);
	Buffer(size_t _size, cl_mem_flags _flags);

	void buildMemObj(cl_mem_flags flags);

	void writeFrom(void* buffer, size_t buffSize);
	void readTo(void* buffer, size_t buffSize);

	cl_mem_flags getFlags() const;
	size_t getSize() const;
	cl_mem getMem() const;

	size_t getTransferTime();

	void zero();

	void clear();
};

class ConstantBuffer : public Buffer
{
public:
	ConstantBuffer(int value);
	void write(int value);
};
