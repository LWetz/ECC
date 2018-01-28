#pragma once

#include "PlatformUtil.hpp"
#include <cstring>

class Buffer
{
private:
	uint8_t* data;
	size_t size;
	cl_mem memObj;
	cl_mem_flags flags;
	cl_event ev;

public:
	Buffer();
	Buffer(size_t _size);
	Buffer(size_t _size, cl_mem_flags _flags);

	void buildMemObj(cl_mem_flags flags);

	void write();
	void writeFrom(void* buffer, size_t buffSize);
	void read();
	void readTo(void* buffer, size_t buffSize);

	cl_mem_flags getFlags() const;
	void* getData() const;
	size_t getSize() const;
	cl_mem getMem() const;

	void clear();
};

class ConstantBuffer : public Buffer
{
public:
	ConstantBuffer(int constant);
};
