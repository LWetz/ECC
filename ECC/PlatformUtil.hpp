#pragma once

#include<CL/cl.h>
#include<iostream>
#include<string>
#include<fstream>
#include"Util.hpp"

class PlatformUtil
{
	static cl_platform_id platform;
	static cl_device_id device;
	static cl_command_queue queue;
	static cl_context context;

	PlatformUtil()
	{
	}

public:
	static bool init(const std::string& vendorName, const std::string& deviceName);
	static void deinit();
	static void checkError(cl_int err);
	static bool buildProgramFromSource(const std::string& source, cl_program& program, std::string options = "");
	static bool buildProgramFromFile(const std::string& fileName, cl_program& program, std::string options = "");

	static cl_mem createBuffer(cl_mem_flags flags, size_t size);
	static cl_command_queue getCommandQueue();
	static cl_device_id getDevice();
	static void finish();
	~PlatformUtil();
};

