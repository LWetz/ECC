#pragma once

#include<CL/cl.h>
#include<iostream>
#include<string>
#include<fstream>
#include"Util.h"

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
	static bool init(const std::string& vendorName, const std::string& deviceName)
	{
		cl_platform_id pids[16];
		cl_uint numPlatforms;
		size_t infoLen;
		std::string buff;
		buff.resize(256);

		if (clGetPlatformIDs(sizeof(pids) / sizeof(pids[0]), pids, &numPlatforms) != CL_SUCCESS)
			return false;
		
		if (!numPlatforms)
		{
			std::cout << "No OpenCL platform found, aborting." << std::endl;
			return false;
		}

		for (int n = 0; n < numPlatforms; n++)
		{
			clGetPlatformInfo(pids[n], CL_PLATFORM_VENDOR, buff.size(), &buff[0], NULL);
			if (buff.find(vendorName) != std::string::npos)
			{
				platform = pids[n];
				break;
			}
			
			if (n == numPlatforms - 1)
			{
				clGetPlatformInfo(pids[0], CL_PLATFORM_VENDOR, buff.size(), &buff[0], &infoLen);
				std::cout << "No OpenCL platform by vendor \"" << vendorName << "\", using " << buff.substr(0, infoLen) << " instead." << std::endl;
				platform = pids[0];
			}
		}
	
		cl_device_id dids[16];
		cl_uint numDevices;
		clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, sizeof(dids) / sizeof(dids[0]), dids, &numDevices);

		if (!numDevices)
		{
			std::cout << "No OpenCL device found, aborting." << std::endl;
			return false;
		}

		for (int n = 0; n < numDevices; n++)
		{
			clGetDeviceInfo(dids[n], CL_DEVICE_NAME, buff.size(), &buff[0], NULL);
			if (buff.find(deviceName) != std::string::npos)
			{
				device = dids[n];
				break;
			}

			if (n == numDevices - 1)
			{
				clGetDeviceInfo(dids[0], CL_DEVICE_NAME, buff.size(), &buff[0], &infoLen);
				std::cout << "No OpenCL device named \"" << deviceName << "\", using " << buff.substr(0, infoLen) << " instead." << std::endl;
				platform = pids[0];
			}
		}

		cl_int ret;
		context = clCreateContext(NULL, 1, &device, NULL, NULL, &ret);
		if (ret != CL_SUCCESS)
		{
			std::cout << "Couldn't create context, aborting" << std::endl;
			return false;
		}

		queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &ret);
		if (ret != CL_SUCCESS)
		{
			std::cout << "Couldn't create command queue, aborting" << std::endl;
			return false;
		}

		return true;
	}

	static void deinit()
	{
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
	}

	static void checkError(cl_int err)
	{
		if (err != CL_SUCCESS)
		{
			std::cout << "OpenCL Error: " << err << std::endl;
		}
	}

	static bool buildProgramFromFile(const std::string& fileName, cl_program& program, std::string options = "")
	{
		std::string source = Util::loadFileToString(fileName);
		const char* str = source.c_str();
		cl_int ret;

		program = clCreateProgramWithSource(context, 1, &str, NULL, &ret);
		if (ret != CL_SUCCESS)
		{
			std::cout << "Couldn't create program" << std::endl;
			return false;
		}

		if (clBuildProgram(program, 1, &device, options.c_str(), NULL, NULL) != CL_SUCCESS)
		{
			std::cout << "Couldn't build program" << std::endl;

			std::string log;
			size_t logSize = 0;
			checkError(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, NULL, NULL, &logSize));
			log.resize(logSize);
			checkError(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, &log[0], NULL));
			std::cout << log << std::endl;

			return false;
		}

		return true;
	}

	static cl_mem createBuffer(cl_mem_flags flags, size_t size)
	{
		return clCreateBuffer(context, flags, size, NULL, NULL);
	}

	static cl_command_queue getCommandQueue()
	{
		return queue;
	}

	~PlatformUtil();
};

