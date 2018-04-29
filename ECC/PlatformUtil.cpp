#include "PlatformUtil.hpp"

cl_platform_id PlatformUtil::platform;
cl_device_id PlatformUtil::device;
cl_command_queue PlatformUtil::queue;
cl_context PlatformUtil::context;

bool PlatformUtil::init(const std::string& vendorName, const std::string& deviceName)
{
	cl_platform_id pids[16];
	cl_uint numPlatforms;
	size_t infoLen;
	std::string buff;
	buff.resize(256);

	cl_int err = clGetPlatformIDs(sizeof(pids) / sizeof(pids[0]), pids, &numPlatforms);
	if (err != CL_SUCCESS)
	{
		std::cout << "clGetPlatfromIDs: " << err << std::endl;
		return false;
	}

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
			device = dids[0];
		}
	}

	cl_int ret;
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &ret);
	if (ret != CL_SUCCESS)
	{
		std::cout << "Couldn't create context, aborting (" << ret << ")" << std::endl;
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

void  PlatformUtil::deinit()
{
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}

void PlatformUtil::checkError(cl_int err)
{
	if (err != CL_SUCCESS)
	{
		std::cout << "OpenCL Error: " << err << std::endl;
		throw std::exception();
	}
}

bool PlatformUtil::buildProgramFromSource(const std::string& source, cl_program& program, std::string options)
{
	const char* str = source.c_str();
	cl_int ret;

	program = clCreateProgramWithSource(context, 1, &str, NULL, &ret);
	if (ret != CL_SUCCESS)
	{
		std::cout << "Couldn't create program" << std::endl;
		return false;
	}

//	options += " -cl-nv-verbose";
	if ((ret = clBuildProgram(program, 1, &device, options.c_str(), NULL, NULL)) != CL_SUCCESS)
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

bool PlatformUtil::buildProgramFromFile(const std::string& fileName, cl_program& program, std::string options)
{
	std::string source = Util::loadFileToString(fileName);
	return buildProgramFromSource(source, program, options);
}

cl_mem PlatformUtil::createBuffer(cl_mem_flags flags, size_t size)
{
	cl_int err;
	auto buff = clCreateBuffer(context, flags, size, NULL, &err);
	PlatformUtil::checkError(err);
	return buff;
}

cl_command_queue PlatformUtil::getCommandQueue()
{
	return queue;
}

cl_device_id PlatformUtil::getDevice()
{
	return device;
}

void PlatformUtil::finish()
{
	checkError(clFinish(getCommandQueue()));
}
