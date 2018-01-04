/*! \file ocl_to_cuda.h
	\brief Converts a given OpenCL kernel to an equivalent CUDA kernel.

	Conversion from OpenCL to CUDA is done in several steps:
	- Replace OpenCL keywords with the CUDA equivalent
	- Replace OpenCL synchronisation with CUDA synchronisation
	- Replace the OpenCL get_xxx_ID functions with CUDA equivalents
*/
// Replace the OpenCL keywords with CUDA equivalent
#ifndef OCL_TO_CUDA_H
#define OCL_TO_CUDA_H


#define __kernel __placeholder__	//!< OpenCL keyword __kernel will be replaced with \__placeholder\__
#define __global					//!< Remove the OpenCL \__global qualifier since pointers to global memory must not be qualified
#define __placeholder__ __global__	//!< \_\_placeholder\_\_ will be replaced with the CUDA keyword \_\_global\_\_
#define __local __shared__			//!< OpenCL keyword \_\_local will be replaced with the CUDA keyword \_\_shared\_\_
#define restrict __restrict__		//!< OpenCL keyword restrict will be replaced with the CUDA keyword \_\_restrict\_\_
#define __private

#define barrier(x) __syncthreads() //!< Replace OpenCL synchronisation with CUDA synchronisation

//! Replace the OpenCL get_local_ID function with CUDA equivalent threadIdx.
/*!
	\param x Specifies the dimension of the thread
	\return Thread id within a block specified by dimension
	\warning Value of x has to be in range of 0 to 2 as the dimensions x,y and z
*/
__device__ int get_local_id(int x)
{

	switch (x)
	{
	case 0:	return threadIdx.x;
	case 1:	return threadIdx.y;
	case 2:	return threadIdx.z;
	default: return -1;
	}
}

//! Replace the OpenCL get_group_id function with CUDA equivalent blockIdx.
/*!
	\param x Specifies the dimension of the block
	\return Block id specified by dimension
	\warning Value of x has to be in range of 0 to 2 as the dimensions x,y and z
*/
__device__ int get_group_id(int x)
{

	switch (x)
	{
	case 0:	return blockIdx.x;
	case 1:	return blockIdx.y;
	case 2:	return blockIdx.z;
	default: return -1;
	}
}

//! Replace the OpenCL get_global_id function with the calculation of global id.
/*!
	\param x Specifies the dimension
	\return Global index of a thread
	\warning Value of x has to be in range of 0 to 2 as the dimensions x,y and z
*/
__device__ int get_global_id(int x)
{

	switch (x)
	{
	case 0:	return blockIdx.x*blockDim.x + threadIdx.x;
	case 1:	return blockIdx.y*blockDim.y + threadIdx.y;
	case 2:	return blockIdx.z*blockDim.z + threadIdx.z;
	default: return -1;
	}
}
//Add the float8 data-type which is not available natively under CUDA
//typedef struct { float s0; float s1; float s2; float s3;
//                  float s4; float s5; float s6; float s7; } float8;

#endif // !OCL_TO_CUDA_H
