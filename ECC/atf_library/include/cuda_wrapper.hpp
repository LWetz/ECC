//
//  cuda_wrapper.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 14/11/2016.
//  Copyright © 2016 Ari Rasch. All rights reserved.
//

#ifndef cuda_wrapper_h
#define cuda_wrapper_h


#include <stdio.h>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <ctime>


// TODO: loeschen, auch Ordner "loeschen" im Projek-Pfad
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "tp_value.hpp"
#include "helper.hpp"

namespace atf
{

namespace cf
{

#define NVRTC_SAFE_CALL(x) \
do { \
    nvrtcResult result = x; \
     if (result != NVRTC_SUCCESS) { \
        std::cerr << "\nerror: " #x " failed with error " \
                  << nvrtcGetErrorString(result) << '\n'; \
        exit(1); \
    } \
} while(0)

#define CUDA_SAFE_CALL(x) \
do { \
    CUresult result = x; \
    if (result != CUDA_SUCCESS) {\
        const char *msg; \
        cuGetErrorName(result, &msg); \
        std::cerr << "\nerror: " #x " failed with error " \
                  << msg << '\n'; \
        exit(1); \
    } \
} while(0)

template< typename... Ts >
class cuda_cf_class
{
  // helper
  cl_uint arg_index = 0;
  
  public:
    cuda_cf_class( const int&                 device_number,

                   const std::string&         source,
                 
                   const std::tuple<Ts...>&   kernel_inputs,

                   unsigned int               grid_dim_x = 1,
                   unsigned int               grid_dim_y = 1,
                   unsigned int               grid_dim_z = 1,
                 
                   unsigned int               block_dim_x = 1,
                   unsigned int               block_dim_y = 1,
                   unsigned int               block_dim_z = 1
                 )
      : _device(), _context(), _program(), _source( source ), _kernel_inputs( kernel_inputs ), _grid_dim_x(grid_dim_x), _grid_dim_y(grid_dim_y), _grid_dim_z(grid_dim_z), _block_dim_x(block_dim_x), _block_dim_y(block_dim_y), _block_dim_z(block_dim_z), _kernel_buffers()
    {
      assert( grid_dim_x  + grid_dim_y  + grid_dim_z == 0  || grid_dim_x  + grid_dim_y  + grid_dim_z  >=3 );
      assert( block_dim_x + block_dim_y + block_dim_z == 0 || block_dim_x + block_dim_y + block_dim_z >=3 );
    
      // create program
      NVRTC_SAFE_CALL( nvrtcCreateProgram( &_program,
                                           _source.c_str(),
                                           NULL,
                                           0,
                                           NULL,
                                           NULL
                                         )
                     );

      // initialize CUDA
      CUDA_SAFE_CALL( cuInit( 0 )                            );
      CUDA_SAFE_CALL( cuDeviceGet( &_device, device_number ) );
      CUDA_SAFE_CALL( cuCtxCreate( &_context, 0, _device )   );
      
      // create kernel buffers
      this->create_buffers_and_set_args( std::make_index_sequence<sizeof...(Ts)>() );
    }
  
  
    size_t operator()( const configuration& configuration )
    {
      // determine grid dimensions if not set in ctor
      if( _grid_dim_x == 0 && _grid_dim_y == 0 && _grid_dim_z == 0 )
      {
        _grid_dim_x = 1;
        _grid_dim_y = 1;
        _grid_dim_z = 1;
        
        for( auto& tp : configuration )
        {
          auto tp_value = tp.second;
          if( tp_value.name().find( "GD" ) != std::string::npos )
          {
            if     ( tp_value.name().back() == 'x' ) _grid_dim_x = tp_value.value();
            else if( tp_value.name().back() == 'y' ) _grid_dim_y = tp_value.value();
            else if( tp_value.name().back() == 'z' ) _grid_dim_z = tp_value.value();
          }
        }
      }
      
      // determine block dimensions if not set in ctor
      if( _block_dim_x == 0 && _block_dim_y == 0 && _block_dim_z == 0 )
      {
        _block_dim_x = 1;
        _block_dim_y = 1;
        _block_dim_z = 1;
        
        for( auto& tp : configuration )
        {
          auto tp_value = tp.second;
          if( tp_value.name().find( "BD" ) != std::string::npos )
          {
            if     ( tp_value.name().back() == 'x' ) _block_dim_x = tp_value.value();
            else if( tp_value.name().back() == 'y' ) _block_dim_y = tp_value.value();
            else if( tp_value.name().back() == 'z' ) _block_dim_z = tp_value.value();
          }
        }
      }
    
      // create flags
      std::vector<std::string> flags;

      for( const auto& tp : configuration )
        flags.emplace_back( " -D " + tp.second.name() + "=" + static_cast<std::string>( tp.second.value() ) );
      
      for( size_t i = 0 ; i < _kernel_buffer_sizes.size() ; ++i )
        flags.emplace_back( std::string( " -D N_" ) + std::to_string( i ) + std::string( "=" ) + std::to_string( _kernel_buffer_sizes[ i ] ) );
      
      std::vector<char*> flags_as_C_array;
      auto string_to_char = []( const std::string& s ){ char* c_string = new char[ s.size() + 1 ];
                                                        std::strcpy( c_string, s.c_str() );
                                                        return c_string;
                                                      };
      std::transform( flags.begin(), flags.end(), std::back_inserter( flags_as_C_array ), string_to_char );
      
      
      // compile kernel
      nvrtcResult compile_result = nvrtcCompileProgram( _program, static_cast<int>( flags_as_C_array.size() ), flags_as_C_array.data() );
      
      // free char* in "flags_as_C_array"
      for( const auto& elem : flags_as_C_array )
        delete[] elem;
      
      // in case of an error: obtain compilation log from the program
      if( compile_result != NVRTC_SUCCESS )
      {
        size_t log_size;
        NVRTC_SAFE_CALL( nvrtcGetProgramLogSize( _program, &log_size ) );
        
        char* log = new char[ log_size ];
        NVRTC_SAFE_CALL( nvrtcGetProgramLog( _program, log ) );
        
        std::cout << log << '\n';
        
        delete[] log;
        
        throw std::exception();
      }

      // Obtain PTX from the program.
      size_t ptx_size;
      NVRTC_SAFE_CALL( nvrtcGetPTXSize( _program, &ptx_size ) );
      
      char* ptx_code = new char[ ptx_size ];
      NVRTC_SAFE_CALL( nvrtcGetPTX( _program, ptx_code ) );
      
      // Load the generated PTX and get a handle to the kernel.
      CUmodule   module;
      CUfunction kernel;

      CUDA_SAFE_CALL( cuModuleLoadDataEx( &module, ptx_code, 0, 0, 0 ) );
      CUDA_SAFE_CALL( cuModuleGetFunction( &kernel, module, "func" )   ); // TODO: code aus PJS WS16/16 übernehmen für automatische Erkennung "func"

      cudaEvent_t start, stop;
      cudaEventCreate( &start );
      cudaEventCreate( &stop );

      // start kernel
      cudaEventRecord( start, 0 );
      CUresult kernelResult = cuLaunchKernel( kernel,
                                              _grid_dim_x, _grid_dim_y, _grid_dim_z,
                                              _block_dim_x, _block_dim_y, _block_dim_z,
                                              0,
                                              NULL,
                                              _kernel_inputs_ptr.data(),
                                              NULL
                                            );
      cudaEventRecord(stop, 0 );

      if( kernelResult != CUDA_SUCCESS )
        throw std::exception();

      // profiling
      float kernel_runtime_in_ms = 0;
      
      cudaEventSynchronize( stop                               );
      cudaEventElapsedTime( &kernel_runtime_in_ms, start, stop );

      cudaEventDestroy( start );
      cudaEventDestroy( stop );

      return static_cast<size_t>( kernel_runtime_in_ms);
    }

  private:
    CUdevice                 _device;
    CUcontext                _context;
  
    nvrtcProgram             _program;
    const std::string&       _source;
  
    std::tuple<Ts...>        _kernel_inputs;
    std::vector<void*>       _kernel_inputs_ptr;
    std::vector<CUdeviceptr> _kernel_buffers;
    std::vector<size_t>      _kernel_buffer_sizes;
  
    unsigned int             _grid_dim_x,  _grid_dim_y,  _grid_dim_z;
    unsigned int             _block_dim_x, _block_dim_y, _block_dim_z;
  

    // helper for creating buffers
    template< size_t... Is >
    void create_buffers_and_set_args( std::index_sequence<Is...> )
    {
      create_buffers_and_set_args_impl( std::get<Is>( _kernel_inputs )... );
    }
  
    template< typename T, typename... ARGs >
    void create_buffers_and_set_args_impl( scalar<T>& scalar, ARGs&... args )
    {
      //set kernel arg
      _kernel_inputs_ptr.emplace_back( scalar.get_ptr() );
      
      create_buffers_and_set_args_impl( args... );
    }

    template< typename T, typename... ARGs >
    void create_buffers_and_set_args_impl( buffer_class<T>& buffer, ARGs&... args )
    {
      auto start_time = std::chrono::system_clock::now();
      
      // add buffer size to _kernel_buffer_sizes
      _kernel_buffer_sizes.emplace_back( buffer.size() );

      _kernel_buffers.emplace_back(); // create new CUdeviceptr
      auto& ptr = _kernel_buffers.back();
      
      CUDA_SAFE_CALL( cuMemAlloc( &ptr, buffer.size() * sizeof( T ) )                          );
      CUDA_SAFE_CALL( cuMemcpyHtoD( ptr, buffer.get() , buffer.size() * sizeof( T ) ) );
      
      auto end_time = std::chrono::system_clock::now();
      auto runtime  = std::chrono::duration_cast<std::chrono::milliseconds>( end_time - start_time ).count();
      std::cout << "Time to create and fill buffers: " << runtime << std::endl;
      
      //set kernel arg
      _kernel_inputs_ptr.emplace_back( &ptr );
      
      create_buffers_and_set_args_impl( args... );
    }
  
    void create_buffers_and_set_args_impl()
    {}
};


// factory
template< typename... Ts >
auto cuda( const int&                 device_number,
                  
           const std::string&         source,
          
           const std::tuple<Ts...>&   kernel_inputs,
                 
           unsigned int               grid_dim_x = 1,
           unsigned int               grid_dim_y = 1,
           unsigned int               grid_dim_z = 1,
         
           unsigned int               block_dim_x = 1,
           unsigned int               block_dim_y = 1,
           unsigned int               block_dim_z = 1
         )
{
  return cuda_cf_class<Ts...>( device_number, source, kernel_inputs, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z );
}

} // namespace cf

} // namespace atf

#endif /* defined( cuda_wrapper_h ) */
