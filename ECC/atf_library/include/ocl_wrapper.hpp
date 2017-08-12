//
//  ocl_wrapper.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 14/11/2016.
//  Copyright © 2016 Ari Rasch. All rights reserved.
//

#ifndef ocl_wrapper_h
#define ocl_wrapper_h


#include <array>
#include <stdio.h>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <type_traits>
#include <utility>

//#ifdef __APPLE__
//  #include <OpenCL/OpenCL.h>
//#else
//  #include <CL/cl.h>
//#endif

#define __CL_ENABLE_EXCEPTIONS
#include "../libraries/cl.hpp"
//#undef  __CL_ENABLE_EXCEPTIONS

#include "tp_value.hpp"
#include "helper.hpp"


#define NUM_EVALUATIONS  5
#define WARM_UPS         0

//enables lazy evaluation
namespace
{
  template< typename T >
  class wrapper
  {
      // save l-values as references and r-values by value
      using T_ref_free  = typename std::remove_reference<T>::type;
      using T_save_type = typename std::conditional_t< std::is_rvalue_reference<T>::value, T_ref_free, T_ref_free& >;
    
    public:
      wrapper( T value )
        : _value( value )
      {}
    
    
      T_ref_free get_value()
      {
        return _value;
      }
    
    
    private:
      T_save_type _value;
  };
}

namespace atf
{

namespace cf
{
  using nd_range_t              = std::array<size_t,3>;
  using thread_configurations_t = std::map< ::atf::configuration, std::array<nd_range_t,2> >;


void check_error(cl_int err);

#if 0

template< typename T_lhs_0, typename T_rhs_0, typename callable_0,
          typename T_lhs_1, typename T_rhs_1, typename callable_1,
          typename T_lhs_2, typename T_rhs_2, typename callable_2
        >
auto GS( op_wrapper_class< T_lhs_0, T_rhs_0, callable_0 > gs_0,
         op_wrapper_class< T_lhs_1, T_rhs_1, callable_1 > gs_1,
         op_wrapper_class< T_lhs_2, T_rhs_2, callable_2 > gs_2
       )
{
  //return std::array< int, 3 >( gs_0.cast(), gs_1.cast(), gs_2.cast() );
  return std::make_tuple( gs_0, gs_1, gs_2 );
}


template< typename T_lhs_0, typename T_rhs_0, typename callable_0,
          typename T_lhs_1, typename T_rhs_1, typename callable_1,
          typename T_lhs_2, typename T_rhs_2, typename callable_2
        >
auto LS( op_wrapper_class< T_lhs_0, T_rhs_0, callable_0 > ls_0,
         op_wrapper_class< T_lhs_1, T_rhs_1, callable_1 > ls_1,
         op_wrapper_class< T_lhs_2, T_rhs_2, callable_2 > ls_2
       )
{
  //return std::array< int, 3 >( ls_0.cast(), ls_1.cast(), ls_2.cast() );
  return std::make_tuple( ls_0, ls_1, ls_2 );
}

#endif

#if 0


template< typename GS_0, typename GS_1, typename GS_2 >
auto GS( GS_0 gs_0, GS_1 gs_1, GS_2 gs_2 )
{
  auto gs_0_val = typename std::conditional<std::is_fundamental<GS_0>::value, scalar_wrapper<GS_0>, GS_0>::type(gs_0);
  auto gs_1_val = typename std::conditional<std::is_fundamental<GS_1>::value, scalar_wrapper<GS_1>, GS_1>::type(gs_1);
  auto gs_2_val = typename std::conditional<std::is_fundamental<GS_2>::value, scalar_wrapper<GS_2>, GS_2>::type(gs_2);
  
  return std::make_tuple( gs_0_val, gs_1_val, gs_2_val );
}


template< typename LS_0, typename LS_1, typename LS_2 >
auto LS( LS_0 ls_0, LS_1 ls_1, LS_2 ls_2 )
{
  return std::make_tuple( ls_0, ls_1, ls_2 );
}

#endif 


#if 1

template< typename T_0, typename T_1 = size_t, typename T_2 = size_t >
auto GS( T_0&& gs_0, T_1&& gs_1 = 1, T_2&& gs_2 = 1 )
{
  return std::make_tuple( wrapper<T_0&&>( std::forward<T_0>(gs_0) ), wrapper<T_1&&>( std::forward<T_1>(gs_1) ), wrapper<T_2&&>( std::forward<T_2>(gs_2) ) );
}


template< typename T_0, typename T_1 = size_t, typename T_2 = size_t >
auto LS( T_0&& ls_0, T_1&& ls_1 = 1, T_2&& ls_2 = 1 )
{
  return std::make_tuple( wrapper<T_0&&>( std::forward<T_0>(ls_0) ), wrapper<T_1&&>( std::forward<T_1>(ls_1) ), wrapper<T_2&&>( std::forward<T_2>(ls_2) ) );
}


#endif

class device_info
{
  public:
    enum device_t { CPU, GPU, ACC };
  
    device_info( cl_device_id device_id )
      : _platform(), _device( cl::Device( device_id ) )
    {
      _platform = _device.getInfo<CL_DEVICE_PLATFORM>();
    }
  
    device_info( const std::string& vendor_name,
                 const device_t&    device_type,
                 const int&         device_number
               )
      : _platform(), _device()
    {
      cl_device_type ocl_device_type;
      switch( device_type )
      {
        case CPU:
          ocl_device_type = CL_DEVICE_TYPE_CPU;
          break;

        case GPU:
          ocl_device_type = CL_DEVICE_TYPE_GPU;
          break;

        case ACC:
          ocl_device_type = CL_DEVICE_TYPE_ACCELERATOR;
          break;

        default:
          assert( false && "unknown device type" );
          break;
      }
      
      // get platform
      std::vector<cl::Platform> platforms;

      bool found = false;
      auto error = cl::Platform::get( &platforms ); check_error( error );
      for( const auto& platform : platforms )
      {
        std::string platform_name;
        platform.getInfo( CL_PLATFORM_VENDOR, &platform_name );
	std::cout << platform_name << std::endl;
        if( platform_name.find( vendor_name ) != std::string::npos )
        {
          _platform = platform;
          std::cout << "Platform with name " << platform_name << " found." << std::endl;
          found = true;
          break;
        }
      }
      
      if( !found )
      {
        std::cout << "Platform not found." << std::endl;
        exit( 1 );
      }

      // get device
      std::vector<cl::Device> devices;
      _platform.getDevices( ocl_device_type, &devices );

      if( device_number < devices.size() )
      {
        _device = devices[ device_number ];
        std::string device_name;
        _device.getInfo( CL_DEVICE_NAME, &device_name );
        std::cout << "Device with name " << device_name << " found." << std::endl;
      }
      else
      {
        std::cout << "Device not found." << std::endl;
        exit( 1 );
      }
    }
  
    cl::Platform platform() const
    {
      return _platform;
    }
  
  
    cl::Device device() const
    {
      return _device;
    }
  
  
  private:
    cl::Platform _platform;
    cl::Device   _device;
};

class kernel_info
{
  public:
    kernel_info( std::string source, std::string name = "func", std::string flags = "" )
      : _source( source ), _name( name ), _flags( flags )
    {}
  
    std::string source() const
    {
      return _source;
    }
  
    std::string name() const
    {
      return _name;
    }
  
    std::string flags() const
    {
      return _flags;
    }
  
  private:
    std::string _source;
    std::string _name;
    std::string _flags;
};


template< typename GS_0, typename GS_1, typename GS_2,
          typename LS_0, typename LS_1, typename LS_2,
          typename... Ts
        >
class ocl_cf_class
{
  // helper
  cl_int  error;
  cl_uint arg_index  = 0;
  size_t  buffer_pos = 0;
  
  public:
    ocl_cf_class( const device_info&             device,
                  const kernel_info&             kernel,
                  const std::tuple<Ts...>&       kernel_inputs,
                  std::tuple< GS_0, GS_1, GS_2 > global_size,
                  std::tuple< LS_0, LS_1, LS_2 > local_size
                )
      : _platform( device.platform() ), _device( device.device() ), _context(), _command_queue(), _program(), _kernel_source( kernel.source() ), _kernel_name( kernel.name() ), _kernel_flags( kernel.flags() ), _kernel_inputs( kernel_inputs ), _kernel_buffers(), _kernel_input_sizes(), _global_size_pattern( global_size ), _local_size_pattern( local_size ), _thread_configuration( nullptr ), _check_result( false ), _num_wrong_results( 0 ), _gold_sizes(), _gold_ptrs()
    {

#if 0
      // check availability of global/local size
      if( global_size[0] == 0 && global_size[1] == 0 && global_size[2] == 0 )
        _global_size_pattern_set_in_ctor = false;
      else
        _global_size_pattern_set_in_ctor = true;

      if( local_size[0] == 0 && local_size[1] == 0 && local_size[2] == 0 )
        _local_size_pattern_set_in_ctor = false;
      else
        _local_size_pattern_set_in_ctor = true;
#endif

      
      // create context and command queue
      cl_context_properties props[] = { CL_CONTEXT_PLATFORM,
                                        reinterpret_cast<cl_context_properties>( _platform() ),
                                        0
                                      };

      _context       = cl::Context( VECTOR_CLASS<cl::Device>( 1, _device ), props );
      _command_queue = cl::CommandQueue( _context, _device, CL_QUEUE_PROFILING_ENABLE );

      // create program
      _program = cl::Program( _context,
                              cl::Program::Sources( 1, std::make_pair( _kernel_source.c_str(), _kernel_source.length() ) )
      );
      
      // create kernel input buffers
      this->create_buffers( std::make_index_sequence<sizeof...(Ts)>() );
    }
  
    ~ocl_cf_class()
    {
      if( _check_result && _num_wrong_results > 0 )
        std::cout << "\nnumber of wrong results: " << _num_wrong_results << std::endl;
    }


    auto& save_thread_configuration( thread_configurations_t& thread_configuration )
    {
      _thread_configuration = &thread_configuration;
      
      return *this;
    }
  
    template< typename... scalar_t >
    auto& check_result( std::vector<scalar_t>&... gold_vectors )
    {
      _check_result = true;
      
      _gold_sizes = std::vector<size_t>( { ( gold_vectors.size() * sizeof(scalar_t) )... } );
      _gold_ptrs  = std::vector<void*>( { static_cast<void*>( gold_vectors.data() )... } );
      
      return *this;
    }
  
  
    size_t operator()(configuration& configuration )
    {
#if 0
// LÖSCHEN !!!
std::cout << "WGD = " << configuration["WGD"];
std::cout << " , MDIMCD = " << configuration["MDIMCD"];
std::cout << " , NDIMCD = " << configuration["NDIMCD"];
std::cout << " , MDIMAD = " << configuration["MDIMAD"];
std::cout << " , NDIMBD = " << configuration["NDIMBD"];
std::cout << " , KWID = " << configuration["KWID"];
std::cout << " , VWMD = " << configuration["VWMD"];
std::cout << " , VWND = " << configuration["VWND"];
std::cout << " , PADA = " << configuration["PADA"];
std::cout << " , PADB = " << configuration["PADB"];
std::cout << " , PRECISION = " << configuration["PRECISION"];

throw std::exception();
#endif 

#if 0 // for OT evaluation
size_t kSizeM = 100;
size_t kSizeN = 10;

auto WGD    = static_cast<int>( configuration["WGD"]    );
auto MDIMCD = static_cast<int>( configuration["MDIMCD"] );
auto NDIMCD = static_cast<int>( configuration["NDIMCD"] );
auto MDIMAD = static_cast<int>( configuration["MDIMAD"] );
auto NDIMBD = static_cast<int>( configuration["NDIMBD"] );
auto KWID   = static_cast<int>( configuration["KWID"]   );
auto VWMD   = static_cast<int>( configuration["VWMD"]   );
auto VWND   = static_cast<int>( configuration["VWND"]   );


if(
    !( WGD % MDIMCD == 0 && kSizeM % MDIMCD == 0) ||
    !( WGD % NDIMCD == 0 && kSizeN % NDIMCD == 0) ||

    !( WGD % MDIMAD == 0 && (MDIMCD*NDIMCD) % MDIMAD == 0 && WGD % ( (MDIMCD*NDIMCD)/MDIMAD ) == 0 ) ||
    !( WGD % NDIMBD == 0 && (MDIMCD*NDIMCD) % NDIMBD == 0 && WGD % ( (MDIMCD*NDIMCD)/NDIMBD ) == 0 ) ||

    !( WGD % KWID == 0 ) ||

    !( (WGD/MDIMCD) % VWMD == 0 && (WGD/MDIMAD) % VWMD == 0 ) ||
    !( (WGD/NDIMCD) % VWND == 0 && (WGD/NDIMBD) % VWND == 0 )
 )
   throw std::exception();
#endif
  
  
#if 0
      // determine global size if not set in ctor
      if( !_global_size_pattern_set_in_ctor )
      {
        std::array< size_t, 3 > gs = { 1, 1, 1 };
        
        for( auto& tp : configuration )
        {
          if( tp.name().find( "GS_" ) == 0 && tp.name().size() == 4 )
          {
            size_t index = atoi( &tp.name().back() ); // tp_value.name().back() - '0'; // convert last char of "tp_value.name()" to integer type
            gs[ index ] = tp.value();
          }
        }
        
        _global_size_pattern = cl::NDRange( gs[0], gs[1], gs[2] );
      }

      // determine local size if not set in ctor
      if( !_local_size_pattern_set_in_ctor )
      {
        std::array< size_t, 3 > ls = { 1, 1, 1 };
        
        for( auto& tp : configuration )
        {
          if( tp.name().find( "LS_" ) == 0  && tp.name().size() == 4 )
          {
            size_t index = atoi( &tp.name().back() );
            ls[ index ] = tp.value();
          }
        }
        
        _local_size_pattern = cl::NDRange( ls[0], ls[1], ls[2] );
      }
#else
     // update tp values
     for( auto& tp : configuration )
     {
       auto value        = tp.second.value();
       auto tp_value_ptr = tp.second.tp_value_ptr();
 
       switch( value.type_id() )
       {
         case value_type::int_t:
           *static_cast<int*>( tp_value_ptr ) = static_cast<int>( value );
           break;
          
         case value_type::size_t_t:
           *static_cast<size_t*>( tp_value_ptr ) = static_cast<size_t>( value );
           break;

         case value_type::float_t:
           *static_cast<float*>( tp_value_ptr ) = static_cast<float>( value );
           break;

         case value_type::double_t:
           *static_cast<double*>( tp_value_ptr ) = static_cast<double>( value );
           break;
        
         default:
           throw std::exception();
       }
     }

     size_t gs_0 = std::get<0>( _global_size_pattern ).get_value();
     size_t gs_1 = std::get<1>( _global_size_pattern ).get_value();
     size_t gs_2 = std::get<2>( _global_size_pattern ).get_value();

     size_t ls_0 = std::get<0>( _local_size_pattern ).get_value();
     size_t ls_1 = std::get<1>( _local_size_pattern ).get_value();
     size_t ls_2 = std::get<2>( _local_size_pattern ).get_value();
#endif

      // create flags
      std::stringstream flags;

      for( const auto& tp : configuration )
        flags << " -D " << tp.second.name() << "=" << tp.second.value();

#if 0
      for( size_t i = 0 ; i < _kernel_input_sizes.size() ; ++i )
        flags << " -D " << "N_" << i << "=" << _kernel_input_sizes[ i ];
#endif
      // set additional kernel flags
      flags << _kernel_flags;


      // compile kernel
      try
      {
        auto start = std::chrono::system_clock::now();
        
        _program.build( std::vector<cl::Device>( 1, _device ), flags.str().c_str() );

        auto end = std::chrono::system_clock::now();
        auto runtime_in_sec = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();
//        std::cout << std::endl << "compilation runtime: " << runtime_in_sec << "msec\n" << std::endl;
      }
      catch( cl::Error& err )
      {
          std::cout << flags.str() << std::endl;
        if( err.err() == CL_BUILD_PROGRAM_FAILURE )
        {
          auto buildLog = _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>( _device );
          std::cout << std::endl << "Build failed! Log:" << std::endl << buildLog << std::endl;
        }

        throw std::exception();
      }

      auto kernel = cl::Kernel( _program, _kernel_name.c_str(), &error ); check_error( error ); // TODO: code aus PJS WS16/16 übernehmen für automatische Erkennung "func"

      // set kernel arguments
      arg_index  = 0;
      buffer_pos = 0;
      
      this->set_kernel_args( kernel, std::make_index_sequence<sizeof...(Ts)>() );
      
      // start kernel
      cl::Event event;
//      printf("GS = (%lu,%lu,%lu) , LS = (%lu,%lu,%lu)\n", gs_0, gs_1, gs_2, ls_0, ls_1, ls_2 );
      cl::NDRange global_size( gs_0, gs_1, gs_2 );
      cl::NDRange local_size( ls_0, ls_1, ls_2 );
      
      // warm ups
      for( size_t i = 0 ; i < WARM_UPS ; ++i )
        error = _command_queue.enqueueNDRangeKernel( kernel, cl::NullRange, global_size, local_size, NULL, &event ); if( error != CL_SUCCESS ) throw std::exception();

      // kernel launch with profiling
      cl_ulong kernel_runtime_in_ns = 0;
      cl_ulong start_time;
      cl_ulong end_time;

      for( size_t i = 0 ; i < NUM_EVALUATIONS ; ++i )
      {
        error = _command_queue.enqueueNDRangeKernel( kernel, cl::NullRange, global_size, local_size, NULL, &event ); if( error != CL_SUCCESS ) throw std::exception();
        // check result
        if( _check_result )
          check_result_helper();
        
        error = event.wait(); check_error( error );
        
        event.getProfilingInfo( CL_PROFILING_COMMAND_START, &start_time );
        event.getProfilingInfo( CL_PROFILING_COMMAND_END,   &end_time   );

        kernel_runtime_in_ns += end_time - start_time;
      }
      
      
      // save thread configuration
      if( _thread_configuration != nullptr )
      {
        nd_range_t gs = { gs_0, gs_1, gs_2 };
        nd_range_t ls = { ls_0, ls_1, ls_2 };
        (*_thread_configuration)[ configuration ] = { gs, ls };
      }

      return kernel_runtime_in_ns / NUM_EVALUATIONS;
    }

    cl::Context& context() {
        return _context;
    }

    cl::CommandQueue& command_queue() {
        return _command_queue;
    }

    std::vector<cl::Buffer>& kernel_buffers() {
        return _kernel_buffers;
    }

    std::vector<size_t>& kernel_input_sizes() {
        return _kernel_input_sizes;
    }

    void read_last_buffer(size_t size, void* out) {
        _command_queue.enqueueReadBuffer(_kernel_buffers.back(), CL_TRUE, 0, size, out);
    }
  
  private:
    cl::Platform                   _platform;
    cl::Device                     _device;
    cl::Context                    _context;
    cl::CommandQueue               _command_queue;
  
    cl::Program                    _program;
    std::string                    _kernel_source;
    std::string                    _kernel_name;
    std::string                    _kernel_flags;
  
    std::tuple<Ts...>              _kernel_inputs;
    std::vector<cl::Buffer>        _kernel_buffers;
    std::vector<size_t>            _kernel_input_sizes;
  
    std::tuple< GS_0, GS_1, GS_2 > _global_size_pattern;
    std::tuple< LS_0, LS_1, LS_2 > _local_size_pattern;

    thread_configurations_t*       _thread_configuration;
  
    bool                           _check_result;
    size_t                         _num_wrong_results;
    std::vector<size_t>            _gold_sizes;
    std::vector<void*>             _gold_ptrs;
  
  
    // helper for creating buffers
    template< size_t... Is >
    void create_buffers( std::index_sequence<Is...> )
    {
      create_buffers_impl( std::get<Is>( _kernel_inputs )... );
    }
  
    template< typename T, typename... ARGs >
    void create_buffers_impl( const scalar<T>& scalar, ARGs&... args )
    {
      _kernel_input_sizes.emplace_back( 1 );
      
      create_buffers_impl( args... );
    }


    template< typename T, typename... ARGs >
    void create_buffers_impl( const buffer_class<T>& buffer, ARGs&... args )
    {
      auto start_time = std::chrono::system_clock::now();
     
      // add buffer size to _kernel_input_sizes
      _kernel_input_sizes.emplace_back( buffer.size() );
      
      // create buffer
      _kernel_buffers.emplace_back( _context, CL_MEM_READ_WRITE, buffer.size() * sizeof( T ) );
      
      try
      {
        error = _command_queue.enqueueWriteBuffer( _kernel_buffers.back(), CL_TRUE, 0, buffer.size() * sizeof( T ), buffer.get() ); check_error(error);
      }
      catch(cl::Error& err)
      {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
        abort();
      }
      
      auto end_time = std::chrono::system_clock::now();
      auto runtime  = std::chrono::duration_cast<std::chrono::milliseconds>( end_time - start_time ).count();
      std::cout << "Time to create and fill buffer: " << runtime << "ms" << std::endl;
      
      create_buffers_impl( args... );
    }
  
  
    template< typename... ARGs >
    void create_buffers_impl( const cl_mem& ocl_buffer, ARGs&... args )
    {
      // get buffer size
      size_t buffer_size;
      clGetMemObjectInfo( ocl_buffer, CL_MEM_SIZE, sizeof( size_t ), &buffer_size, NULL );
      
      _kernel_input_sizes.emplace_back( buffer_size );
      
      
      // set buffer
      _kernel_buffers.push_back( cl::Buffer(ocl_buffer) );
    }
  
  
    void create_buffers_impl()
    {}
  
    // helper for set kernel arguments
    template< size_t... Is >
    void set_kernel_args( cl::Kernel& kernel, std::index_sequence<Is...> )
    {
      set_kernel_args_impl( kernel, std::get<Is>( _kernel_inputs )... );
    }
  
  
    template< typename T, typename... ARGs >
    void set_kernel_args_impl( cl::Kernel& kernel, scalar<T> scalar, ARGs... args )
    {
      kernel.setArg( arg_index++, scalar.get() );
      
      set_kernel_args_impl( kernel, args... );
    }


    template< typename T, typename... ARGs >
    void set_kernel_args_impl( cl::Kernel& kernel, buffer_class<T> buffer, ARGs... args )
    {
      kernel.setArg( arg_index++, _kernel_buffers[ buffer_pos++ ] );
      
      set_kernel_args_impl( kernel, args... );
    }


    template< typename... ARGs >
    void set_kernel_args_impl( cl::Kernel& kernel, cl_mem buffer, ARGs... args )
    {
      kernel.setArg( arg_index++, _kernel_buffers[ buffer_pos++ ] );
      
      set_kernel_args_impl( kernel, args... );
    }
  
  
    void set_kernel_args_impl( cl::Kernel& kernel )
    {}
  
  
    void check_result_helper()
    {
      for( size_t i = 0 ; i < _kernel_buffers.size() ; ++i )
      {
        size_t size = _gold_sizes[ i ];
        
        char* dev_buffer_ptr = new char[ size ];
        _command_queue.enqueueReadBuffer( _kernel_buffers[ i ], CL_TRUE, 0, size, dev_buffer_ptr );
        
        auto equal = ( memcmp( dev_buffer_ptr, _gold_ptrs[i], size ) == 0 ) ? true : false;
        
#if 0 // delete (only for testing)
if( !equal )//i == 2 )
{
 for( size_t j = 0 ; j < size/sizeof(float) ; ++j )
   std::cout << "i =  " << i << ", dev -> " << reinterpret_cast<float*>(dev_buffer_ptr)[ j ] << std::endl;

 std::cout << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;
 
 for( size_t j = 0 ; j < size/sizeof(float) ; ++j )
   std::cout << "i =  " << i << ", gold -> " << reinterpret_cast<float*>( _gold_ptrs[i] )[j] << std::endl;
}
#endif

        if( !equal )
        {
          std::cout << "computation finished: RESULT NOT CORRECT ! ! !\n";
          ++_num_wrong_results;
          throw std::exception();
          return;
        }
      }
      
      std::cout << "computation finished: result correct\n";
      return;
    }
};



// factory<
template< typename GS_0, typename GS_1, typename GS_2,
          typename LS_0, typename LS_1, typename LS_2,
          typename... Ts
        >
auto ocl( const device_info&       device,
          const kernel_info&       kernel,
          const std::tuple<Ts...>& kernel_inputs,
         
          std::tuple< GS_0, GS_1, GS_2 > global_size,
          std::tuple< LS_0, LS_1, LS_2 > local_size
        )
{
  return ocl_cf_class< GS_0, GS_1, GS_2,
                       LS_0, LS_1, LS_2,
                       Ts...
                     >
                     ( device, kernel, kernel_inputs, global_size, local_size );
}


} // namespace cf

} // namespace atf

#endif /* defined( ocl_wrapper_h ) */
