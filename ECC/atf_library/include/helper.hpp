//
//  helper.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 14/11/2016.
//  Copyright © 2016 Ari Rasch. All rights reserved.
//

#ifndef helper_h
#define helper_h

#include <sstream>
#include <vector>
#include <map>

namespace atf {

  // helper for large vectors
  template< typename T >
  class sparse_vector
  {
    public:
      sparse_vector() = default;
      sparse_vector( const size_t size, const T& default_value )
        : _size( size ), _default_value( default_value )
      {}
    
//      void set_size( const size_t size, const T& default_value )
//      {
//
//      }
    
      T& operator[]( const size_t& index )
      {
//        return _index_value_pairs[ index ];

#if 1
        try
        {
          return _index_value_pairs.at( index );
        }
        catch( std::out_of_range )
        {
          _index_value_pairs[ index ] = _default_value;
          return _index_value_pairs[ index ];
        }
#endif
      }
    
    
      size_t size() const
      {
        return _size;
      }
    
    
    private:
      size_t               _size;
      T                    _default_value;
      std::map< size_t, T> _index_value_pairs;
    
  };



template< typename T >
struct eval_t
{
  using type = T;
};


template< typename T >
struct casted_eval_t
{
  using type = decltype( std::declval<T>().cast() );
};


template<>
struct casted_eval_t< std::string >
{
  using type = std::string;
};


template< typename T >
struct T_res_eval_t
{
  using type = typename T::T_res;
};

template< class T >
std::string to_string(const T& t)
{
  std::ostringstream oss; // create a stream
  oss << t;               // insert value to stream
  return oss.str();       // extract value and return
}


template< typename T >
class scalar
{
  public:
    scalar( T val )
      : _val( val )
    {}

    scalar()
      : _val()
    {
      const T min_value = std::numeric_limits<T>::min();
      const T max_value = std::numeric_limits<T>::max();

      const auto random_seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::srand( static_cast<unsigned int>( random_seed ) ); // seed random number generator with current time
    
      double normalized_random_number  = static_cast<double>( rand() ) / static_cast<double>( RAND_MAX ); // random number in [0,1]_double
      T      range                     = max_value / static_cast<T>( 2 );
      T      res                       = min_value + static_cast<T>( normalized_random_number * range );
      
      _val = res;
    }


    T get()
    {
      return _val;
    }
  
    T* get_ptr()
    {
      return &_val;
    }
  
  private:
    T _val;
};


template< typename T >
class buffer_class
{
  public:
    buffer_class( const std::vector<T>& vector )
      : _vector( vector )
    {}

    buffer_class( const size_t& size )
      : _vector( size )
    {
      fill_vector_with_random_numbers();
    }

    size_t size() const
    {
      return _vector.size();
    }
  
    const T* get() const
    {
      return _vector.data();
    }
  
    const std::vector<T>& get_vector() const
    {
      return _vector;
    }
  
  private:
    std::vector<T> _vector;
  
    //TODO: prüfen
    void fill_vector_with_random_numbers()
    {
      const T min_value = std::numeric_limits<T>::min();
      const T max_value = std::numeric_limits<T>::max();

      const auto random_seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::srand( static_cast<unsigned int>( random_seed ) ); // seed random number generator with current time
      
      for( auto& elem : _vector )
      {
        double normalized_random_number  = static_cast<double>( rand() ) / static_cast<double>( RAND_MAX ); // random number in [0,1]_double
        T      range                     = max_value / static_cast<T>( 2 );
        T      res                       = min_value + static_cast<T>( normalized_random_number * range );
        
        elem = res;
      }
    }
  
};

// factory for "buffer"
template< typename T >
auto buffer( const std::vector<T>& vector )
{
  return buffer_class<T>( vector );
}

template< typename T >
auto buffer( const size_t& size )
{
  return buffer_class<T>( size );
}


//template< typename T >
//class rand_buffer
//{
//  public:
//    rand_buffer( size_t size )
//      : _size( size )
//    {}
//  
//    size_t size() const
//    {
//      return _size;
//    }
//  
//  private:
//    size_t _size;
//};


template< typename... Ts >
std::tuple<Ts...> inputs( Ts... inputs )
{
  return std::tuple<Ts...>( inputs... );
}



// helper for is_callable

	namespace detail
	{
		// void_t: Helper alias that sinks all supplied template parameters
		// and is always void.
		template< typename... Ts >
		using void_t = void;

		// is_callable_impl: Implements "is_callable" type-trait using SFINAE.
		// We check for occurence of any valid operator() overload in the given type.
		
		// Fallback case: this is selected when no operator() is found.
		template< typename Void, typename T >
		struct is_callable_impl
			: ::std::false_type
		{
		};

		// TRUE case
		template< typename T >
		struct is_callable_impl	<void_t<decltype(&T::operator())>, T>
			: ::std::true_type
		{
		};
		//
	}

	// is_callable: Type trait that applies to all callable types, excluding function pointers.
	template< typename T >
	using is_callable = detail::is_callable_impl<void, T>;

	template< typename T >
	constexpr bool is_callable_v = is_callable<T>::value;



} // namespace "atf"

#endif /* helper_h */
