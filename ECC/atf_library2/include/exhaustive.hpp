//
//  exhaustive.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 14/01/2017.
//  Copyright © 2017 Ari Rasch. All rights reserved.
//

#ifndef exhaustive_h
#define exhaustive_h

#include "tuner_with_constraints.hpp"
#include "tuner_without_constraints.hpp"

namespace atf
{

template< typename T = tuner_with_constraints>
class exhaustive_class : public T
{
  public:
    template< typename... Ts >
    exhaustive_class( Ts... params )
      : T(  params... )
    {}

  
    void initialize( const search_space& search_space )
    {
      _search_space = &search_space; // unnecessary: _search_space is initialized in parent "tuner" after call of "operator()( G_classes)"
    }
  
  
    configuration get_next_config()
    {
      static size_t pos = 0;
      
      if( pos == _search_space->num_configs() )
        pos = 0;
      
      return (*_search_space)[ pos++ ];
    }
  
    
    void report_result( const size_t& result )
    {}
  
  
    void finalize()
    {}

  
  private:
    search_space const* _search_space;
};


template< typename... Ts >
auto exhaustive( Ts... args )
{
  return exhaustive_class<>{ args... };
}


template< typename T, typename... Ts >
auto exhaustive( Ts... args )
{
  return exhaustive_class<T>{ args... };
}


} // namespace "atf"



#endif /* exhaustive_h */
