//
//  tp.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 28/10/2016.
//  Copyright © 2016 Ari Rasch. All rights reserved.
//

#ifndef tp_h
#define tp_h

#include <vector>
#include <functional>
#include <math.h>

#include "range.hpp"

namespace atf
{


template< typename T, typename range_t, typename callable >
class tp_t
{
  friend class tuner_with_constraints;
  friend class tuner_without_constraints;
  
//  using lambda_type = T(*)(); //[](){ return T();}
  
  public:
    using type = T;
  
    // TODO: multiple predicates: "divides(M) && is_multiple_of(2)", etc.
    tp_t( const std::string& name, range_t range, const callable& predicate = [](T){ return true; } )
      : _name( name ), _range( range ), _predicate( predicate ), _act_elem( std::make_shared<T>() )
    {}
  
  
    std::string name() const
    {
      return _name;
    }
  
    // private machen; "tp_t" in tuner class als friend
    bool get_next_value( T& elem )
    {
      // get next element
      if( !_range.next_elem( elem ) )
        return false;

      // while predicate is not fullfilled on "elem" then get next element
      while( !_predicate( elem ) )
        if( !_range.next_elem( elem ) )
          return false;

      *_act_elem = elem;
      return true;
    }
  
    operator T() const
    {
      return *_act_elem;
    }
  
  
    auto cast() const
    {
      return *_act_elem;
    }
  
  
    range const* get_range_ptr() const
    {
      return &_range;
    }
  
  
//    operator op_wrapper_class<int,int, lambda_type> ()
//    {
//      return op_wrapper( 0, 0, [&](){ return _act_elem;} );
//    }
  
  private:
    const std::string        _name;
          range_t            _range;
    const callable           _predicate;
          std::shared_ptr<T> _act_elem;
};

//// required for conversion from tp_t to std::string
//template< typename T, typename range_t, typename callable >
//std::ostream& operator<<( std::ostream& os, tp_t< T, range_t, callable > const& tp)
//{
//  return os << tp.i;
//}

template< typename range_t, typename callable, typename T = typename range_t::out_type >
auto tp( const std::string& name, range_t range, const callable& predicate )
{
  return tp_t<T,range_t,callable>( name, range, predicate );
}


// function for deducing class template parameters of "tp_t"
template< typename range_t, typename T = typename range_t::in_type >
auto tp( const std::string& name, range_t range )
{
  return tp( name, range, [](T){ return true; }  );
}



// enables initializer lists in tp definition

template< typename T, typename callable >
auto tp( const std::string& name, const std::initializer_list<T>& elems, const callable& predicate )
{
  return tp_t<T,set<T>,callable>( name, set< typename std::remove_reference<T>::type >( elems ), predicate );
}
 

// function for deducing class template parameters of "tp_t"
template< typename T >
auto tp( const std::string& name, const std::initializer_list<T>& elems )
{
  return tp( name, elems, [](T){ return true; }  );
}



} // namespace "atf"

#endif /* tp_h */
