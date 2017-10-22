//
//  predicates.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 11/11/2016.
//  Copyright © 2016 Ari Rasch. All rights reserved.
//

#ifndef predicates_h
#define predicates_h


#include "op_wrapper.hpp"

namespace atf {

// TODO: in neue Datei "generators.hpp"
auto pow_2 = []( auto i )->int{ return pow(2,i); };


// divides
template< typename T> //, typename T_res = typename std::conditional_t< std::is_fundamental<T>::value, eval_t<T>, T_res_eval_t<T> >::type >
auto divides( const T& M ) //TODO: ref hier richtig?
{
//  if( M == 0 )
//    return [=]( auto i ){ return false; };
  
//  return [&]( auto i ){ return M % i == 0; };
  return [=]( auto i )->bool{ return (M / i) * i == M; };
}

// multiple_of
template< typename T> //, typename T_res = typename std::conditional_t< std::is_fundamental<T>::value, eval_t<T>, T_res_eval_t<T> >::type >
auto multiple_of( const T& M )
{
  return [=]( auto i )->bool{ return (i / M) * M == i; };
}

// less than
template< typename T>
auto less_than( const T& M )
{
  return [=]( auto i )->bool{ return i < M; };
}

// greater than
template< typename T>
auto greater_than( const T& M )
{
  return [=]( auto i )->bool{ return i > M; };
}

// less than or equal
template< typename T>
auto less_than_or_eq( const T& M )
{
  return [=]( auto i )->bool{ return i <= M; };
}

// greater than or equal
template< typename T>
auto greater_than_or_eq( const T& M )
{
  return [=]( auto i )->bool{ return i >= M; };
}


// equal
template< typename T>
auto equal( const T& M )
{
  return [=]( auto i )->bool{ return i == M; };
}


// unequal
template< typename T>
auto unequal( const T& M )
{
  return [=]( auto i )->bool{ return i != M; };
}


} // namespace "atf"


// operators
template<	typename func_t_1, typename func_t_2> // TODO: find error -> "typename = ::std::enable_if_t< atf::is_callable_v<func_t_1> && atf::is_callable_v<func_t_2> > >"
auto operator&&( func_t_1 lhs, func_t_2 rhs )
{
  return [=]( auto x )
            {
              if( lhs(x) ==  false )  // enables short circuit evaluation
                return false;
              else
                return static_cast<bool>( rhs(x) );
            };
}


template< typename func_t >
auto operator||( func_t lhs, func_t rhs )
{
  return [=]( auto x )
            {
              if( lhs(x) ==  true )  // enables short circuit evaluation
                return true;
              else
                return static_cast<bool>( rhs(x) );
            };
}


#endif /* predicates_h */
