//
//  op_wrapper.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 14/11/2016.
//  Copyright © 2016 Ari Rasch. All rights reserved.
//

#ifndef op_wrapper_h
#define op_wrapper_h


#include "helper.hpp"


namespace atf {

template< typename T >
void print_type()
{
  std::cout << "unknown type" << std::endl;
}

template<>
void print_type<int>()
{
  std::cout << "int" << std::endl;
}

template<>
void print_type<size_t>()
{
  std::cout << "size_t" << std::endl;
}


template<>
void print_type<bool>()
{
  std::cout << "bool" << std::endl;
}


template< typename T_lhs, typename T_rhs,
          typename callable
//          ,typename T_lhs_hlpr = typename std::conditional_t< std::is_rvalue_reference<T_lhs>::value, const typename std::remove_reference<T_lhs>::type, typename std::remove_reference<T_lhs>::type& >,
//          // TODO: zeile oben und unten der Übersicht halber runter in wrapper_class
//          typename T_rhs_hlpr = typename std::conditional_t< std::is_rvalue_reference<T_rhs>::value, const typename std::remove_reference<T_rhs>::type, typename std::remove_reference<T_rhs>::type& >
        >
class op_wrapper_class
{
    using T_lhs_ref_free = typename std::remove_reference<T_lhs>::type;
    using T_rhs_ref_free = typename std::remove_reference<T_rhs>::type;
  
    static const bool T_lhs_is_fundamental = std::is_fundamental< T_lhs_ref_free >::value;
    static const bool T_rhs_is_fundamental = std::is_fundamental< T_rhs_ref_free >::value;
  
    static const bool hard_copy_lhs = T_lhs_is_fundamental || std::is_rvalue_reference< T_lhs >::value;
    static const bool hard_copy_rhs = T_rhs_is_fundamental || std::is_rvalue_reference< T_rhs >::value;
  
    using T_lhs_save_type = std::conditional_t< hard_copy_lhs, T_lhs_ref_free, T_lhs_ref_free& >;
    using T_rhs_save_type = std::conditional_t< hard_copy_rhs, T_rhs_ref_free, T_rhs_ref_free& >;
  
    using T_lhs_scalar_type = typename std::remove_reference< typename std::conditional_t< T_lhs_is_fundamental, eval_t<T_lhs>, casted_eval_t<T_lhs> >::type >::type;
    using T_rhs_scalar_type = typename std::remove_reference< typename std::conditional_t< T_rhs_is_fundamental, eval_t<T_rhs>, casted_eval_t<T_rhs> >::type >::type;

  private:
    T_lhs_save_type _lhs; // big speedup (~2x) when not using T_lhs as reference
    T_rhs_save_type _rhs;
    callable        _func;

//    static const bool lhs_is_fundamental = std::is_fundamental< typename std::remove_reference<T_lhs_hlpr>::type >::value;
//    static const bool rhs_is_fundamental = std::is_fundamental< typename std::remove_reference<T_rhs_hlpr>::type >::value;
//    
//    using lhs_fundamental = typename std::remove_reference< typename std::conditional_t< lhs_is_fundamental, eval_t<T_lhs>, casted_eval_t<T_lhs> >::type >::type;
//    using rhs_fundamental = typename std::remove_reference< typename std::conditional_t< rhs_is_fundamental, eval_t<T_rhs>, casted_eval_t<T_rhs> >::type >::type;
  
  public:
    // TODO: make constructor private?
    op_wrapper_class( T_lhs lhs, T_rhs rhs, callable func )
      : _lhs( lhs ), _rhs( rhs ), _func( func )
    {
    }

//    template< typename tp_T, typename tp_range_t, typename tp_callable >
//    op_wrapper_class( const tp_t<tp_T, tp_range_t, tp_callable>& tp )
//      : _lhs( 0 ), _rhs( 0 ), _func( [&](){ return tp.cast(); } )
//    {}


    using T_res = typename std::result_of<callable(T_lhs_scalar_type, T_rhs_scalar_type)>::type; // BESSER
    //    using T_res = decltype( _func(std::declval<lhs_fundamental>(), std::declval<rhs_fundamental>()) ); // VORHEr
    //    using T_res = decltype( std::declval<lhs_fundamental>() + std::declval<rhs_fundamental>() );  // FUNKTIONIERT

  
    auto cast() const
    {
      return operator T_res();
    }
  
    operator T_res() const
    {
      auto lhs = static_cast<T_lhs_scalar_type>( _lhs );
      auto rhs = static_cast<T_rhs_scalar_type>( _rhs );
      
      return _func( lhs, rhs );
    }
};


// default
template< typename T_lhs_hlpr, typename T_rhs_hlpr, typename callable >
auto op_wrapper( T_lhs_hlpr&& lhs, T_rhs_hlpr&& rhs, callable func )
{
  return op_wrapper_class<T_lhs_hlpr&&, T_rhs_hlpr&&, callable>( std::forward<T_lhs_hlpr>(lhs), std::forward<T_rhs_hlpr>(rhs), func);
}

#if 0
// scalar wrapper
template< typename T, std::enable_if_t<std::is_fundamental<T>::value> >
auto op_wrapper( T scalar )
{
  return op_wrapper( 0, 0, [=](){ return scalar; } );
}


// tp wrapper
template< typename T, typename range_t, typename callable >
auto op_wrapper( const tp_t<T, range_t, callable>& tp )
{
  return op_wrapper( 0, 0, [&](){ return tp.cast(); } );
}
#endif


} // namespace "atf"

#endif /* op_wrapper_h */
