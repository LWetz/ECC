//
//  tp_value.cpp
//  new_atf_lib
//
//  Created by Ari Rasch on 31/01/2017.
//  Copyright © 2017 Ari Rasch. All rights reserved.
//

#include "../include/tp_value.hpp"


namespace atf
{

tp_value::tp_value( const std::string& name, const value_type& value, void* tp_value_ptr )
  : _name( name ), _value( value ), _tp_value_ptr( tp_value_ptr )
{}

// read / write
std::string& tp_value::name()
{
  return _name;
}

value_type& tp_value::value()
{
  return _value;
}

//void* tp_value::tp_value_ptr() const
//{
//  return _tp_value_ptr;
//}


// read only
const std::string& tp_value::name() const
{
  return _name;
}

const value_type& tp_value::value() const
{
  return _value;
}


    void tp_value::update_tp() const
    {
//      switch( _value.type_id() )
//      {
//        case value_type::root_t:
//          assert( false && "should never be reached" );
//          throw std::exception();
//          break;
//          
//        case value_type::int_t:
//          *static_cast<int*>( _tp_value_ptr.get() ) = _value.int_val();
//          break;
//          
//        case value_type::size_t_t:
//          *static_cast<size_t*>( _tp_value_ptr.get() ) = _value.size_t_val();
//          break;
//
//        case value_type::float_t:
//          *static_cast<float*>( _tp_value_ptr.get() ) = _value.float_val();
//          break;
//
//        case value_type::double_t:
//          *static_cast<double*>( _tp_value_ptr.get() ) = _value.double_val();
//          break;
//
//        case value_type::string_t:
//          *static_cast<std::string*>( _tp_value_ptr.get() ) = _value.string_val();
//          break;
//        
//        default:
//          throw std::exception();
//      }
      switch( _value.type_id() )
      {
        case value_type::root_t:
          assert( false && "should never be reached" );
          throw std::exception();
          break;
          
        case value_type::int_t:
          *static_cast<int*>( _tp_value_ptr ) = _value.int_val();
          break;
          
        case value_type::size_t_t:
          *static_cast<size_t*>( _tp_value_ptr ) = _value.size_t_val();
          break;

        case value_type::float_t:
          *static_cast<float*>( _tp_value_ptr ) = _value.float_val();
          break;

        case value_type::double_t:
          *static_cast<double*>( _tp_value_ptr ) = _value.double_val();
          break;

        case value_type::string_t:
          *static_cast<std::string*>( _tp_value_ptr ) = _value.string_val();
          break;
        
        default:
          throw std::exception();
      }
    }


  
// operators
std::ostream& operator<< (std::ostream &out, const tp_value& tp_value )
{
  auto value = tp_value.value();
  
  return operator<<(out, value);
}


bool operator<( const tp_value& lhs, const tp_value& rhs )
{
  assert( lhs.name() == rhs.name() );
  
  return lhs.value() < rhs.value();
}


} // namespace "atf"
