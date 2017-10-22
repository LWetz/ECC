//
//  tp_value.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 18/11/2016.
//  Copyright © 2016 Ari Rasch. All rights reserved.
//

#ifndef tp_value_h
#define tp_value_h

#include <map>

#include "tp_value_node.hpp"
#include "value_type.hpp"

namespace atf
{

class tp_value
{
  public:
    tp_value() = default;
  
    tp_value& operator=( const tp_value& other ) = default;
    
    tp_value( const std::string& name, const value_type& value, void* tp_value_ptr );
  
    // read / write
    std::string& name();
    value_type&  value();
    void*        tp_value_ptr() const;


    // read only
    const std::string& name()  const;
    const value_type&  value() const;
  
  
    void update_tp() const;
  
    template< typename T >
    operator T()
    {
      return static_cast<T>( _value );
    }

//    template<>
    operator std::string()
    {
      return static_cast<std::string>( _value );
    }



  
  private:
    std::string _name;
    value_type  _value;
    void*       _tp_value_ptr;
};

// operators
std::ostream& operator<< (std::ostream &out, const tp_value& tp_value );
bool operator<( const tp_value& lhs, const tp_value& rhs );

// typedefs
typedef std::map<std::string, tp_value> configuration; //TODO refac: configuration -> config


} // namespace "atf"


#endif /* tp_value_h */
