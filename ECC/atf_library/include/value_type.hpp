//
//  value_type.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 17/12/2016.
//  Copyright © 2016 Ari Rasch. All rights reserved.
//

#ifndef value_type_h
#define value_type_h

#include <string>

#if 0
    basic_ostream& operator<<(bool __n);
    basic_ostream& operator<<(short __n);
    basic_ostream& operator<<(unsigned short __n);
    basic_ostream& operator<<(int __n);
    basic_ostream& operator<<(unsigned int __n);
    basic_ostream& operator<<(long __n);
    basic_ostream& operator<<(unsigned long __n);
    basic_ostream& operator<<(long long __n);
    basic_ostream& operator<<(unsigned long long __n);
    basic_ostream& operator<<(float __f);
    basic_ostream& operator<<(double __f);
    basic_ostream& operator<<(long double __f);
    basic_ostream& operator<<(const void* __p);
#endif

namespace atf
{

class value_type
{
  public:
    enum type_id_t { root_t, bool_t, int_t, size_t_t, float_t, double_t, string_t };
  
    type_id_t type_id() const;
  
    // ctors
    value_type();
  
    value_type( const bool&        b   );
    value_type( const int&         i   );
    value_type( const size_t&      s_t );
    value_type( const float&       f   );
    value_type( const double&      d   );
    value_type( const std::string& s   );
  
    // TODO ...

  
    // access values
    bool        bool_val()   const;
    int         int_val()    const;
    size_t      size_t_val() const;
    float       float_val()  const;
    double      double_val() const;
    std::string string_val() const;
  
    // TODO ...

//    // const access values
//    const int&         int_val()    const;
//    const size_t&      size_t_val() const;
//    const std::string& string_val() const;
//  
//    // TODO ...

  
  
    // implicit cast operators
    operator bool()        const;
    operator int()         const;
    operator size_t()      const;
    operator float()       const;
    operator double()      const;
    operator std::string() const;
  
    // TODO ...
  
  
    // c-ctor, c-assignment and ddtor have to be explicitly stated due to std::string in union member
    value_type( const value_type& other );
  
    value_type& operator=( const value_type& other );
  
    ~value_type();
  
  private:
    type_id_t _type_id;
  
    union //value_t
    {
      bool        _bool_val;
      int         _int_val;
      size_t      _size_t_val;
      float       _float_val;
      double      _double_val;
      std::string _string_val;
    };
};

// overloaded operators
std::ostream& operator<< ( std::ostream &out, const value_type& value   );
bool          operator!= ( const value_type& lhs, const value_type& rhs );
bool          operator<  ( const value_type& lhs, const value_type& rhs );

} // namespace "atf"

#endif /* value_type_h */
