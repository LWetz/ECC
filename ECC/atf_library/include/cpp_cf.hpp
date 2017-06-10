//
//  cpp_cf.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 12/01/2017.
//  Copyright © 2017 Ari Rasch. All rights reserved.
//

#ifndef cpp_cf_h
#define cpp_cf_h

#include <chrono>

namespace atf
{

namespace cf
{


template< typename duration = std::chrono::milliseconds, typename callable_t >
auto cpp( callable_t callable)
{
  return [=]( auto config )
         {  
           auto start = std::chrono::system_clock::now();
           
           callable( config );
           
           auto end = std::chrono::system_clock::now();
           auto runtime = std::chrono::duration_cast<duration>( end - start ).count();
           
           return runtime;
         };
}


} // namespace cf

} // namespace atf

#endif /* cpp_cf_h */
