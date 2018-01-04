//
//  tuner.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 29/10/2016.
//  Copyright © 2016 Ari Rasch. All rights reserved.
//

#ifndef tuner_h
#define tuner_h

#include <tuple>
#include <thread>

#include "helper.hpp"
#include "tp.hpp"

#include "abort_conditions.hpp"
#include "search_space.hpp"


namespace atf
{


  
class tuner
{
  public:
    virtual size_t        number_of_evaluated_configs()       const = 0;
    virtual size_t        number_of_valid_evaluated_configs() const = 0;
    virtual size_t        best_measured_result()              const = 0;
    virtual configuration best_configuration()                const = 0;
  
    virtual std::chrono::high_resolution_clock::time_point tuning_start_time() const = 0;
  
  protected:
    virtual void          initialize( const search_space& search_space) = 0;
    virtual void          finalize()                                    = 0;
    virtual configuration get_next_config()                             = 0;
    virtual void          report_result( const size_t& result )         = 0;
};


} // namespace "atf"

#endif /* tuner_h */
