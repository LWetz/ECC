//
//  tuner_without_constraints.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 21/03/2017.
//  Copyright © 2017 Ari Rasch. All rights reserved.
//

#ifndef tuner_without_constraints_h
#define tuner_without_constraints_h

#include <tuple>
#include <thread>

#include "tuner.hpp"

#include "helper.hpp"
#include "tp.hpp"

#include "abort_conditions.hpp"
#include "search_space_simple.hpp"



namespace atf
{
  
class tuner_without_constraints : public tuner
{
  public:
    template< typename abort_condition_t >
    tuner_without_constraints( const abort_condition_t& abort_condition, const bool& abort_on_error = false );
  
    tuner_without_constraints( const bool& abort_on_error );
    tuner_without_constraints();

    tuner_without_constraints( const tuner_without_constraints&  ) = default;
    tuner_without_constraints(       tuner_without_constraints&& ) = default;
  
    virtual ~tuner_without_constraints();

    // set tuning parameters
    template< typename    T , typename    range_t , typename    callable ,
              typename... Ts, typename... range_ts, typename... callables
            >
    tuner_without_constraints& operator()( tp_t<T,range_t,callable>& tp, tp_t<Ts,range_ts,callables>&... tps );
  
    // first application of operator(): IS (required due to ambiguity)
    template< typename T, typename range_t, typename callable >
    tuner_without_constraints& operator()( tp_t<T,range_t,callable>& tp );
  

    template< typename callable >
    configuration operator()( callable program ); // program must take config_t and return a size_t

    std::chrono::high_resolution_clock::time_point tuning_start_time() const;

    size_t number_of_evaluated_configs()       const;
    size_t number_of_valid_evaluated_configs() const;
  
    size_t best_measured_result()      const;
  
    configuration best_configuration() const;
  

  protected:
    virtual void          initialize( const search_space& search_space) = 0;
    virtual void          finalize()                                    = 0;
    virtual configuration get_next_config()                             = 0;
    virtual void          report_result( const size_t& result )         = 0;

    search_space_simple        _search_space;
  
  private:
    std::unique_ptr<cond::abort> _abort_condition;
    const bool                   _abort_on_error;
  
    size_t                       _number_of_evaluated_configs;
    size_t                       _number_of_invalid_configs;
    size_t                       _evaluations_required_to_find_best_found_result;
    size_t                       _valid_evaluations_required_to_find_best_found_result;
  
    using                         history_entry = std::tuple< std::chrono::high_resolution_clock::time_point, configuration, size_t >; // entry: actual tuning runtime, configuration, configuration's cost
    std::vector<history_entry>   _history; // history of best results
    
    std::vector<std::thread>     _threads;
  
  
    template< typename T, typename range_t, typename callable, typename... Ts >
    void insert_tp_names_in_search_space( tp_t<T,range_t,callable>& tp, Ts&... tps );
  
};

} // namespace "atf"

using NO_CONSTRAINTS = atf::tuner_without_constraints;

#include "detail/tuner_without_constraints_def.hpp"

#endif /* tuner_without_constraints_h */
