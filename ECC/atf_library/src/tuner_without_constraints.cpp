//
//  tuner_without_constraints.cpp
//  new_atf_lib
//
//  Created by Ari Rasch on 21/03/2017.
//  Copyright © 2017 Ari Rasch. All rights reserved.
//

#include <stdio.h>

#include "../include/tuner_without_constraints.hpp"

namespace atf
{
  
tuner_without_constraints::tuner_without_constraints( const bool& abort_on_error )
  : _search_space(), _abort_on_error( abort_on_error ), _number_of_evaluated_configs(), _number_of_invalid_configs(), _evaluations_required_to_find_best_found_result(), _history()
{
  _abort_condition = NULL; // new abort_condition_t( abort_condition );
  
  _history.emplace_back( std::chrono::high_resolution_clock::now(),
                         configuration{},
                         std::numeric_limits<size_t>::max()
                       );
}
  

tuner_without_constraints::tuner_without_constraints()
  : _search_space(), _abort_on_error( false ), _number_of_evaluated_configs(), _number_of_invalid_configs(), _evaluations_required_to_find_best_found_result(), _history()
{
  _abort_condition = NULL; // new abort_condition_t( abort_condition );
  
  _history.emplace_back( std::chrono::high_resolution_clock::now(),
                         configuration{},
                         std::numeric_limits<size_t>::max()
                       );
}
  
  
tuner_without_constraints::~tuner_without_constraints()
{
  std::cout << "\nNumber of Tree nodes: " << tp_value_node::number_of_nodes() << std::endl;
}


std::chrono::high_resolution_clock::time_point tuner_without_constraints::tuning_start_time() const
{
  return std::get<0>( _history.front() );
}


size_t tuner_without_constraints::number_of_evaluated_configs() const
{
  return _number_of_evaluated_configs;
}


size_t tuner_without_constraints::number_of_valid_evaluated_configs() const
{
  return _number_of_evaluated_configs - _number_of_invalid_configs;
}


size_t tuner_without_constraints::best_measured_result() const
{
  return std::get<2>( _history.back() );
}

configuration tuner_without_constraints::best_configuration() const
{
  return std::get<1>( _history.back() );
}


} // namespace "atf"
