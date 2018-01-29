//
//  tuner_with_constraints_with_constraints.cpp
//  new_atf_lib
//
//  Created by Ari Rasch on 21/03/2017.
//  Copyright © 2017 Ari Rasch. All rights reserved.
//

#include <stdio.h>

#include "../include/tuner_with_constraints.hpp"
#include "../include/tp_value_node.hpp"


namespace atf
{
  
tuner_with_constraints::tuner_with_constraints( const bool& abort_on_error )
  : _search_space(), _abort_on_error( abort_on_error ), _number_of_evaluated_configs(), _number_of_invalid_configs(), _evaluations_required_to_find_best_found_result(), _history()
{
  _abort_condition = NULL; // new abort_condition_t( abort_condition );
  
  _history.emplace_back( std::chrono::high_resolution_clock::now(),
                         configuration{},
                         std::numeric_limits<size_t>::max()
                       );
}
  

tuner_with_constraints::tuner_with_constraints()
  : _search_space(), _abort_on_error( false ), _number_of_evaluated_configs(), _number_of_invalid_configs(), _evaluations_required_to_find_best_found_result(), _history()
{
  _abort_condition = NULL; // new abort_condition_t( abort_condition );
  
  _history.emplace_back( std::chrono::high_resolution_clock::now(),
                         configuration{},
                         std::numeric_limits<size_t>::max()
                       );
}
  
  
tuner_with_constraints::~tuner_with_constraints()
{
  std::cout << "\nNumber of Tree nodes: " << tp_value_node::number_of_nodes() << std::endl;
}


std::chrono::high_resolution_clock::time_point tuner_with_constraints::tuning_start_time() const
{
  return std::get<0>( _history.front() );
}


size_t tuner_with_constraints::search_space_size() const
{
	return _search_space.num_configs();
}

size_t tuner_with_constraints::number_of_evaluated_configs() const
{
  return _number_of_evaluated_configs;
}

size_t tuner_with_constraints::number_of_valid_evaluated_configs() const
{
  return _number_of_evaluated_configs - _number_of_invalid_configs;
}


size_t tuner_with_constraints::best_measured_result() const
{
  return std::get<2>( _history.back() );
}

configuration tuner_with_constraints::best_configuration() const
{
  return std::get<1>( _history.back() );
}

void tuner_with_constraints::insert_tp_names_in_search_space()
{}

void tuner_with_constraints::insert_tp_names_of_one_tree_in_search_space()
{}



// loeschen
void tuner_with_constraints::print_path()
{
  std::cout << std::endl;
}


} // namespace "atf"

