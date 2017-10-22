//
//  tuner_without_constraints_def.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 21/03/2017.
//  Copyright © 2017 Ari Rasch. All rights reserved.
//

#ifndef tuner_without_constraints_def_h
#define tuner_without_constraints_def_h


namespace atf
{

template< typename abort_condition_t >
tuner_without_constraints::tuner_without_constraints( const abort_condition_t& abort_condition, const bool& abort_on_error )
  : _search_space(), _abort_on_error( abort_on_error ), _number_of_evaluated_configs(), _number_of_invalid_configs(), _evaluations_required_to_find_best_found_result(), _valid_evaluations_required_to_find_best_found_result(), _history()
{
  _abort_condition = std::unique_ptr<cond::abort>( new abort_condition_t( abort_condition ) );
  
  _history.emplace_back( std::chrono::high_resolution_clock::now(),
                         configuration{},
                         std::numeric_limits<size_t>::max()
                       );

}


// first application of operator(): IS
template< typename    T , typename    range_t , typename    callable ,
          typename... Ts, typename... range_ts, typename... callables
        >
tuner_without_constraints& tuner_without_constraints::operator()( tp_t<T,range_t,callable>& tp, tp_t<Ts,range_ts,callables>&... tps )
{
  _search_space.add_tp( tp, static_cast<void*>( tp._act_elem.get() ) );
  
  this->operator()( tps... );
  
  return *this;
}

// first application of operator(): IS (required due to ambiguity)
template< typename T, typename range_t, typename callable >
tuner_without_constraints& tuner_without_constraints::operator()( tp_t<T,range_t,callable>& tp )
{
  _search_space.add_tp( tp, static_cast<void*>( tp._act_elem.get() ) );
  
  return *this;
}




template< typename callable >
configuration tuner_without_constraints::operator()( callable program ) // func must take config_t and return a value for which "<" is defined.
{
  std::cout << "\nsearch space size: " << _search_space.num_configs() << std::endl << std::endl;
  
  // if now abort condition is specified then iterate over the whole search space.
  if( _abort_condition == NULL )
    _abort_condition = std::unique_ptr<cond::abort>( new cond::evaluations( _search_space.num_configs() ) );
 
 
  auto start = std::chrono::system_clock::now();

  initialize( _search_space );
  
  size_t program_runtime = std::numeric_limits<size_t>::max();
  
  while( !_abort_condition->stop( *this ) )
  {
    auto config = get_next_config();
    
    ++_number_of_evaluated_configs;
    try
    {
      program_runtime = program( config );
    }
    catch( ... )
    {
      ++_number_of_invalid_configs;
      
      if( _abort_on_error )
        abort();
      else
        program_runtime = std::numeric_limits<size_t>::max();
    }
    
    auto current_best_result = std::get<2>( _history.back() );
    if( program_runtime < current_best_result  )
    {
      _evaluations_required_to_find_best_found_result = this->number_of_evaluated_configs();
      _valid_evaluations_required_to_find_best_found_result = this->number_of_valid_evaluated_configs();
      _history.emplace_back( std::chrono::high_resolution_clock::now(),
                             config,
                             program_runtime
                           );
    }

    report_result( program_runtime ); // TODO: refac "program_runtime" -> "program_cost"
    
//    #ifdef VERBOSE
    std::cout << std::endl << "evaluated configs: " << this->number_of_evaluated_configs() << " , valid configs: " << this->number_of_valid_evaluated_configs() << " , program cost: " << program_runtime << " , current best result: " << this->best_measured_result() << std::endl << std::endl;
//    #endif
  }
  
  finalize();
  
  std::cout << "\nnumber of evaluated configs: " << this->number_of_evaluated_configs() << " , number of valid configs: " << this->number_of_valid_evaluated_configs() << " , number of invalid configs: " << _number_of_invalid_configs << " , evaluations required to find best found result: " << _evaluations_required_to_find_best_found_result << " , valid evaluations required to find best found result: " << _valid_evaluations_required_to_find_best_found_result << std::endl;

  auto end = std::chrono::system_clock::now();
  auto runtime_in_sec = std::chrono::duration_cast<std::chrono::seconds>( end - start ).count();
  std::cout << std::endl << "total runtime for tuning = " << runtime_in_sec << "sec" << std::endl;
  
  
  // output
  auto        best_config = best_configuration();
  std::string seperator = "";
  std::cout << "\nbest configuration: [ ";
  for( const auto& tp : best_config )
  {
    auto tp_value = tp.second;
    std::cout << seperator << tp_value.name() << " = " << tp_value.value();
    seperator = " ; ";
  }
  std::cout << " ] with cost: " << this->best_measured_result() << std::endl << std::endl;

  // store best found result in file
  std::stringstream tp_names;
  std::stringstream tp_vals;
  for( auto& tp : best_config )
  {
    tp_names << tp.first  << ";";
    tp_vals  << tp.second << ";";
  }
  
  std::ofstream outfile;
  outfile.open("results.csv", std::ofstream::app ); // TODO: "/Users/arirasch/results.csv"
  outfile << "best_measured_result" << ";" << "number_of_valid_evaluated_configs" << ";" << "evaluations_required_to_find_best_found_result" << ";" << "valid_evaluations_required_to_find_best_found_result" ";" << tp_names.str() << std::endl;
  outfile << this->best_measured_result() << ";" << this->number_of_valid_evaluated_configs() << ";" << _evaluations_required_to_find_best_found_result << ";" << _valid_evaluations_required_to_find_best_found_result << ";" << tp_vals.str() << std::endl;
  outfile.close();
  
  auto best_configuration = std::get<1>( _history.back() );
  return best_configuration;
}


} // namespace "atf"


#endif /* tuner_without_constraints_def_h */
