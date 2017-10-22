//
//  tuner_with_constraints_def.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 21/03/2017.
//  Copyright © 2017 Ari Rasch. All rights reserved.
//

#ifndef tuner_with_constraints_def_h
#define tuner_with_constraints_def_h

#include <fstream>
#include <limits>
#include <chrono>


namespace atf
{


template< typename... TPs >
G_class<TPs...>::G_class( TPs&... tps )
  : _tps( tps... )
{}
  
template< typename... TPs >
auto G_class<TPs...>::tps() const
{
  return _tps;
}


template< typename abort_condition_t >
tuner_with_constraints::tuner_with_constraints( const abort_condition_t& abort_condition, const bool& abort_on_error )
  : _search_space(), _abort_on_error( abort_on_error ), _number_of_evaluated_configs(), _number_of_invalid_configs(), _evaluations_required_to_find_best_found_result(), _valid_evaluations_required_to_find_best_found_result(), _history()
{
  _abort_condition = std::unique_ptr<cond::abort>( new abort_condition_t( abort_condition ) );
  
  _history.emplace_back( std::chrono::high_resolution_clock::now(),
                         configuration{},
                         std::numeric_limits<size_t>::max()
                       );

}


// first application of operator(): IS
template< typename... Ts, typename... range_ts, typename... callables >
tuner_with_constraints& tuner_with_constraints::operator()( tp_t<Ts,range_ts,callables>&... tps )
{
  return this->operator()( G(tps...) );
}

// first application of operator(): IS (required due to ambiguity)
template< typename T, typename range_t, typename callable >
tuner_with_constraints& tuner_with_constraints::operator()( tp_t<T,range_t,callable>& tp )
{
  return this->operator()( G(tp) );
}


// second case
template< typename... Ts, typename... G_CLASSES >
tuner_with_constraints& tuner_with_constraints::operator()( G_class<Ts...> G_class, G_CLASSES... G_classes )
{
  const size_t num_trees = sizeof...(G_CLASSES) + 1;
  _search_space.append_new_trees( num_trees );
  
  insert_tp_names_in_search_space( G_class, G_classes... );
  
  return generate_config_trees<num_trees>( G_class, G_classes... );
}


template< typename callable >
configuration tuner_with_constraints::operator()( callable program ) // func must take config_t and return a value for which "<" is defined.
{
  std::cout << "\nsearch space size: " << _search_space.num_configs() << std::endl << std::endl;
  
  // if no abort condition is specified then iterate over the whole search space.
  if( _abort_condition == NULL )
    _abort_condition = std::unique_ptr<cond::abort>( new cond::evaluations( _search_space.num_configs() ) );
 
 
  auto start = std::chrono::system_clock::now();

  initialize( _search_space );
  
  size_t program_runtime = std::numeric_limits<size_t>::max();
  
  // logging: cost
  std::ofstream outfile_cost;
  outfile_cost.open( "/Users/arirasch/cost.csv", std::ofstream::app );
  
  // logging: meta
  std::ofstream outfile_meta;
  outfile_meta.open( "/Users/arirasch/meta.csv", std::ofstream::app );
  
  auto start_time = std::chrono::system_clock::now();
  
  while( !_abort_condition->stop( *this ) )
  {
    auto config = get_next_config();
    
    // logging: write headers
    static bool first_run = true;
    if( first_run )
    {
      // cost
      for( auto& tp : config )
        outfile_cost << tp.first << ";";
      outfile_cost << "cost" << std::endl;
      
      // meta
      outfile_meta << "number_of_evaluated_configs" << ";" << "number_of_valid_evaluated_configs" << ";" << "number_of_invalid_evaluated_configs"                                           << ";" << "evaluations_required_to_find_best_found_result" << ";" << "valid_evaluations_required_to_find_best_found_result"<< ";" << std::endl;
      
      first_run = false;
    }
    
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

    // logging: cost
    if( start_time + std::chrono::seconds(1) <= std::chrono::system_clock::now() )
    {
      start_time += std::chrono::seconds(1);
      
      for( auto& tp : this->best_configuration() )
        outfile_cost << tp.second << ";";
      outfile_cost << this->best_measured_result() << std::endl;
    }
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

  // logging: cost
  outfile_cost.close();
  
  // logging: meta
  outfile_meta << this->number_of_evaluated_configs() << ";" << this->number_of_valid_evaluated_configs() << ";" << this->number_of_evaluated_configs() - this->number_of_valid_evaluated_configs() << ";" << _evaluations_required_to_find_best_found_result  << ";" << _valid_evaluations_required_to_find_best_found_result << ";" << std::endl;
  outfile_meta.close();
  
  auto best_configuration = std::get<1>( _history.back() );
  return best_configuration;
}


template< typename... Ts, typename... rest_tp_tuples >
void tuner_with_constraints::insert_tp_names_in_search_space( G_class<Ts...> tp_tuple, rest_tp_tuples... tuples )
{
  insert_tp_names_of_one_tree_in_search_space ( tp_tuple, std::make_index_sequence<sizeof...(Ts)>{} );
  
  insert_tp_names_in_search_space( tuples... );
}


template< typename... Ts, size_t... Is>
void tuner_with_constraints::insert_tp_names_of_one_tree_in_search_space ( G_class<Ts...> tp_tuple, std::index_sequence<Is...> ) // TODO refac: insert_tp_names_of_one_tree_in_search_space  -> insert_tp_names_of_one_tree_in_search_space
{
  insert_tp_names_of_one_tree_in_search_space ( std::get<Is>( tp_tuple.tps() )... );
}

template< typename T, typename range_t, typename callable, typename... Ts >
void tuner_with_constraints::insert_tp_names_of_one_tree_in_search_space ( tp_t<T,range_t,callable>& tp, Ts&... tps )
{
  _search_space.add_name( tp.name() );
  
  insert_tp_names_of_one_tree_in_search_space ( tps... );
}


template< size_t TREE_ID, typename... Ts, typename... rest_tp_tuples>
tuner_with_constraints& tuner_with_constraints::generate_config_trees( G_class<Ts...> tp_tuple, rest_tp_tuples... tuples )
{
  return generate_config_trees<TREE_ID>( tp_tuple, std::make_index_sequence<sizeof...(Ts)>{}, tuples... );
}


template< size_t TREE_ID, typename... Ts, size_t... Is, typename... rest_tp_tuples >
tuner_with_constraints& tuner_with_constraints::generate_config_trees( G_class<Ts...> tp_tuple, std::index_sequence<Is...>, rest_tp_tuples... tuples )
{
  // fill generated config tree
  const size_t TREE_DEPTH = sizeof...(Is);
  _threads.emplace_back( [=](){ generate_single_config_tree< TREE_ID, TREE_DEPTH >( std::get<Is>( tp_tuple.tps() )... ); } );

  return generate_config_trees< TREE_ID - 1 >( tuples... );
}


template< size_t TREE_ID >
tuner_with_constraints& tuner_with_constraints::generate_config_trees()
{
  for( auto& thread : _threads )
    thread.join();
  
  _threads.clear();

  return *this;
}


template< size_t TREE_ID, size_t TREE_DEPTH, typename T, typename range_t, typename callable, typename... Ts, std::enable_if_t<( TREE_DEPTH > 0 )>* >
void tuner_with_constraints::generate_single_config_tree( tp_t<T,range_t,callable>& tp, Ts&... tps )
{
  T value;
  while( tp.get_next_value(value) )
  {
    auto value_tp_pair = std::make_pair( value, static_cast<void*>( tp._act_elem.get() /*&(tp._act_elem)*/ ) );
    generate_single_config_tree< TREE_ID, TREE_DEPTH-1 >( tps..., value_tp_pair );
  }
}


template< size_t TREE_ID, size_t TREE_DEPTH, typename... Ts, std::enable_if_t<( TREE_DEPTH == 0 )>*  >
void tuner_with_constraints::generate_single_config_tree( const Ts&... values )
{
  _search_space.tree( _search_space.num_trees() -  TREE_ID ).insert( values... ); // read TREE_IDs from right to left!
  //print_path( tps... );
}


// TODO: loeschen
template< typename T, typename... Ts >
void tuner_with_constraints::print_path(T val, Ts... tps)
{
  std::cout << val << " ";
  print_path(tps...);
}


} // namespace "atf"

#endif /* tuner_with_constraints_def_h */
