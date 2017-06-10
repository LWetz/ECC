//
//  abort_conditions.cpp
//  new_atf_lib
//
//  Created by Ari Rasch on 19/11/2016.
//  Copyright © 2016 Ari Rasch. All rights reserved.
//

#include <iostream>


#include "../include/tuner.hpp"
#include "../include/abort_conditions.hpp"


namespace atf
{

namespace cond
{


abort::~abort()
{}

//template< typename... Ts >
//or_class::or_class( const Ts&... conditions )
//{
//  add( conditions... );
//}

//or::~or()
//{
//  for( const auto& elem : s )
//    delete elem;
//}

bool or_class::stop( const tuner& tuner )
{
  for( const auto& cond : s )
    if( cond->stop( tuner ) )
      return true;
  
  return false;
}

// IS
//template< typename T, typename... Ts, std::enable_if_t< std::is_base_of<abort, T>::value >* >
//void or_class::add( const T& condition, const Ts&... )
//{
//  auto res = std::make_shared<T>( condition );
//  
//  s.push_back( res );
//}

// IA
void or_class::add()
{}

//template< typename T_lhs, typename T_rhs, std::enable_if_t< std::is_base_of<abort, T_lhs>::value && std::is_base_of<abort, T_rhs>::value > >
//or_class operator||( const T_lhs& lhs, const T_rhs& rhs )
//{
//  return or_class( lhs, rhs );
//}


//template< typename... Ts >
//and_class::and_class( const Ts&... conditions )
//{
//  add( conditions... );
//}

//and::~and()
//{
//  for( const auto& elem : s )
//    delete elem;
//}

bool and_class::stop( const tuner& tuner )
{
  for ( const auto& cond : s )
    if ( cond->stop(tuner))
      return false;
  
  return true;
}

// IS
//template< typename T, typename... Ts, std::enable_if_t< std::is_base_of<abort, T>::value >* >
//void and_class::add( const T& condition, const Ts&... )
//{
//  auto res = std::make_shared<T>(condition);
//  
//  s.push_back( res );
//}

// IA
void and_class::add()
{}



//template< typename T_lhs, typename T_rhs, std::enable_if_t< std::is_base_of<abort, T_lhs>::value && std::is_base_of<abort, T_rhs>::value > >
//and_class operator&&( const T_lhs& lhs, const T_rhs& rhs )
//{
//  return and_class( lhs, rhs );
//}


evaluations::evaluations( const size_t& num_evaluations )
  : _num_evaluations( num_evaluations )
{}


bool evaluations::stop( const tuner& tuner )
{
  auto number_of_evaluated_configs = tuner.number_of_evaluated_configs();
  
  //#ifdef VERBOSE
  //std::cout << "number of evaluated configs: " << number_of_evaluated_configs << std::endl;
  //#endif
  
  return number_of_evaluated_configs >= _num_evaluations;
}


valid_evaluations::valid_evaluations( const size_t& num_evaluations )
  : _num_evaluations( num_evaluations )
{}


bool valid_evaluations::stop( const tuner& tuner )
{
  auto number_of_valid_evaluated_configs = tuner.number_of_valid_evaluated_configs();
  
  //#ifdef VERBOSE
  // std::cout << "number of evaluated configs: " << number_of_evaluated_configs << std::endl;
  //#endif
  
  return number_of_valid_evaluated_configs >= _num_evaluations;
}


speedup::speedup( const double& speedup, const size_t& num_configs, const bool& only_valid_configs )
  : _speedup( speedup ), _duration(), _num_configs( num_configs ), _type( DurationType::NUM_CONFIGS ), _only_valid_configs( only_valid_configs )
{}

speedup::speedup( const double& speedup, const std::chrono::milliseconds& duration, const bool& only_valid_configs )
  : _speedup( speedup ), _duration( duration ), _num_configs(), _type( DurationType::TIME ), _only_valid_configs( only_valid_configs )
{}

bool speedup::stop( const tuner& tuner )
{
  if( _only_valid_configs && tuner.best_measured_result() == std::numeric_limits<size_t>::max() );
  else
    _verbose_history.emplace_back( tuner.best_measured_result() );
  
  // two cases
  if( _type == NUM_CONFIGS )
  {
    // starting phase
    if( _verbose_history.size() < _num_configs )
      return false;
    else
    {
      auto last_best_result = _verbose_history[ _verbose_history.size() - _num_configs ];
      auto best_result      = _verbose_history.back();
      auto speedup          = static_cast<double>( last_best_result ) / static_cast<double>( best_result );
      
      //#ifdef VERBOSE
      std::cout << "last best result: " << last_best_result << std::endl;
      std::cout << "best result:      " << best_result      << std::endl;
      std::cout << "Speedup:          " << speedup          << std::endl << std::endl;
      //#endif
    
      return speedup <= _speedup ;
    }
  }
  
  else if( _type == TIME )
  {
    assert( false ); //TODO
  }
  
  assert( false ); // should never be reached
  return true;
}


//  // running phase
//  size_t best_result      = tuner.best_measured_result();
//  size_t last_best_result;// = tuner.worst_measured_result(2); // TODO: "2"?
// 
//  if( _type == NUM_CONFIGS )
//    last_best_result = tuner.worst_measured_result( _num_configs );
//    
//  else if( _type == TIME )
//    last_best_result = tuner.worst_measured_result( _duration );
//  
//  auto speedup = last_best_result / best_result;
//  
//  //#ifdef VERBOSE
//  std::cout << "last best result: " << last_best_result << std::endl;
//  std::cout << "best result:      " << best_result      << std::endl;
//  std::cout << "Speedup:          " << speedup          << std::endl << std::endl;
//  //#endif
//  
//  return speedup <= _speedup; // "=" required for speedup = 1 (i.e., the last configurations have to be equal cost)
//}


//template< typename T >
//duration::duration( std::chrono::duration<T> duration )
//  : _duration( duration )
//{}

template< typename duration_t >
bool duration< duration_t >::stop( const tuner& tuner )
{
  auto current_tuning_time = std::chrono::high_resolution_clock::now() - tuner.tuning_start_time();
  
//  #ifdef VERBOSE
//  std::cout << "current tuning duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(current_tuning_duration).count() << std::endl;
//  #endif
  
  return current_tuning_time > _duration;
}


template class duration<std::chrono::nanoseconds>;
template class duration<std::chrono::microseconds>;
template class duration<std::chrono::milliseconds>;
template class duration<std::chrono::seconds>;
template class duration<std::chrono::minutes>;
template class duration<std::chrono::hours>;


  
result::result( size_t result )
  : _result( result )
{}
  
bool result::stop( const tuner& tuner )
{
  auto current_best_result = tuner.best_measured_result();
  
//  #ifdef VERBOSE
//  std::cout << "current best result: " << current_best_result << std::endl << std::endl;
//  #endif
  
  return current_best_result <= _result;
}


//no_changes::no_changes( size_t num_configs )
//{
//
//}
//  
//bool no_changes::stop( const tuner& tuner )
//{
//  static size_t actual_cost = tuner.best_measured_result();
//}



} // namespace "cond"

} // namespace "atf"
