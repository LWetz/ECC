//
//  tuner_with_constraints.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 21/03/2017.
//  Copyright © 2017 Ari Rasch. All rights reserved.
//

#ifndef tuner_with_constraints_h
#define tuner_with_constraints_h

#include <tuple>
#include <thread>

#include "tuner.hpp"

#include "helper.hpp"
#include "tp.hpp"

#include "abort_conditions.hpp"
#include "search_space_tree.hpp"


namespace atf
{

// helper
template< typename... TPs >
class G_class
{
  public:
    G_class( TPs&... tps );
  
    G_class()                 = delete;
    G_class( const G_class& ) = default;
    G_class(       G_class& ) = default;
  
    ~G_class() = default;
  
    auto tps() const;

  private:
    std::tuple<TPs&...> _tps;
  
};

template< typename... TPs >
auto G( TPs&... tps )
{
  return G_class<TPs...>( tps... );
}
  
  
class tuner_with_constraints : public tuner
{
  public:
    template< typename abort_condition_t >
    tuner_with_constraints( const abort_condition_t& abort_condition, const bool& abort_on_error = false );
  
    tuner_with_constraints( const bool& abort_on_error );
    tuner_with_constraints();


    tuner_with_constraints( const tuner_with_constraints& other ) // = default;
      :
      _abort_condition( other._abort_condition->copy() ),
      _abort_on_error( other._abort_on_error ),
      _number_of_evaluated_configs( other._number_of_evaluated_configs ),
      _number_of_invalid_configs( other._number_of_invalid_configs ),
      _evaluations_required_to_find_best_found_result( other._evaluations_required_to_find_best_found_result  ),
      _valid_evaluations_required_to_find_best_found_result( other._valid_evaluations_required_to_find_best_found_result ),
      _history( other._history ),
      _threads()
    {}

    tuner_with_constraints( tuner_with_constraints&& other ) //= default;
      :
      _abort_condition( other._abort_condition->copy() ),
      _abort_on_error( other._abort_on_error ),
      _number_of_evaluated_configs( other._number_of_evaluated_configs ),
      _number_of_invalid_configs( other._number_of_invalid_configs ),
      _evaluations_required_to_find_best_found_result( other._evaluations_required_to_find_best_found_result  ),
      _valid_evaluations_required_to_find_best_found_result( other._valid_evaluations_required_to_find_best_found_result ),
      _history( other._history ),
      _threads()
    {}

  
    virtual ~tuner_with_constraints();

    // set tuning parameters
    template< typename... Ts, typename... range_ts, typename... callables >
    tuner_with_constraints& operator()( tp_t<Ts,range_ts,callables>&... tps );
  
    // first application of operator(): IS (required due to ambiguity)
    template< typename T, typename range_t, typename callable >
    tuner_with_constraints& operator()( tp_t<T,range_t,callable>& tp );
  

    // set tuning parameters
    template< typename... Ts, typename... G_CLASSES >
    tuner_with_constraints& operator()( G_class<Ts...> G_class, G_CLASSES... G_classes );

    template< typename callable >
//          typename T, typename range_t, typename tp_callable, std::enable_if_t<( !std::is_same<callable, tp_t<T,range_t,tp_callable> >::value )>* = nullptr >
    configuration operator()( callable program ); // program must take config_t and return a size_t

    std::chrono::high_resolution_clock::time_point tuning_start_time() const;

    size_t number_of_evaluated_configs()       const;
    size_t number_of_valid_evaluated_configs() const;
  
    size_t best_measured_result()      const;
  
    configuration best_configuration() const;
  
	size_t search_space_size()		const;

  private:
    template< typename... Ts, typename... rest_tp_tuples >
    void insert_tp_names_in_search_space( G_class<Ts...> tp_tuple, rest_tp_tuples... tuples );

    void insert_tp_names_in_search_space();

    template< typename... Ts, size_t... Is >
    void insert_tp_names_of_one_tree_in_search_space( G_class<Ts...> tp_tuple, std::index_sequence<Is...> );

    template< typename T, typename range_t, typename callable, typename... Ts >
    void insert_tp_names_of_one_tree_in_search_space( tp_t<T,range_t,callable>& tp, Ts&... tps );
  
    void insert_tp_names_of_one_tree_in_search_space();
  
    template< size_t TREE_ID, typename... Ts, typename... rest_tp_tuples>
    tuner_with_constraints& generate_config_trees( G_class<Ts...> tp_tuple, rest_tp_tuples... tuples );

    template< size_t TREE_ID, typename... Ts, size_t... Is, typename... rest_tp_tuples >
    tuner_with_constraints& generate_config_trees( G_class<Ts...> tp_tuple, std::index_sequence<Is...>, rest_tp_tuples... tuples );

    template< size_t TREE_ID >
    tuner_with_constraints& generate_config_trees();

    template< size_t TREE_ID, size_t TREE_DEPTH, typename T, typename range_t, typename callable, typename... Ts, std::enable_if_t<( TREE_DEPTH>0 )>* = nullptr >
    void generate_single_config_tree( tp_t<T,range_t,callable>& tp, Ts&... tps );
  
    template< size_t TREE_ID, size_t TREE_DEPTH, typename... Ts, std::enable_if_t<( TREE_DEPTH==0 )>* = nullptr >
    void generate_single_config_tree( const Ts&... values );

    // TODO: loeschen
    template< typename T, typename... Ts >
    void print_path(T val, Ts... tps);

    // TODO: loeschen
    void print_path();


  protected:
    virtual void          initialize( const search_space& search_space) = 0;
    virtual void          finalize()                                    = 0;
    virtual configuration get_next_config()                             = 0;
    virtual void          report_result( const size_t& result )         = 0;

    search_space_tree            _search_space;
  
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
};


} // namespace "atf"

#include "detail/tuner_with_constraints_def.hpp"

#endif /* tuner_with_constraints_h */
