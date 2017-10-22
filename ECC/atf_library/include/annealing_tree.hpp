//
//  annealing_tree.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 20/03/2017.
//  Copyright © 2017 Ari Rasch. All rights reserved.
//

#ifndef annealing_tree_h
#define annealing_tree_h


#include <cstring>
#include <algorithm>
#include <set>

#include "tuner_with_constraints.hpp"
#include "tuner_without_constraints.hpp"

#include "helper.hpp"


#define NEIGHBOUR_RANGE               5
#define k_MAX_ALREADY_VISITED_STATES  10 // 10 is the default k_MAX_ALREADY_VISITED_STATES in CLTune


namespace atf
{

class annealing_1d
{
  public:
    annealing_1d( size_t range )
      : _range( range ), _number_of_evaluated_configs( 0 ), _execution_times( _range, std::numeric_limits<double>::max() ), _act_index(), _generator( std::default_random_engine( random_seed() ) ), _int_distribution( std::uniform_int_distribution<int>( 0, static_cast<int>( _range ) ) ), _probability_distribution( std::uniform_real_distribution<double>( 0.0, 1.0 ) ), _current_state( static_cast<size_t>( _int_distribution(_generator) ) ), _neighbour_state( 0 ), _num_already_visited_states( 0 )
    {}
  
    annealing_1d( const annealing_1d&  other ) = default;
    annealing_1d(       annealing_1d&& other ) = default;
  
    size_t get_next_index()
    {
      // Computes the new temperature
      const auto configs_to_visit = std::max( size_t{1}, static_cast<size_t>( _range * FRACTION ) ); //TODO: _range has to be replaced by search_space size !!
      const auto progress         = _number_of_evaluated_configs / static_cast<double>( configs_to_visit );
      const auto temperature      = double{MAX_TEMPERATURE} * std::max( std::numeric_limits<double>::min(),  1.0 - progress ); // TODO: etwas besser überlegen um die Temperatur zu senken
      
      // init
      static bool flag = true;
      if( flag == true )
      {
        flag = false;
        
        // choose index value as _current_state
        return _current_state ;
      }

      // Determines whether to continue with the neighbour or with the current ID
      assert( _current_state   <= _execution_times.size() );
      assert( _neighbour_state <= _execution_times.size() );
      
      auto ap = acceptance_probability( _execution_times[ _current_state   ],
                                        _execution_times[ _neighbour_state ],
                                        temperature
                                      );
      
      auto random_probability = _probability_distribution( _generator );
      if( ap >= random_probability )
        _current_state = _neighbour_state;
      
      // Computes the new neighbour state
      auto random_int = _int_distribution(_generator);
      auto sign       = std::pow(-1, random_int);
      auto delta      = sign * (random_int % NEIGHBOUR_RANGE );
      
      _neighbour_state = _current_state + delta;
      _neighbour_state = std::max( size_t{0}  , _neighbour_state );
      _neighbour_state = std::min( _range - 1 , _neighbour_state );
      
      // Checks whether this neighbour was already visited. If so, calculate a new neighbour instead.
      // This continues up to a maximum number, because all neighbours might already be visited. In
      // that case, the algorithm terminates.
      if( _execution_times[ _neighbour_state ] != std::numeric_limits<double>::max() && _num_already_visited_states < k_MAX_ALREADY_VISITED_STATES )
      {
        ++_num_already_visited_states;
        return get_next_index();
      }
      _num_already_visited_states = 0;

      _act_index = _neighbour_state;
      
      assert( _neighbour_state < _range );
      return _neighbour_state;
    }
  

    void report_result( const size_t& result )
    {
      _execution_times[ _act_index ] = result;
    }
  
  
  private:
    size_t                                 _range;
    size_t                                 _number_of_evaluated_configs;
    size_t                                 _act_index;
    sparse_vector<double>                  _execution_times;
    std::default_random_engine             _generator;
    std::uniform_int_distribution<int>     _int_distribution;
    std::uniform_real_distribution<double> _probability_distribution;
  
    size_t _current_state;
    size_t _neighbour_state;
    size_t _num_already_visited_states;
  
  
    // helper
    unsigned int random_seed() const
    {
      return static_cast<unsigned int>( std::chrono::system_clock::now().time_since_epoch().count() );
    }

    double acceptance_probability( const double& current_energy,
                                   const double& neighbour_energy,
                                   const double& temperature
                                 )
    {
      if( neighbour_energy < current_energy )
        return 1.0;
      
      return exp( - (neighbour_energy - current_energy) / temperature );
    }
};



template< typename T = tuner_with_constraints>
class annealing_tree_class : public T
{
  public:
    template< typename... Ts >
    annealing_tree_class( Ts... params )
      : T( params... ), _annealings_1d(), _indices()
    {}
  
    annealing_tree_class( const annealing_tree_class&  other ) = default;
    annealing_tree_class(       annealing_tree_class&& other ) = default;

    void initialize( const search_space& search_space ) // unnecessary: _search_space is initialized in parent "tuner" after call of "operator()( G_classes)"
    {
      size_t num_layers = this->_search_space.num_params();

      // random int generator
      _generator        = std::default_random_engine( random_seed() );
      _int_distribution = std::uniform_int_distribution<int>( 0, static_cast<int>( num_layers-1 ) );
      
      for( size_t i = 0 ; i < num_layers ; ++i )
        _annealings_1d.emplace_back( this->_search_space.max_childs( i ) );
      
      // get initial tree path
      for( int i = 0 ; i < _annealings_1d.size() ; ++i )
      {
        _indices.emplace_back( _annealings_1d[i].get_next_index() );
        
        // adapt indices to the actual number of childs of the considered sub-tree
        auto shrinked_indices = _indices;
        shrinked_indices.erase( shrinked_indices.begin() + i, shrinked_indices.begin() + shrinked_indices.size() ); // erease element with an index > i
        
        _indices[ i ] = _indices[ i ] % this->_search_space.max_childs_of_node( shrinked_indices );
      }

    }
  
  
    configuration get_next_config()
    {
      // get k_MAX_DIFFERENCES random positions for the _indices vector
      std::set<size_t> random_indices;
      while( random_indices.size() < k_MAX_DIFFERENCES )
        random_indices.insert( _int_distribution(_generator) );
      
      // adapt indices to the actual number of childs of the considered sub-tree
      //for( auto& i : random_indices )
      for( int i = 0 ; i < _annealings_1d.size() ; ++i )
      {
        _indices[i] = _annealings_1d[ i ].get_next_index();
        
        auto shrinked_indices = _indices;
        shrinked_indices.erase( shrinked_indices.begin() + i, shrinked_indices.begin() + shrinked_indices.size() ); // erease element with an index > i
        
        _indices[ i ] = _indices[ i ] % this->_search_space.max_childs_of_node( shrinked_indices );
      }
      configuration config = this->_search_space.get_configuration( _indices );
      
      return config;
    }
  
    
    void report_result( const size_t& result )
    {
      for( auto& annealing_search : _annealings_1d )
        annealing_search.report_result( result );
    }
  
  
    void finalize()
    {}

  private:
    std::vector<annealing_1d> _annealings_1d;
    std::vector<size_t>       _indices;

    std::default_random_engine         _generator;
    std::uniform_int_distribution<int> _int_distribution;
  
    // helper
    unsigned int random_seed() const
    {
      return static_cast<unsigned int>( std::chrono::system_clock::now().time_since_epoch().count() );
    }
  
};


// factory
template< typename... Ts >
auto annealing_tree( Ts... args )
{
  return annealing_tree_class<>{ args... };
}


template< typename T, typename... Ts >
auto annealing_tree( Ts... args )
{
  return annealing_tree_class<T>{ args... };
}


} // namespace "atf"


#endif /* annealing_tree_h */
