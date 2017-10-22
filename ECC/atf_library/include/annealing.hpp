//
//  annealing.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 15/01/2017.
//  Copyright © 2017 Ari Rasch. All rights reserved.
//

#ifndef annealing_h
#define annealing_h

#include <random>
#include <map>
#include <limits>
#include <algorithm>

#include "tuner_with_constraints.hpp"
#include "tuner_without_constraints.hpp"

#include "helper.hpp"

#define NUM_THREADS                  1

//#define NEIGHBOURS_FRACTION          1              // for large search spaces
//#define MAX_NEIGHBOURS               300000         // for large search spaces

#define MAX_TEMPERATURE              4              // 4  is the default maximal temperature          in CLTune
#define k_MAX_DIFFERENCES            3              // 3  is the default k_MAX_DIFFERENCES            in CLTune
#define k_MAX_ALREADY_VISITED_STATES 10             // 10 is the default k_MAX_ALREADY_VISITED_STATES in CLTune
#define FRACTION                     (1.0f/2048.0f) // gemm
//#define FRACTION                   (1.0f/64.0f)   // conv

namespace atf
{


template< typename T = tuner_with_constraints>
class annealing_class : public T
{
  public:
    template< typename... Ts >
    annealing_class( Ts... params )
      : T(  params... ), _search_space(), _search_space_size( 0 ), _execution_times() /*, _act_index( static_cast<size_t>( _int_distribution(_generator) ) )*/
    {}

  
    void initialize( const search_space& search_space ) // unnecessary: _search_space is initialized in parent "tuner" after call of "operator()( G_classes)"
    {
      _search_space      = &search_space; //.get_all_configurations();
      _search_space_size = _search_space->num_configs();
      
      _execution_times = sparse_vector<double>( _search_space_size, std::numeric_limits<double>::max() );
    
      _generator                = std::default_random_engine( random_seed() );
      _int_distribution         = std::uniform_int_distribution<int>( 0, static_cast<int>( _search_space_size ) );
      _probability_distribution = std::uniform_real_distribution<double>( 0.0, 1.0 );
      
      _act_index = static_cast<size_t>( _int_distribution(_generator) );
    }
  
  
    configuration get_next_config()
    {
      // state
      static size_t current_state              = _act_index;
      static size_t neighbour_state            = 0;
      static size_t num_already_visited_states = 0;
      
      // Computes the new temperature
      const auto configs_to_visit  = std::max( size_t{1}, static_cast<size_t>( _search_space_size * FRACTION ) ); //TODO: static machen?
      const auto progress          = this->number_of_valid_evaluated_configs() / static_cast<double>( configs_to_visit );
      const auto temperature       = double{MAX_TEMPERATURE} * std::max( std::numeric_limits<double>::min(),  1.0 - progress );  // TODO: etwas besser überlegen um die Temperatur zu senken
      
      // init
      static bool flag = true;
      if( flag == true )
      {
        flag = false;
        
        // choose random configuration as current_state
        return (*_search_space)[ current_state ];
      }

      // Determines whether to continue with the neighbour or with the current ID
      assert( current_state   < _execution_times.size() );
      assert( neighbour_state < _execution_times.size() );
      
      auto ap = acceptance_probability( _execution_times[ current_state   ],
                                        _execution_times[ neighbour_state ],
                                        temperature
                                      );
      
      auto random_probability = _probability_distribution( _generator );
      if( ap >= random_probability )
        current_state = neighbour_state;
      
      // Computes the new neighbour state
      auto neighbours = get_neighbours_of( current_state );
      if( neighbours.size() != 0 )
        neighbour_state = neighbours[ static_cast<size_t>( _int_distribution(_generator) ) % neighbours.size() ];

      // Checks whether this neighbour was already visited. If so, calculate a new neighbour instead.
      // This continues up to a maximum number, because all neighbours might already be visited. In
      // that case, the algorithm terminates.
      if( _execution_times[ neighbour_state ] != std::numeric_limits<double>::max() && num_already_visited_states < k_MAX_ALREADY_VISITED_STATES )
      {
        ++num_already_visited_states;
        return get_next_config();
      }
      num_already_visited_states = 0;

      _act_index = neighbour_state;
      
      return (*_search_space)[ neighbour_state ];
    }
  
    
    void report_result( const size_t& result )
    {
      _execution_times[ _act_index ] = result;
    }
  
  
    void finalize()
    {}

  
  private:
    // Random number generation
    std::default_random_engine             _generator;
    std::uniform_int_distribution<int>     _int_distribution;
    std::uniform_real_distribution<double> _probability_distribution;
    // xxx
    search_space const*    _search_space;
    size_t                 _search_space_size;
    sparse_vector<double>  _execution_times;
    size_t                 _act_index;
  
  
  
    //helper
    unsigned int random_seed() const
    {
      return static_cast<unsigned int>( std::chrono::system_clock::now().time_since_epoch().count() );
    }
  
    std::vector<size_t> get_neighbours_of( const size_t& reference_id )
    {
      auto neighbours = std::array<std::vector<size_t>, NUM_THREADS>{};
      
      // iteration constants
      auto start = std::array<size_t, NUM_THREADS>{};
      auto end   = std::array<size_t, NUM_THREADS>{};
      
      for( size_t i = 0 ; i < NUM_THREADS ; ++i )
      {
        const auto chunk_size = _search_space_size / NUM_THREADS;
        
        start[ i ] = i     * chunk_size;
        end[ i ]   = (i+1) * chunk_size;
      }
      
      auto parallel_neighbour_search = [&]( auto start, auto end, auto neighbours )
                                       {
                                         for( auto i = start ; i < end ; ++i )
                                         {
                                           const auto& configuration = (*_search_space)[ i ];
                                          
                                           // Count the number of different settings for this configuration
                                           auto differences = size_t{0};
                                           auto setting_id  = size_t{0};
                                           for( const auto& tp : configuration )
                                           {
                                             auto tp_name = tp.first;
                                             auto setting = tp.second;
                                             if( setting.value() != (*_search_space)[ reference_id ][ tp_name ].value() )
                                               ++differences;

                                             ++setting_id;
                                           }
                  
                                           // Consider this configuration a neighbour if there is at most a certain amount of differences
                                           if( differences == k_MAX_DIFFERENCES ) // NOTE: "==" requires from the user program to comprise more or equal than k_MAX_DIFFERENCES tuning parameters 
                                             neighbours.get().push_back(i);
                                         }
                                       };
      
      
      // start parallel neighbour search
      std::vector<std::thread> threads;
      for( size_t i = 0 ; i < NUM_THREADS ; ++i )
        threads.emplace_back( parallel_neighbour_search, start[ i ], end[ i ], std::ref( neighbours[ i ] ) );
      for( size_t i = 0 ; i < NUM_THREADS ; ++i )
        threads[i].join();
      
      
      // fuse all neighbours in neighbours[0]
      for( size_t i = 1 ; i < NUM_THREADS; ++i )
        neighbours[0].insert( neighbours[0].end(), neighbours[i].begin(), neighbours[i].end() );
      
      return neighbours[0];
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

template< typename... Ts >
auto annealing( Ts... args )
{
  return annealing_class<>{ args... };
}


template< typename T, typename... Ts >
auto annealing( Ts... args )
{
  return annealing_class<T>{ args... };
}


} // namespace "atf"


#endif /* annealing_h */
