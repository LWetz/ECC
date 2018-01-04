//
//  search_space_simple.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 22/03/2017.
//  Copyright © 2017 Ari Rasch. All rights reserved.
//

#ifndef search_space_simple_h
#define search_space_simple_h

#include <iostream>
#include <assert.h>

#include <fstream>

#include "tp_value_node.hpp"
#include "tp_value.hpp"

namespace atf
{


class search_space_simple : public search_space
{
  public:
    search_space_simple()                                    = default;
    search_space_simple( const search_space_simple&  other ) = default;
    search_space_simple(       search_space_simple&& other ) = default;

  
    template< typename T, typename range_t, typename callable >
    void add_tp( const tp_t<T,range_t,callable>& tp, void* act_elem_ptr )
    {
      add_name( tp.name() );
      _ranges.emplace_back( tp.get_range_ptr() );
      _act_elem_ptrs.emplace_back( act_elem_ptr );
    }

  
    size_t num_configs() const
    {
      size_t num_configs = 1;
      
      for( const auto& range : _ranges )
        num_configs *= range->size();
      
      return num_configs;
    }


    iterator begin() const
    {
      return iterator( this );
    }


    iterator end() const
    {
      return iterator( this->num_configs() );
    }

  
    void add_name( const std::string& name )
    {
      _tp_names.emplace_back( name );
    }
  
  
    configuration operator[]( const size_t& index ) const
    {
      configuration config;
      
      // iterate over ranges bottom up
      for( int i = static_cast<int>( _ranges.size() - 1) ; i >= 0 ; --i )
      {
        size_t num_configs_of_lower_ranges = 1;
        
        // adapt num_configs_of_lower_ranges
        for( size_t j = i + 1 ; j < _ranges.size() ; ++j  )
          num_configs_of_lower_ranges *= _ranges[ j ]->size();

        size_t range_elem_index = ( index / num_configs_of_lower_ranges ) % _ranges[ i ]->size();

        config.emplace( std::piecewise_construct,
                        std::forward_as_tuple( this->name( i )                                                                    ),
                        std::forward_as_tuple( this->name( i ), _ranges[ i ]->operator[]( range_elem_index ), _act_elem_ptrs[ i ] )
                      );
      }
      
      return config;
    }
 
  
    configuration get_configuration( const std::vector<size_t>& indices ) const
    {
      assert( _ranges.size() == indices.size() );
      
      configuration config;
      
      for( size_t i = 0 ; i < indices.size() ; ++i )
      {
        auto value = _ranges[ i ]->operator[]( indices[i] );
        config.emplace( std::piecewise_construct,
                        std::forward_as_tuple( this->name( i )                             ),
                        std::forward_as_tuple( this->name( i ), value, _act_elem_ptrs[ i ] )
                      );
      }

      return config;
    }


    // the number of TPs, i.e. the tree depth
    size_t num_params() const
    {
      assert( _tp_names.size() == _ranges.size() && _act_elem_ptrs.size() );

      return _tp_names.size();
    }


    size_t max_childs( size_t layer ) const
    {
      return _ranges[ layer ]->size();
    }
  

    size_t max_childs_of_node( std::vector<size_t>& indices ) const // TODO: search space sollte selbe interface haben wie tp_value_node
    {
      return _ranges[ indices.size() ]->size();
    }
  
  
    const std::vector< std::string >& names() const
    {
      return _tp_names;
    }


    const std::string& name( size_t i ) const
    {
      return _tp_names[ i ];
    }

  
  
  private:
    std::vector< std::string >  _tp_names;
    std::vector< range const* > _ranges;
    std::vector< void* >        _act_elem_ptrs;
};


// // factory
// template< typename... Ts, typename... range_ts, typename... callables >
// auto search_space_simple( tp_t<Ts,range_ts,callables>&... tps )
// {
//   return search_space_simple_class< tp_t<Ts,range_ts,callables>... >( tps... );
// }


} // namespace "atf"


#endif /* search_space_simple_h */
