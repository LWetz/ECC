//
//  search_space.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 20/11/2016.
//  Copyright © 2016 Ari Rasch. All rights reserved.
//

#ifndef search_space_h
#define search_space_h

#include <iostream>
#include <assert.h>

#include <fstream>

#include "tp_value_node.hpp"
#include "tp_value.hpp"

namespace atf
{


class search_space
{
  protected:
    class iterator : std::iterator< std::input_iterator_tag, const configuration, size_t, configuration const*, configuration const& >
    {
      public:
        iterator( search_space const* search_space )
          : _search_space( search_space ), _num_configs( search_space->num_configs() ), _act_pos( 0 )
        {}
      

        iterator( size_t end_pos )
          : _search_space( nullptr ), _num_configs( end_pos ), _act_pos( end_pos )
        {}
      
      
        bool operator==( const iterator& other )
        {
          if( this->_search_space == other._search_space && this->_num_configs == other._num_configs && this->_act_pos == other._act_pos )
            return true;
          else
            return false;
        }
      
        bool operator!=( const iterator& other )
        {
          return !( this->operator==( other ) );
        }
      
      
        configuration operator*()
        {
          return (*_search_space)[ _act_pos ];
        }


        // prefix
        iterator& operator++()
        {
          ++_act_pos;
          
          return *this;
        }


        // postfix
        iterator operator++(int)
        {
          auto retval = *this;
          ++( *this );
          
          return retval;
        }
      
      private:
        search_space const*  _search_space;
        size_t               _num_configs;
        size_t               _act_pos;
    };
  
  public:
    virtual size_t num_configs() const = 0;

    virtual iterator begin() const = 0;

    virtual iterator end() const = 0;
  
    virtual void add_name( const std::string& name ) = 0;
  
    virtual configuration operator[]( const size_t& index ) const = 0;

    virtual configuration get_configuration( const std::vector<size_t>& indices ) const = 0;

    virtual size_t num_params() const = 0;

    virtual size_t max_childs( size_t layer ) const = 0;
  
    virtual size_t max_childs_of_node( std::vector<size_t>& indices ) const = 0; // TODO: search space sollte selbe interface haben wie tp_value_node

    virtual const std::vector< std::string >& names() const = 0;

    virtual const std::string& name( size_t i ) const = 0;
};


} // namespace "atf"

#endif /* search_space_h */
