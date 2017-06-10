//
//  tp_value_node.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 20/11/2016.
//  Copyright © 2016 Ari Rasch. All rights reserved.
//

#ifndef tp_value_node_h
#define tp_value_node_h

#include <vector>
#include <assert.h>
#include <memory>

#include "value_type.hpp"

namespace atf
{


//TODO in .hpp und .cpp splitten
class tp_value_node
{
  public:
    tp_value_node()
      : _value(), _parent( nullptr ), _childs(), _depth( 0 )
    {
      ++__number_tree_nodes;
    }
  

    template< typename T >
    tp_value_node( const T& value, void* tp_value_ptr, tp_value_node* parent )
      : _value( value ), _tp_value_ptr( tp_value_ptr ), _parent( parent ), _childs(), _depth( 0 )
    {
      ++__number_tree_nodes;
    }


    tp_value_node( const tp_value_node& other ) = delete;
//    {
//      assert( false );
//    }


    tp_value_node( const tp_value_node&& other ) = delete;
//    {
//      assert( false );
//    }
  

#if 0
    tp_value_node( const tp_value_node& other )
    {
      this->_value  = other._value;
      this->_childs = other._childs;
      this->_depth  = other._depth;
      
      set_parents_of_childs_recursively();
    }

  
    void set_parents_of_childs_recursively()
    {
      for( auto& child : _childs )
      {
        child._parent = this;
        child.set_parents_of_childs_recursively();
      }
    }
#endif

    auto value() const
    {
      return _value;
    }

    auto tp_value_ptr() const
    {
      return _tp_value_ptr;
    }

  

#if 0
    auto childs() const
    {
      return _childs;
    }
#endif  

    auto num_childs() const
    {
      return _childs.size();
    }
  
  
    const auto& child( size_t i ) const
    {
      assert( i < _childs.size() );
      
      return *_childs[ i ];
    }
  
  
    const auto& child( const std::vector<size_t>& indices ) const
    {
      const tp_value_node* res = this;
      
      for( const auto& index : indices )
        res = &res->child( index );
      
      return *res;
    }
  
    const tp_value_node& parent() const
    {
      return *_parent;
    }


#if 0
    //TODO: noetig?
    auto operator[]( size_t i ) const
    {
      return child( i );
    }
#endif  
  
    size_t num_params() const
    {
      return _depth;
    }
  
  
    // return value is leaf node corresponding to the inserted path
    template< typename T, typename... T_rest >
    const tp_value_node& insert( T fst, T_rest... rest )
    {
      auto value        = std::get<0>( fst );
      auto tp_value_ptr = std::get<1>( fst ); //TODO: refac "tp_act_elem_ptr" !!!
      
      const size_t num_params = 1 + sizeof...( rest );
      if( _depth < num_params )
        _depth = num_params;
      
      if( _childs.size() == 0 || value != static_cast< decltype(value) >( _childs.back()->value() ) ) //TODO: cast nötig?
      {
        auto parent = this;
        _childs.emplace_back( std::make_unique<tp_value_node>( value, tp_value_ptr, parent ) );
        return _childs.back()->insert( rest... );
      }
      
      else
        return _childs.back()->insert( rest... );
    }
  
  
//    size_t num_leafs() const
//    {
//      size_t num_leafs = 0;
//      
//      this->num_leafs_hlpr( num_leafs );
//      
//      return num_leafs;
//    }
//  
//  
//    const tp_value_node& leaf( size_t i ) const
//    {
//      assert( _childs.size() > i && "tree has too few leafs" );
//    
//      size_t               count = 0;
////      tp_value_node const* res;
//      
////      this->leaf_hlpr( i, count, res );
//      
//      return this->leaf_hlpr( i, count ); // *res;
//    }
//    
//  
//  
//    const tp_value_node& leaf_hlpr( const size_t& i, size_t& count ) const //, tp_value_node const* res ) const
//    {
//      assert( _childs.size() > 0 );
//      
//      for( const auto& child : _childs )
//        if( child.num_childs() == 0 )
//        {
//          if( count++ == i )
//            return child;
//        }
//        else
//          return child.leaf_hlpr( i, count, res );
//    }
  
    // returns corresponding leaf node
    const tp_value_node& insert()
    {
      return *this;
//      tp_value_node const* tmp_node = this;
//      while( tmp_node->num_childs() != 0 )
//        tmp_node = &tmp_node->parent();//&tmp_node->childs().back();
//      
//      return *tmp_node;
    }
  
  
    void print() const
    {
//      std::cout << _value << " ";
//      for( const auto& child : _childs )
//        child.print();
//
//      if( _childs.size() == 0 )
//        std::cout << std::endl;
    }
  
  
    // maximal number of childs for a node in the layer "layer". root has layer "0".
    size_t max_childs( size_t layer ) const
    {
      if( layer == 0 )
        return num_childs();
      
      size_t max_childs = 0;
      for( const auto& child : this->_childs )
        if( child->max_childs( layer - 1 ) > max_childs )
          max_childs = child->max_childs( layer - 1 );
      
      return max_childs;
    }
  
  
  protected: //TODO: warum protected?
    value_type                                     _value;
    void*                                          _tp_value_ptr;
    tp_value_node*                                 _parent;
    std::vector< std::unique_ptr<tp_value_node> >  _childs;
//    std::vector<tp_value_node*> _leafs;
    size_t                                         _depth;

    size_t index() const; //TODO: was ist das?


// static member
  public:
    static size_t number_of_nodes()
    {
      return __number_tree_nodes;
    }
  
  private:
    static size_t __number_tree_nodes;
  
};

} // namespace "atf"

#endif /* tp_value_node_h */
