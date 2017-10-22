//
//  search_space_tree.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 22/03/2017.
//  Copyright © 2017 Ari Rasch. All rights reserved.
//

#ifndef search_space_tree_h
#define search_space_tree_h

#include <iostream>
#include <assert.h>

#include <fstream>

#include "tp_value_node.hpp"
#include "tp_value.hpp"

namespace atf
{

//TODO: in own file
class Tree
{
  public:
    // ctors
    Tree()
      : _root( std::make_unique<tp_value_node>() ), _leafs()
    {}
    
    Tree(       Tree&& other ) = default;
    Tree( const Tree&  other ) = default;
  

    size_t num_configs() const
    {
      return _leafs.size();
    }
  
  
    size_t depth() const
    {
      return _root->num_params();
    }
  
    template< typename... Ts >
    void insert( Ts... params )
    {
      //_root->insert( params... );
      const tp_value_node& leaf = _root->insert( params... );
      _leafs.emplace_back( &leaf );
    }


    const tp_value_node& root() const
    {
      return *_root;
    }

  
    const tp_value_node& leaf( size_t i ) const
    {
      return *_leafs[ i ];
    }
  
  
    // forwards to tp_value node
  
    template< typename... Ts >
    const auto& child( Ts... params ) const
    {
      return _root->child( params... );
    }


    template< typename... Ts >
    size_t max_childs( Ts... params ) const
    {
      return _root->max_childs( params... );
    }


    template< typename... Ts >
    size_t num_params( Ts... params ) const
    {
      return _root->num_params( params... );
    }

  
    // cast
    operator tp_value_node&()
    {
      return *_root;
    }
    

  private:
    std::shared_ptr< tp_value_node >    _root;
    std::vector< tp_value_node const* > _leafs;
};



class search_space_tree : public search_space
{
  public:
    search_space_tree()
      : _trees(), _tree_sizes(), _tp_names()
    {}
  
    search_space_tree( const search_space_tree&  other ) = default;
    search_space_tree(       search_space_tree&& other ) = default;
  
    size_t num_configs() const
    {
      size_t num_configs = 1;
      
      for( const auto& tree : _trees )
        num_configs *= tree.num_configs();
      
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
  
  
    void append_new_trees( size_t num ) // as "friend"
    {
      auto old_size = _trees.size();
      _trees.resize( old_size + num );
      _tree_sizes.emplace_back( 0 );
    }
  
    template< typename... Ts >
    void insert_in_last_tree( Ts... values )  // as "friend"
    {
      _trees.back().insert( values... );
      ++_tree_sizes.back();
    }
  
    Tree& tree( size_t tree_id )  // as "friend"
    {
      return _trees[ tree_id ];
    }
  
    void add_name( const std::string& name )
    {
      _tp_names.emplace_back( name );
    }
  

//    tp_value_node get_child( const std::vector<size_t>& indices ) const
//    {
//      assert( indices.size() <= this->num_params() );
//
////      const auto num_params = this->num_params();
//      
//      size_t i_global = 0;
//      for( const auto& tree : _trees ) //TODO refac: "_trees" -> "_sub_trees" o.ä.
//      {
//        tp_value_node node;
//        for( size_t i = 0 ; i < tree.num_params() ; ++i, ++i_global )
//          tree_node = tree.child( indices[ i_global ] );
//      }
//
//      return node;
//    }
  
  
    configuration operator[]( const size_t& index ) const
    {
      auto config = configuration{};//( this->num_params() );
      
      size_t pos = this->num_params() - 1;
      
      // iterate over trees (bottom-up)
      for( int tree_id = static_cast<int>( _trees.size() ) - 1 ; tree_id >= 0 ; --tree_id )
      {
        const auto& tree = _trees[ tree_id ];
        
        // select leaf
        size_t num_configs_of_lower_trees = 1;
        for( int lower_tree = tree_id + 1; lower_tree < _trees.size() ; ++lower_tree )
          num_configs_of_lower_trees *= _trees[ lower_tree ].num_configs();
        
        //assert( index % num_configs_of_lower_trees == 0 || index < num_configs_of_lower_trees );
        auto leaf_id = ( index / num_configs_of_lower_trees ) % tree.num_configs(); //TODO: wrong for #trees > 1
        
        // go leaf up and insert TP values in config
        tp_value_node const* tree_node = &tree.leaf( leaf_id );
        for( size_t i = 0 ; i < tree.num_params() ; ++i )
        {
//          config[ pos ] = tp_value( this->name( pos ), tree_node->value() );
//          config.emplace( config.begin(), this->name( name_pos-- ), tree_node->value() );
//          auto tmp = tp_value( this->name( pos ), tree_node->value(), tree_node->tp_value_ptr() );
//          config[ this->name( pos ) ] = tmp;
          config.emplace( std::piecewise_construct,
                          std::forward_as_tuple( this->name(pos)                                                ),
                          std::forward_as_tuple( this->name(pos), tree_node->value(), tree_node->tp_value_ptr() )
                        );
          --pos;
//          std::cout << "tree_node->value(): " << static_cast<int>( tree_node->value() ) << std::endl;
          tree_node = &( tree_node->parent() );
          //assert( tree_node != nullptr );
        }
      }
      
      return config;
    }
  
  
    configuration get_configuration( const std::vector<size_t>& indices ) const
    {
      assert( indices.size()   == this->num_params() );
      assert( _tp_names.size() == this->num_params() );

//      const auto num_params = this->num_params();
      
      configuration config;//( num_params );
//      config.reserve( num_params );
      
      size_t i_global = 0;                      // TODO: Zugriffe auf Baum-Ebenen von "int" auf "size_t" umstellen
      for( const auto& tree : _trees )
      {
        const tp_value_node* tree_node = &tree.root();
        for( size_t i = 0 ; i < tree.num_params() ; ++i, ++i_global )
        {
          tree_node = &( tree_node->child( indices[ i_global ] ) );
//          config.emplace_back( this->name(i_global), tree_node->value(), tree_node->tp_value_ptr() );
          config.emplace( std::piecewise_construct,
                          std::forward_as_tuple( this->name(i_global)                                               ),
                          std::forward_as_tuple(this->name(i_global), tree_node->value(), tree_node->tp_value_ptr() )
                        );
//          config[ i_global ].name()  = this->name( i_global );
//          config[ i_global ].value() = tree_node.value();
        }
      }

      assert( i_global == config.size() ); // TODO: loeschen
      return config;
    }
  
  
//    std::vector<configuration> get_all_configurations() const
//    {
//      std::vector<configuration> result;
//      result.reserve( this->num_params() );
//      
//      std::vector<size_t> indices;
//      indices.reserve( this->num_params() );
//      
////      for( const auto& tree : _trees )
////        get_all_configurations_impl( tree, indices, result );
//
//      get_all_configurations_impl( 0, _trees[0], indices, result );
//
//      
//      return result;
//    }
//
//
//    void get_all_configurations_impl( const size_t& tree_id, const tp_value_node& tree_node, std::vector<size_t>& indices, std::vector<configuration>& result ) const
//    {
//      assert( _tp_names.size() == this->num_params() );
//      
//      // case "tree_node is a leaf node": insert configuration induced by "indices" into "result"
//      if( indices.size() == this->num_params() )
//      {
//        auto config = this->get_configuration( indices );
//        result.emplace_back( config );
////        indices.erase( indices.begin(), indices.end() );
////        indices.reserve( this->num_params() );
//        return;
//      }
//
//      // case "tree_node is a leaf node" and last tree is not reached
//      if( tree_node.num_childs() == 0 )
//      {
//        get_all_configurations_impl( tree_id + 1, _trees[ tree_id + 1  ], indices, result );
//        return;
//      }
//
//      indices.emplace_back();
//      for( size_t i = 0 ; i < tree_node.num_childs() ; ++i )
//      {
//        indices.back() = i;
//        this->get_all_configurations_impl( tree_id, tree_node.child( i ), indices, result );
//      }
//      indices.pop_back();
//    }
  
  
//    size_t get_num_configurations() const
//    {
//      size_t result = 1;
//      
//      std::vector<size_t> indices;
//      indices.reserve( this->num_params() );
//      
//      for( const auto& tree : _trees )
//        result *= get_num_configurations_for_tree( tree );
//
//      return result;
//    }
  
//    size_t get_num_configurations_for_tree( const tp_value_node& tree ) const
//    {
//      assert( _tp_names.size() == this->num_params() );
//      
//      // IA
//      if( tree.num_childs() == 0 )
//        return 1;
//
//      // IS
//      size_t num_leafs = 0;
//
//      for( size_t i = 0 ; i < tree.num_childs() ; ++i )
//        num_leafs += this->get_num_configurations_for_tree( tree.child( i ) );
//      
//      return num_leafs;
//    }
  
    // the number of TPs, i.e. the tree depth
    size_t num_params() const
    {
      size_t num_params_of_all_trees = 0;
      
      for( const auto& tree : _trees )
        num_params_of_all_trees += tree.num_params();
      
      return num_params_of_all_trees;
    }


    size_t max_childs( size_t layer ) const
    {
      assert( layer < this->num_params() );
      
      for( const auto& tree : _trees )
      {
        if( layer < tree.num_params() )
          return tree.max_childs( layer );
        else
          layer -= tree.num_params(); // go to the next config_tree generated by "G(...)"
      }
  
      assert( false ); // should never be reached
      
      return 0;
    }
  

    size_t max_childs_of_node( std::vector<size_t>& indices ) const // TODO: search space sollte selbe interface haben wie tp_value_node
    {
      auto lhs = indices.size(); lhs=lhs;
      auto rhs = this->num_params(); rhs=rhs;
      assert( indices.size() < this->num_params() ); // "<=" makes no sense -- this would access a leaf node

      size_t tree_index = 0;
      
      while( indices.size() >= _trees[ tree_index ].num_params() )
        indices.erase( indices.begin(), indices.begin() + _trees[ tree_index++ ].num_params() );

      return _trees[ tree_index ].child( indices ).num_childs();
    }
  
  
    const std::vector< std::string >& names() const
    {
      return _tp_names;
    }


    const std::string& name( size_t i ) const
    {
      return _tp_names[ i ];
    }

  
    size_t num_trees() const // as "friend"
    {
      return _trees.size();
    }

    const std::vector<Tree>& trees() const  // as "friend"; TODO: nötig?
    {
      return _trees;
    }
  
  
    // TODO: loeschen
    ~search_space_tree()
    {

      std::cout << "search space size: " << this->num_configs() << std::endl;
#if 0
      for( size_t i = 0 ; i < this->num_configs() ; ++i )
      {
        auto config = this->operator[]( i );
        
        static size_t n = 0;
        std::cout << n++ << ":\n";
        for( const auto& tp : config )
          std::cout << tp.name() << " = " << tp.value() << std::endl;
        
        std::cout << "\n";
      }
#endif

//      for( size_t i = 0 ; i < this->num_configs() ; ++i )
//      {
//        auto config = this->operator[]( i );
//        for( const auto& tp : config )
//          std::cout << tp.name() << " = " << tp.value() << std::endl;
//      }
      
      
      
      
//      auto& search_space = *this;
//      auto config = search_space[ 0 ];
//      auto tp     = config[ 0 ];
//      
//      std::cout << tp.name() << " = " << tp.value() << std::endl;

      
      
//      auto search_space = this->get_all_configurations();
//      
//      std::ofstream outfile;
//      outfile.open("/Users/arirasch/log_atf.txt", std::ofstream::app ); // TODO: "/Users/arirasch/results.txt"
//      for( const auto& config : search_space )
//      {
//        for( const auto& tp : config )
//          if( !(tp.name().find( "GS_" ) == 0 && tp.name().size() == 4) && !(tp.name().find( "LS_" ) == 0  && tp.name().size() == 4) )
//            outfile << tp.name() << " = " << tp.value() << std::endl;
//        
//        outfile << std::endl;
//      }
//      
//      outfile.close();
    }
  
  private:
    std::vector< Tree >        _trees;
    std::vector< size_t >      _tree_sizes;
    std::vector< std::string > _tp_names;
};


} // namespace "atf"


#endif /* search_space_tree_h */
