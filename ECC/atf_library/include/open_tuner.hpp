//
//  open_tuner.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 19/11/2016.
//  Copyright © 2016 Ari Rasch. All rights reserved.
//

#ifndef open_tuner_h
#define open_tuner_h

#ifdef __APPLE__
  #include <Python/Python.h>
#else
  #include <Python.h>
#endif

#include <cstring>
#include <algorithm>
#include <cstdlib>

#include "tuner_with_constraints.hpp"
#include "tuner_without_constraints.hpp"

namespace atf
{

template< typename T = tuner_with_constraints>
class open_tuner_class : public T
{
  public:
    template< typename... Ts >
    open_tuner_class( Ts... params )
      : T(  params... )
    {}
  
    void set_path_to_database( const std::string& path )
    {
      _path_to_database = path;
    }

  
    void initialize( const search_space& search_space )
    {
      //_search_space = &search_space; // unnecessary: _search_space is initialized in parent "tuner" after call of "operator()( G_classes)"
      
      std::string python_code =
                                #include "../src/python_template.py" //TODO: Datei vernünftig einlesen
                              ;

      std::stringstream tp_parameter_code;
      for( size_t i = 0 ; i < this->_search_space.num_params() ; ++ i )
        tp_parameter_code << "manipulator.add_parameter(IntegerParameter('" << this->_search_space.name( i ) << "', 0, " << this->_search_space.max_childs( i ) - 1 << "))\n";
      
      
      size_t start_pos = python_code.find(":::parameters:::");
      python_code.replace( start_pos, strlen(":::parameters:::"), tp_parameter_code.str() );

//      Py_SetProgramName( argv[0] ); // optional but recommended

      static bool first_time = true;
      if( first_time )
      {
        Py_Initialize();
        first_time = false;
      }
      
      std::vector< std::string > opentuner_cmd_line_arguments;
      
      opentuner_cmd_line_arguments.push_back( "python_template.py" ); // python program name
      opentuner_cmd_line_arguments.push_back( "--no-dups"          ); // supresses printing warnings for duplicate requests'
      
      if ( !_path_to_database.empty() )
      {
        opentuner_cmd_line_arguments.push_back( "--database"      );   // path to OpenTuner database
        opentuner_cmd_line_arguments.push_back( _path_to_database );
      }
      
      // convert vector of std::string to vector of char*
      std::vector< char* > opentuner_cmd_line_arguments_as_c_str;
      auto str_to_char = []( const std::string& str ){ char* c_str = new char[ str.size() + 1 ];
                                                       std::strcpy( c_str, str.c_str() );
                                                       return c_str;
                                                     };
      std::transform( opentuner_cmd_line_arguments.begin(), opentuner_cmd_line_arguments.end(), std::back_inserter( opentuner_cmd_line_arguments_as_c_str ), str_to_char );
      
      // set command line arguments
      PySys_SetArgv( static_cast<int>( opentuner_cmd_line_arguments_as_c_str.size() ), opentuner_cmd_line_arguments_as_c_str.data() );
      
      // let code run by the python interpreter in the "__main__" module
      auto error = PyRun_SimpleString( python_code.c_str() ); if( error == -1 ) std::cout << "error: running python script fails" << std::endl;
      
      auto p_module             = PyImport_ImportModule( "__main__" );
      
      p_get_next_desired_result = PyObject_GetAttrString( p_module, "get_next_desired_result");
      p_report_result           = PyObject_GetAttrString( p_module, "report_result"          );
      p_finish                  = PyObject_GetAttrString( p_module, "finish"                 );
    }
  
  
    configuration get_next_config()
    {
      // get indices
      PyObject* p_config = PyObject_CallObject( p_get_next_desired_result, NULL );

      std::vector< size_t > indices;
      for( const auto& name : this->_search_space.names() )
      {
        PyObject* pValue = PyDict_GetItemString( p_config, name.c_str() );
        indices.push_back( PyInt_AsSsize_t( pValue ) );
      }

      Py_DECREF( p_config );
      
      // adapt indices to the actual number of childs of the considered sub-tree
      for( size_t i = 0 ; i < indices.size() ; ++i )
      {
        auto shrinked_indices = indices;
        shrinked_indices.erase( shrinked_indices.begin() + i, shrinked_indices.begin() + shrinked_indices.size() ); // erease element with an index > i
        
        indices[ i ] = indices[ i ] % this->_search_space.max_childs_of_node( shrinked_indices );
      }
      configuration config = this->_search_space.get_configuration( indices );
      
      return config;
    }
  
    
    void report_result( const size_t& result )
    {
      PyObject* arg = PyTuple_New( 1 );
      PyTuple_SetItem( arg, 0, PyFloat_FromDouble( result ) ); //TODO: PyFloat_FromDouble richtig?
      
      PyObject_CallObject( p_report_result, arg );
      
      Py_DECREF(arg);
    }
  
  
    void finalize()
    {
      PyObject_CallObject( p_finish, NULL );

      Py_XDECREF( p_get_next_desired_result );
      Py_XDECREF( p_report_result           );
      Py_XDECREF( p_finish                  );
      
      static bool first_time = true;
      if( first_time )
      {
        std::atexit( Py_Finalize );
        first_time = false;
      }
      
    }

  
  private:
    std::string _path_to_database;

    PyObject* p_get_next_desired_result;
    PyObject* p_report_result;
    PyObject* p_finish;
};


template< typename... Ts >
auto open_tuner( Ts... args )
{
  return open_tuner_class<>{ args... };
}


template< typename T, typename... Ts >
auto open_tuner( Ts... args )
{
  return open_tuner_class<T>{ args... };
}


} // namespace "atf"

#endif /* open_tuner_h */
