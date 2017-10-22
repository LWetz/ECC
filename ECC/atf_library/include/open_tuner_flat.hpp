//
//  open_tuner_flat.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 17/01/2017.
//  Copyright © 2017 Ari Rasch. All rights reserved.
//

#ifndef open_tuner_flat_h
#define open_tuner_flat_h

#include <cstdlib>

#include "tuner_with_constraints.hpp"
#include "tuner_without_constraints.hpp"

namespace atf
{

template< typename T = tuner_with_constraints>
class open_tuner_flat_class : public T
{
  public:
    template< typename... Ts >
    open_tuner_flat_class( Ts... params )
      : T(  params... )
    {}
  
    void set_path_to_database( const std::string& path )
    {
      _path_to_database = path;
    }

  
    void initialize( const search_space& search_space )
    {
      _search_space = &search_space; // unnecessary: _search_space is initialized in parent "tuner" after call of "operator()( G_classes)"
      
      std::string python_code =
                                #include "../src/python_template.py" //TODO: Datei vernünftig einlesen
                              ;

      std::stringstream tp_parameter_code;
      tp_parameter_code << "manipulator.add_parameter(IntegerParameter('" << "TP" << "', 0, " << _search_space->num_configs() - 1 << "))\n";
      
      
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
      // get TP value
      PyObject* p_config = PyObject_CallObject( p_get_next_desired_result, NULL );
      PyObject* pValue   = PyDict_GetItemString( p_config, "TP" );
      size_t    tp_value = PyInt_AsSsize_t( pValue );
      Py_DECREF( p_config );
      
      configuration config = (*_search_space)[ tp_value ];
      
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
    search_space const* _search_space;
    std::string         _path_to_database;

    PyObject* p_get_next_desired_result;
    PyObject* p_report_result;
    PyObject* p_finish;
};

template< typename... Ts >
auto open_tuner_flat( Ts... args )
{
  return open_tuner_flat_class<>{ args... };
}

template< typename T, typename... Ts >
auto open_tuner_flat( Ts... args )
{
  return open_tuner_flat_class<T>{ args... };
}

} // namespace "atf"



#endif /* open_tuner_flat_h */
