//
//  abort_conditions.hpp
//  new_atf_lib
//
//  Created by Ari Rasch on 17/11/2016.
//  Copyright © 2016 Ari Rasch. All rights reserved.
//

#ifndef abort_conditions_h
#define abort_conditions_h


#include <chrono>
#include <vector>
#include <memory>
#include <iostream>

namespace atf
{

class tuner;

namespace cond
{

class abort
{
  public:
    virtual bool stop(const tuner& tuner) = 0;
  
    virtual std::unique_ptr<abort> copy() const = 0;
//    {
//      return *this;
//    };

    virtual ~abort() = 0;
};

class or_class : public abort
{
  public:
    template< typename... Ts >
    or_class( const Ts&... conditions )
    {
      add( conditions... );
    }

  
    bool stop( const tuner& tuner );
  
    virtual std::unique_ptr<abort> copy() const
    {
      return std::unique_ptr<abort>( new or_class( *this ) );
    }
  

  private:
    std::vector<std::shared_ptr<abort>> s;

    // IS
    template< typename T, typename... Ts, std::enable_if_t< std::is_base_of<abort, T>::value >* = nullptr >
    void add( const T& condition, const Ts&... conditions )
    {
      auto res = std::make_shared<T>( condition );
      
      s.push_back( res );
      
      add( conditions... );
    }


    // IA
    void add();
};

template< typename T_lhs, typename T_rhs, std::enable_if_t< std::is_base_of<abort, T_lhs>::value && std::is_base_of<abort, T_rhs>::value >* = nullptr >
or_class operator||( const T_lhs& lhs, const T_rhs& rhs )
{
  return or_class( lhs, rhs );
}

class and_class : public abort
{
  public:
    template< typename... Ts >
    and_class( const Ts&... conditions )
    {
      add( conditions... );
    }

    
    bool stop( const tuner& tuner );

    virtual std::unique_ptr<abort> copy() const
    {
      return std::unique_ptr<abort>( new and_class( *this ) );
    }

  

  private:
    std::vector<std::shared_ptr<abort>> s;

    // IS
    template< typename T, typename... Ts, std::enable_if_t< std::is_base_of<abort, T>::value >* = nullptr >
    void add( const T& condition, const Ts&... conditions )
    {
      auto res = std::make_shared<T>(condition);
      
      s.push_back( res );
      
      add( conditions... );
    }


    // IA
    void add();
};

template< typename T_lhs, typename T_rhs, std::enable_if_t< std::is_base_of<abort, T_lhs>::value && std::is_base_of<abort, T_rhs>::value >* = nullptr >
and_class operator&&( const T_lhs& lhs, const T_rhs& rhs )
{
  return and_class( lhs, rhs );
}



class evaluations : public abort
{
  public:
    evaluations( const size_t& num_evaluations );
  
    bool stop( const tuner& tuner );

    virtual std::unique_ptr<abort> copy() const
    {
      return std::unique_ptr<abort>( new evaluations( *this ) );
    }
  
  
  private:
    size_t _num_evaluations;
};


class valid_evaluations : public abort
{
  public:
    valid_evaluations( const size_t& num_evaluations );
  
    bool stop( const tuner& tuner );

    virtual std::unique_ptr<abort> copy() const
    {
      return std::unique_ptr<abort>( new valid_evaluations( *this ) );
    }
  
  
  private:
    size_t _num_evaluations;
};


class speedup : public abort
{
    enum DurationType
    {
      NUM_CONFIGS,
      TIME
    };
  
  public:
    speedup( const double& speedup, const size_t& num_configs = 1            , const bool& only_valid_configs = true );
    speedup( const double& speedup, const std::chrono::milliseconds& duration, const bool& only_valid_configs = true );
  
    ~speedup()
    {
//      static size_t pos = 0;
//      for( auto& elem : _verbose_history )
//        std::cout << pos++ << ": " << elem << std::endl;
    }
  
    bool stop( const tuner& tuner );
  
    virtual std::unique_ptr<abort> copy() const
    {
      return std::unique_ptr<abort>( new speedup( *this ) );
    }
  
  private:
    double                    _speedup;
    std::chrono::milliseconds _duration;
    size_t                    _num_configs;
    DurationType              _type;
    std::vector<size_t>       _verbose_history;
    bool                      _only_valid_configs;
};


template< typename duration_t >
class duration : public abort
{
  public:
    duration( size_t duration )
      : _duration( duration )
    {}
  
  
    bool stop( const tuner& tuner );
  
    virtual std::unique_ptr<abort> copy() const
    {
      return std::unique_ptr<abort>( new duration( *this ) );
    }

//    {
//      auto current_tuning_time = std::chrono::high_resolution_clock::now() - tuner.tuning_start_time();
//      
//    //  #ifdef VERBOSE
//    //  std::cout << "current tuning duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(current_tuning_duration).count() << std::endl;
//    //  #endif
//      
//      return current_tuning_time > _duration;
//    }
  
  private:
    duration_t _duration;
};


class result : public abort
{
  public:
    result( size_t result );
  
    bool stop( const tuner& tuner );
  
    virtual std::unique_ptr<abort> copy() const
    {
      return std::unique_ptr<abort>( new result( *this ) );
    }

  private:
    size_t _result;
};


//class no_changes : public abort
//{
//  public:
//    no_changes( size_t num_configs );
//  
//    bool stop( const tuner& tuner );
//
//  private:
//    size_t              _num_configs;
//    std::vector<size_t> _verbose_history;
//};


} // namespace "cond"

} // namespace "atf"


#endif /* abort_conditions_h */
