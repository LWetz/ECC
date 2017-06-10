//
// Created by richard on 11.04.17.
//
#include <iostream>
#include "../include/op_wrapper.hpp"

namespace atf {
    template<>
    void print_type<int>() {
        std::cout << "int" << std::endl;
    }

    template<>
    void print_type<size_t>() {
        std::cout << "size_t" << std::endl;
    }


    template<>
    void print_type<bool>() {
        std::cout << "bool" << std::endl;
    }
}