//
// Created by richard on 11.04.17.
//
#include "../include/ocl_wrapper.hpp"

namespace atf {

    namespace cf {

        void check_error(cl_int err) {
            if (err != CL_SUCCESS) {
                printf("Error with errorcode: %d\n", err);
//    throw std::exception();
//    exit(1);
            }
        }
    }
}