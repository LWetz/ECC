//
// Created by Richard Schulze on 25.04.2017.
//

#ifndef MD_HOM_OCL_MD_HOM_WRAPPER_CLASS_HPP
#define MD_HOM_OCL_MD_HOM_WRAPPER_CLASS_HPP

#include "../atf.h"

template<typename GS_0, typename GS_1, typename GS_2,
         typename LS_0, typename LS_1, typename LS_2,
         typename... Ts>
class ocl_md_hom_wrapper_class : public atf::cf::ocl_cf_class<GS_0, GS_1, GS_2, LS_0, LS_1, LS_2, Ts...> {
private:
    typedef atf::cf::ocl_cf_class<GS_0, GS_1, GS_2, LS_0, LS_1, LS_2, Ts...> super;
    std::vector<std::string> _wg_r_tp_names;
    size_t _last_buffer_elem_size;

public:
    ocl_md_hom_wrapper_class(const atf::cf::device_info &device, const atf::cf::kernel_info &kernel,
                             const std::tuple<Ts...> &kernel_inputs, const std::tuple<GS_0, GS_1, GS_2> &global_size,
                             const std::tuple<LS_0, LS_1, LS_2> &local_size, std::vector<std::string> wg_r_tp_names) :
            atf::cf::ocl_cf_class<GS_0, GS_1, GS_2, LS_0, LS_1, LS_2, Ts...>(device, kernel, kernel_inputs, global_size, local_size),
            _wg_r_tp_names(wg_r_tp_names) {
        super::kernel_buffers().back().getInfo(CL_MEM_SIZE, &_last_buffer_elem_size);
        _last_buffer_elem_size /= super::kernel_input_sizes().back();
    }

    size_t operator()( atf::configuration& configuration ) {
        // adapt size of res_glb
        // calculate new size
        size_t num_wg_r = 1;
        for (auto& name : _wg_r_tp_names) {
            num_wg_r *= configuration[name].value().size_t_val();
        }
        size_t new_size = super::kernel_input_sizes().back() * num_wg_r * _last_buffer_elem_size;
        // replace buffer
        super::kernel_buffers().back() = cl::Buffer(super::context(), CL_MEM_READ_WRITE, new_size);

        return atf::cf::ocl_cf_class<GS_0, GS_1, GS_2, LS_0, LS_1, LS_2, Ts...>::operator()(configuration);
    }
};

template<typename GS_0, typename GS_1, typename GS_2,
         typename LS_0, typename LS_1, typename LS_2,
         typename... Ts>
auto ocl_md_hom(const atf::cf::device_info&       device,
                const atf::cf::kernel_info&       kernel,
                const std::tuple<Ts...>& kernel_inputs,

                std::tuple< GS_0, GS_1, GS_2 > global_size,
                std::tuple< LS_0, LS_1, LS_2 > local_size,

                std::vector<std::string> wg_r_tp_names = {}) {
    return ocl_md_hom_wrapper_class<GS_0, GS_1, GS_2,
                                    LS_0, LS_1, LS_2,
                                    Ts...>
            (device, kernel, kernel_inputs, global_size, local_size, wg_r_tp_names);
}


#endif //MD_HOM_OCL_MD_HOM_WRAPPER_CLASS_HPP
