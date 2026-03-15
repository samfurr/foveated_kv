#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mlx/mlx.h"
#include "foveated_attn.h"

namespace nb = nanobind;
using namespace mlx::core;

NB_MODULE(foveated_ext, m) {
    m.doc() = "C++ extension for foveated attention on Apple Silicon";

    m.def(
        "foveated_attention",
        &foveated::foveated_attention,
        nb::arg("foveal_k"), nb::arg("foveal_v"),
        nb::arg("periph_k"), nb::arg("periph_v"),
        nb::arg("periph_k_scale"), nb::arg("periph_k_zero"),
        nb::arg("periph_v_scale"), nb::arg("periph_v_zero"),
        nb::arg("far_k"), nb::arg("far_v"),
        nb::arg("far_k_scale"), nb::arg("far_k_zero"),
        nb::arg("far_v_scale"), nb::arg("far_v_zero"),
        nb::arg("foveal_valid"),
        nb::arg("query"), nb::arg("decode_k"), nb::arg("decode_v"),
        nb::arg("override_k"), nb::arg("override_v"),
        nb::arg("override_far_idx"), nb::arg("override_count"),
        nb::arg("spike_margin") = 0.5f,
        nb::arg("split_size") = 256,
        "Fused foveated attention via Split-K Metal kernel (C++ dispatch).");

    // Debug: can we create and return an mlx array?
    m.def("test_array", []() -> array {
        return zeros({2, 3}, float32);
    });

    m.def("test_accept", [](const array& a) -> int {
        return a.size();
    });

    nb::class_<foveated::FoveatedHandle>(m, "FoveatedHandle")
        .def(nb::init<
            const array&, const array&, const array&, const array&,
            const array&, const array&, const array&, const array&,
            const array&, const array&, const array&, const array&,
            const array&, const array&, const array&,
            float, int>(),
            nb::arg("foveal_k"), nb::arg("foveal_v"),
            nb::arg("periph_k"), nb::arg("periph_v"),
            nb::arg("periph_k_scale"), nb::arg("periph_k_zero"),
            nb::arg("periph_v_scale"), nb::arg("periph_v_zero"),
            nb::arg("far_k"), nb::arg("far_v"),
            nb::arg("far_k_scale"), nb::arg("far_k_zero"),
            nb::arg("far_v_scale"), nb::arg("far_v_zero"),
            nb::arg("foveal_valid"),
            nb::arg("spike_margin") = 0.5f,
            nb::arg("max_ov") = 32,
            "Create a handle with pre-bound static tier arrays.")
        .def("__call__",
            &foveated::FoveatedHandle::operator(),
            nb::arg("query"),
            nb::arg("decode_k"), nb::arg("decode_v"),
            nb::arg("override_k"), nb::arg("override_v"),
            nb::arg("override_far_idx"), nb::arg("override_count"),
            "Dispatch with dynamic inputs only (7 arrays).");

    m.def("is_available", []() -> bool {
        try {
            auto q = zeros({1, 1, 1, 64}, float16);
            auto fk = zeros({1, 1, 1, 64}, float16);
            auto fv = zeros({1, 1, 1, 64}, float16);
            auto pk = zeros({1, 1, 1, 64}, uint8);
            auto pv = zeros({1, 1, 1, 64}, uint8);
            auto ps = zeros({1, 1, 64}, float16);
            auto pvs = zeros({1, 1, 1}, float16);
            auto fkk = zeros({1, 1, 1, 64}, uint8);
            auto fvv = zeros({1, 1, 1, 32}, uint8);
            auto fks = zeros({1, 1, 64}, float16);
            auto fvs = zeros({1, 1, 1}, float16);
            auto fval = full({1}, 1, uint32);
            auto ovk = zeros({1, 32, 64}, float16);
            auto ovv = zeros({1, 32, 64}, float16);
            auto ovi = zeros({1, 32}, int32);
            auto ovc = zeros({1}, int32);

            auto results = foveated::foveated_attention(
                fk, fv, pk, pv, ps, ps, pvs, pvs,
                fkk, fvv, fks, fks, fvs, fvs, fval,
                q, fk, fv, ovk, ovv, ovi, ovc,
                0.5f, 256);
            eval(results[0]);
            return true;
        } catch (...) {
            return false;
        }
    }, "Check if the C++ foveated extension is available.");
}
