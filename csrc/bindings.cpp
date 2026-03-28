#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mlx/mlx.h"
#include "foveated_attn.h"
#include "foveated_compress.h"

namespace nb = nanobind;
using namespace mlx::core;

NB_MODULE(foveated_ext, m) {
    m.doc() = "Foveated 2-tier attention: near (fp16) + far (fp8 E4M3 K, int4 V)";

    // ---- BlobWriteInfo (opaque, passed between FoveatedHandle and Pipeline) ----
    nb::class_<foveated::BlobWriteInfo>(m, "BlobWriteInfo");

    // ---- FoveatedHandle: per-layer attention dispatch ----
    nb::class_<foveated::FoveatedHandle>(m, "FoveatedHandle")
        .def(nb::init<
            const array&, const array&, const array&, const array&,
            const array&, const array&, const array&,
            float, const std::string&>(),
            nb::arg("near_k"), nb::arg("near_v"),
            nb::arg("far_k"), nb::arg("far_v"),
            nb::arg("far_v_scale"), nb::arg("far_v_zero"),
            nb::arg("near_valid"),
            nb::arg("spike_margin") = 0.5f,
            nb::arg("metallib_path") = "",
            "Pre-bind static tier arrays into a single blob.")
        .def("__call__",
            &foveated::FoveatedHandle::operator(),
            nb::arg("query"),
            nb::arg("decode_k"), nb::arg("decode_v"),
            "Dispatch fused kernel. Returns (out, spike_flags, spike_tokens).")
        .def("get_blob_info",
            &foveated::FoveatedHandle::get_blob_info,
            "Get blob write info for promotion pipeline registration.");

    // ---- PromotionPipeline: cross-layer promotion worker ----
    nb::class_<foveated::PromotionPipeline>(m, "PromotionPipeline")
        .def(nb::init<int, int, int>(),
            nb::arg("n_layers"),
            nb::arg("cooldown_steps") = 5,
            nb::arg("max_per_drain") = 8,
            "Create promotion pipeline spanning n_layers.")
        .def("register_archive",
            [](foveated::PromotionPipeline& self, int layer_idx,
               const std::string& path_k, const std::string& path_v,
               int H_kv, int S_arc, int D,
               const std::vector<int32_t>& archive_idx) {
                self.register_archive(layer_idx, path_k, path_v,
                                      H_kv, S_arc, D,
                                      archive_idx.data(), (int)archive_idx.size());
            },
            nb::arg("layer_idx"),
            nb::arg("path_k"), nb::arg("path_v"),
            nb::arg("H_kv"), nb::arg("S_arc"), nb::arg("D"),
            nb::arg("archive_idx"),
            "Register disk archive for one layer (builds O(1) lookup maps).")
        .def("register_blob",
            &foveated::PromotionPipeline::register_blob,
            nb::arg("layer_idx"), nb::arg("info"),
            "Register blob write target for one layer.")
        .def("drain_spikes",
            [](foveated::PromotionPipeline& self, int layer_idx,
               const array& spike_flags, const array& spike_tokens,
               const array& far_idx, int current_step) {
                eval({spike_flags, spike_tokens, far_idx});
                const int32_t* flags_ptr = spike_flags.data<int32_t>();
                const int32_t* tokens_ptr = spike_tokens.data<int32_t>();
                const int32_t* far_idx_ptr = far_idx.data<int32_t>();
                int B = spike_flags.shape(0);
                int H_q = spike_flags.shape(1);
                int H_kv = far_idx.shape(1);
                int N_far = far_idx.shape(2);
                self.drain_spikes(layer_idx, flags_ptr, tokens_ptr, far_idx_ptr,
                                  B, H_q, H_kv, N_far, current_step);
            },
            nb::arg("layer_idx"),
            nb::arg("spike_flags"), nb::arg("spike_tokens"),
            nb::arg("far_idx"), nb::arg("current_step"),
            "Process kernel spike outputs for one layer (zero-copy from MLX).")
        .def("get_stats",
            [](foveated::PromotionPipeline& self) -> std::map<std::string, uint64_t> {
                auto& s = self.stats();
                return {
                    {"spikes_detected", s.spikes_detected.load()},
                    {"spikes_queued", s.spikes_queued.load()},
                    {"spikes_deduplicated", s.spikes_deduplicated.load()},
                    {"spikes_cooled_down", s.spikes_cooled_down.load()},
                    {"promotions_completed", s.promotions_completed.load()},
                    {"promotions_headroom_full", s.promotions_headroom_full.load()},
                };
            },
            "Get promotion statistics.")
        .def("stop",
            &foveated::PromotionPipeline::stop,
            "Stop the worker thread.");

    // ---- TurboFoveatedHandle: TurboQuant attention dispatch ----
    nb::class_<foveated::TurboFoveatedHandle>(m, "TurboFoveatedHandle")
        .def(nb::init<
            const array&, const array&,
            const array&, const array&, const array&, const array&,
            const array&, const array&, const array&,
            const array&, const array&, const array&,
            float, const std::string&>(),
            nb::arg("near_k"), nb::arg("near_v"),
            nb::arg("far_k_indices"), nb::arg("far_k_signs"),
            nb::arg("far_k_norm"), nb::arg("far_k_gamma"),
            nb::arg("far_v_packed"), nb::arg("far_v_scale"),
            nb::arg("near_valid"),
            nb::arg("Pi"), nb::arg("S_mat"), nb::arg("centroids"),
            nb::arg("spike_margin") = 0.5f,
            nb::arg("metallib_path") = "",
            "Build TurboQuant handle with compressed tier arrays and constant matrices.")
        .def("__call__",
            &foveated::TurboFoveatedHandle::operator(),
            nb::arg("query"),
            nb::arg("decode_k"), nb::arg("decode_v"),
            "Dispatch TurboQuant kernel. Returns (out, spike_flags, spike_tokens).")
        .def("get_blob_info",
            &foveated::TurboFoveatedHandle::get_blob_info,
            "Get blob write info for promotion pipeline registration.");

    // ---- Compression ----
    nb::class_<foveated::TierConfig>(m, "TierConfig")
        .def(nb::init<>())
        .def_rw("near_pct", &foveated::TierConfig::near_pct)
        .def_rw("n_sinks", &foveated::TierConfig::n_sinks)
        .def_rw("window_size", &foveated::TierConfig::window_size)
        .def_rw("promo_headroom_pct", &foveated::TierConfig::promo_headroom_pct)
        .def_rw("promo_headroom_min", &foveated::TierConfig::promo_headroom_min);

    nb::class_<foveated::CompressedLayer>(m, "CompressedLayer")
        .def_ro("near_k", &foveated::CompressedLayer::near_k)
        .def_ro("near_v", &foveated::CompressedLayer::near_v)
        .def_ro("far_k", &foveated::CompressedLayer::far_k)
        .def_ro("far_v", &foveated::CompressedLayer::far_v)
        .def_ro("far_v_scale", &foveated::CompressedLayer::far_v_scale)
        .def_ro("far_v_zero", &foveated::CompressedLayer::far_v_zero)
        .def_ro("near_valid", &foveated::CompressedLayer::near_valid)
        .def_ro("n_near_actual", &foveated::CompressedLayer::n_near_actual);

    nb::class_<foveated::CompressHandle>(m, "CompressHandle")
        .def(nb::init<const foveated::TierConfig&, const std::string&>(),
            nb::arg("cfg"), nb::arg("metallib_path") = "",
            "Build a compression handle with tier config and metallib path.")
        .def("compress_layer", &foveated::CompressHandle::compress_layer,
            nb::arg("keys"), nb::arg("values"),
            "Compress one layer (graph only, no eval).")
        .def("compress_all", &foveated::CompressHandle::compress_all,
            nb::arg("all_keys"), nb::arg("all_values"),
            "Compress all layers with one mx.eval at the end.");

    m.def("is_available", []() -> bool { return true; },
        "Returns true — the extension loaded successfully.");
}
