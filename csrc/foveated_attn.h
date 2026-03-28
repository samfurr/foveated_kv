#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/primitives.h"
#include "promotion_pipeline.h"

namespace MTL { class Buffer; class ComputePipelineState; }

namespace foveated {

// ============================================================================
// Shared structs — match the Metal kernel layout exactly
//
// 2-tier: near (fp16) + far (fp8 E4M3 K, int4 V)
// ============================================================================

struct BlobLayout {
    size_t near_k, near_v;
    size_t far_k, far_v;
    size_t far_v_scale, far_v_zero;
    size_t near_valid;
    size_t total;
};

struct BlobOffsets {
    uint32_t near_k, near_v;
    uint32_t far_k, far_v;
    uint32_t far_v_scale, far_v_zero;
    uint32_t near_valid;
};

struct FoveatedParams {
    uint32_t total_bh_q;
    uint32_t n_decode;
    float spike_margin;
};


// ============================================================================
// FoveatedPrimitive: zero-overhead eval_gpu
// ============================================================================

class FoveatedPrimitive : public mlx::core::Primitive {
 public:
    FoveatedPrimitive(
        mlx::core::Stream stream,
        MTL::ComputePipelineState* pipeline,
        const MTL::Buffer* blob_buf,
        int64_t blob_offset,
        BlobOffsets blob_offsets,
        FoveatedParams params,
        int total_bh_q,
        int num_splits)
        : Primitive(stream),
          pipeline_(pipeline),
          blob_buf_(blob_buf),
          blob_offset_(blob_offset),
          blob_offsets_(blob_offsets),
          params_(params),
          total_bh_q_(total_bh_q),
          num_splits_(num_splits) {}

    void eval_cpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override {
        throw std::runtime_error("FoveatedPrimitive only runs on GPU");
    }

    void eval_gpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override;

    const char* name() const override { return "FoveatedPrimitive"; }

    bool is_equivalent(const Primitive& other) const override {
        auto* o = dynamic_cast<const FoveatedPrimitive*>(&other);
        return o && pipeline_ == o->pipeline_
            && params_.total_bh_q == o->params_.total_bh_q
            && params_.n_decode == o->params_.n_decode;
    }

 private:
    MTL::ComputePipelineState* pipeline_;
    const MTL::Buffer* blob_buf_;
    int64_t blob_offset_;
    BlobOffsets blob_offsets_;
    FoveatedParams params_;
    int total_bh_q_;
    int num_splits_;
};


// ============================================================================
// FoveatedHandle: per-layer attention cache + kernel dispatch
//
// Does NOT own the promotion pipeline. The pipeline is a separate object
// that spans all layers. FoveatedHandle exposes get_blob_info() so the
// pipeline can register each layer's blob independently.
// ============================================================================

class FoveatedHandle {
 public:
    FoveatedHandle(
        const mlx::core::array& near_k, const mlx::core::array& near_v,
        const mlx::core::array& far_k, const mlx::core::array& far_v,
        const mlx::core::array& far_v_scale, const mlx::core::array& far_v_zero,
        const mlx::core::array& near_valid,
        float spike_margin = 0.5f,
        const std::string& metallib_path = "");

    // Dispatch: query + decode K,V → (output, spike_flags, spike_tokens)
    std::vector<mlx::core::array> operator()(
        const mlx::core::array& query,
        const mlx::core::array& decode_k, const mlx::core::array& decode_v);

    // Expose blob write info for the promotion pipeline to register.
    BlobWriteInfo get_blob_info() const;

 private:
    mlx::core::array blob_;
    const MTL::Buffer* blob_buf_;
    int64_t blob_offset_;
    BlobOffsets blob_offsets_;

    std::unordered_map<uint64_t, MTL::ComputePipelineState*> pipelines_;

    int B_, H_kv_, D_, N_near_, N_far_, N_static_;
    float spike_margin_;
    std::string metallib_path_;
};

// ============================================================================
// TurboQuant structs — match the Metal TurboBlobOffsets layout
// ============================================================================

struct TurboBlobLayout {
    size_t near_k, near_v;
    size_t far_k_indices, far_k_signs;
    size_t far_k_norm, far_k_gamma;
    size_t far_v_packed, far_v_scale;
    size_t near_valid;
    size_t total;
};

struct TurboBlobOffsets {
    uint32_t near_k, near_v;
    uint32_t far_k_indices, far_k_signs;
    uint32_t far_k_norm, far_k_gamma;
    uint32_t far_v_packed, far_v_scale;
    uint32_t near_valid;
};


// ============================================================================
// TurboPrimitive: TurboQuant attention eval_gpu
// ============================================================================

class TurboPrimitive : public mlx::core::Primitive {
 public:
    TurboPrimitive(
        mlx::core::Stream stream,
        MTL::ComputePipelineState* pipeline,
        const MTL::Buffer* blob_buf,
        int64_t blob_offset,
        TurboBlobOffsets blob_offsets,
        FoveatedParams params,
        int total_bh_q,
        int num_splits)
        : Primitive(stream),
          pipeline_(pipeline),
          blob_buf_(blob_buf),
          blob_offset_(blob_offset),
          blob_offsets_(blob_offsets),
          params_(params),
          total_bh_q_(total_bh_q),
          num_splits_(num_splits) {}

    void eval_cpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override {
        throw std::runtime_error("TurboPrimitive only runs on GPU");
    }

    void eval_gpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override;

    const char* name() const override { return "TurboPrimitive"; }

    bool is_equivalent(const Primitive& other) const override {
        auto* o = dynamic_cast<const TurboPrimitive*>(&other);
        return o && pipeline_ == o->pipeline_
            && params_.total_bh_q == o->params_.total_bh_q
            && params_.n_decode == o->params_.n_decode;
    }

 private:
    MTL::ComputePipelineState* pipeline_;
    const MTL::Buffer* blob_buf_;
    int64_t blob_offset_;
    TurboBlobOffsets blob_offsets_;
    FoveatedParams params_;
    int total_bh_q_;
    int num_splits_;
};


// ============================================================================
// TurboFoveatedHandle: per-layer TurboQuant attention dispatch
// ============================================================================

class TurboFoveatedHandle {
 public:
    TurboFoveatedHandle(
        const mlx::core::array& near_k, const mlx::core::array& near_v,
        const mlx::core::array& far_k_indices,
        const mlx::core::array& far_k_signs,
        const mlx::core::array& far_k_norm,
        const mlx::core::array& far_k_gamma,
        const mlx::core::array& far_v_packed,
        const mlx::core::array& far_v_scale,
        const mlx::core::array& near_valid,
        const mlx::core::array& Pi,
        const mlx::core::array& S_mat,
        const mlx::core::array& centroids,
        float spike_margin = 0.5f,
        const std::string& metallib_path = "");

    std::vector<mlx::core::array> operator()(
        const mlx::core::array& query,
        const mlx::core::array& decode_k, const mlx::core::array& decode_v);

    BlobWriteInfo get_blob_info() const;

 private:
    mlx::core::array blob_;
    const MTL::Buffer* blob_buf_;
    int64_t blob_offset_;
    TurboBlobOffsets blob_offsets_;

    // Shared constant arrays (not in blob — passed as separate buffers)
    mlx::core::array Pi_, S_mat_, centroids_;

    std::unordered_map<uint64_t, MTL::ComputePipelineState*> pipelines_;

    int B_, H_kv_, D_, N_near_, N_far_, N_static_;
    float spike_margin_;
    std::string metallib_path_;
};

} // namespace foveated
