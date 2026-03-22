// Promotion pipeline: reads fp16 K,V from disk mmap, writes into near-tier
// headroom in the blob's unified memory. One atomic uint32 increment on
// near_valid[h] makes the token visible to the kernel on next dispatch.

#include "promotion_pipeline.h"

#include <atomic>
#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace foveated {

// ============================================================================
// Construction / destruction
// ============================================================================

PromotionPipeline::PromotionPipeline(int n_layers, int cooldown_steps, int max_per_drain)
    : n_layers_(n_layers),
      cooldown_steps_(cooldown_steps),
      max_per_drain_(max_per_drain),
      archives_(n_layers),
      blobs_(n_layers)
{
    worker_ = std::thread(&PromotionPipeline::worker_loop, this);
}

PromotionPipeline::~PromotionPipeline() {
    stop();
    for (auto& a : archives_) {
        if (a.mmap_k && a.mmap_k != MAP_FAILED) munmap(a.mmap_k, a.mmap_size_k);
        if (a.mmap_v && a.mmap_v != MAP_FAILED) munmap(a.mmap_v, a.mmap_size_v);
        if (a.fd_k >= 0) close(a.fd_k);
        if (a.fd_v >= 0) close(a.fd_v);
    }
}


// ============================================================================
// Registration
// ============================================================================

void PromotionPipeline::register_archive(
    int layer_idx, const std::string& path_k, const std::string& path_v,
    int H_kv, int S_arc, int D,
    const int32_t* archive_idx_data, int archive_idx_len)
{
    if (layer_idx < 0 || layer_idx >= n_layers_) return;

    auto& a = archives_[layer_idx];
    a.H_kv = H_kv;
    a.S_arc = S_arc;
    a.D = D;

    // Build O(1) position → local index maps, one per KV head.
    a.pos_to_local.resize(H_kv);
    for (int h = 0; h < H_kv; h++) {
        auto& m = a.pos_to_local[h];
        m.reserve(S_arc);
        int base = h * S_arc;
        for (int i = 0; i < S_arc && (base + i) < archive_idx_len; i++)
            m[archive_idx_data[base + i]] = i;
    }

    // POSIX mmap for K and V files.
    size_t bytes = (size_t)H_kv * S_arc * D * sizeof(uint16_t);

    a.fd_k = open(path_k.c_str(), O_RDONLY);
    if (a.fd_k >= 0) {
        a.mmap_size_k = bytes;
        a.mmap_k = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, a.fd_k, 0);
        if (a.mmap_k == MAP_FAILED) a.mmap_k = nullptr;
    }

    a.fd_v = open(path_v.c_str(), O_RDONLY);
    if (a.fd_v >= 0) {
        a.mmap_size_v = bytes;
        a.mmap_v = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, a.fd_v, 0);
        if (a.mmap_v == MAP_FAILED) a.mmap_v = nullptr;
    }
}

void PromotionPipeline::register_blob(int layer_idx, BlobWriteInfo info) {
    if (layer_idx < 0 || layer_idx >= n_layers_) return;
    blobs_[layer_idx] = info;
}


// ============================================================================
// drain_spikes — main thread, called once per step per layer
//
// Reads spike_flags/tokens zero-copy from MLX unified memory. Filters through
// GQA dedup, per-head cooldown, and position dedup. Pushes SpikeRecords to
// the worker queue. O(1) archive lookup via pre-built hash maps.
// ============================================================================

void PromotionPipeline::drain_spikes(
    int layer_idx,
    const int32_t* spike_flags,
    const int32_t* spike_tokens,
    const int32_t* far_idx,
    int B, int H_q, int H_kv, int N_far,
    int current_step)
{
    if (layer_idx < 0 || layer_idx >= n_layers_) return;

    const auto& archive = archives_[layer_idx];
    if (!archive.mmap_k || archive.S_arc == 0) return;

    const int gqa = (H_kv > 0) ? (H_q / H_kv) : 1;
    int queued = 0;

    // GQA dedup: track which KV heads we've already processed this call.
    // Use a small vector for H_kv > 64 (avoids bitmask truncation).
    thread_local std::vector<bool> seen_kv;
    seen_kv.assign(H_kv, false);

    for (int b = 0; b < B && queued < max_per_drain_; b++) {
        for (int h_q = 0; h_q < H_q && queued < max_per_drain_; h_q++) {
            const int flat = b * H_q + h_q;
            if (spike_flags[flat] == 0) continue;

            const int h_kv = h_q / gqa;
            if (seen_kv[h_kv]) continue;
            seen_kv[h_kv] = true;

            stats_.spikes_detected.fetch_add(1, std::memory_order_relaxed);

            // Per-(layer, head) cooldown
            const uint32_t cd_key = ((uint32_t)layer_idx << 16) | (uint32_t)h_kv;
            auto cd_it = last_spike_step_.find(cd_key);
            if (cd_it != last_spike_step_.end() &&
                (current_step - cd_it->second) < cooldown_steps_) {
                stats_.spikes_cooled_down.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            last_spike_step_[cd_key] = current_step;

            // Resolve far-local index to global position
            const int far_local = spike_tokens[flat];
            if (far_local < 0 || far_local >= N_far) continue;
            const int position = far_idx[b * H_kv * N_far + h_kv * N_far + far_local];

            // Position dedup — full 64-bit key, no truncation.
            // Pack: layer in upper 16, head in next 16, position in lower 32.
            const uint64_t dedup_key =
                ((uint64_t)(layer_idx & 0xFFFF) << 48) |
                ((uint64_t)(h_kv & 0xFFFF) << 32) |
                ((uint64_t)(uint32_t)position);
            if (promoted_.count(dedup_key)) {
                stats_.spikes_deduplicated.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            promoted_.insert(dedup_key);

            // O(1) archive lookup via pre-built hash map
            if (h_kv >= (int)archive.pos_to_local.size()) continue;
            const auto& pos_map = archive.pos_to_local[h_kv];
            auto arc_it = pos_map.find(position);
            if (arc_it == pos_map.end()) continue;
            const int arc_local = arc_it->second;

            // Push to worker queue
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                spike_queue_.push_back({layer_idx, h_kv, arc_local, position});
            }
            queue_cv_.notify_one();
            stats_.spikes_queued.fetch_add(1, std::memory_order_relaxed);
            queued++;
        }
    }
}


// ============================================================================
// Worker thread — reads fp16 from mmap, writes into blob headroom
//
// Commit protocol:
//   1. memcpy K data into blob near_k at slot near_valid[h]
//   2. memcpy V data into blob near_v at slot near_valid[h]
//   3. atomic_thread_fence(release) — ensures K,V visible before count
//   4. atomic store near_valid[h] += 1 — commit point
//
// The kernel reads near_valid[h] once at dispatch start. ARM64 guarantees
// word-aligned uint32 reads/writes are atomic. The release fence ensures
// the kernel never sees incremented count without the K,V data behind it.
// ============================================================================

void PromotionPipeline::worker_loop() {
    while (running_.load(std::memory_order_relaxed)) {
        SpikeRecord rec;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::milliseconds(1), [this] {
                return !spike_queue_.empty() || !running_.load(std::memory_order_relaxed);
            });
            if (spike_queue_.empty()) continue;
            rec = spike_queue_.front();
            spike_queue_.pop_front();
        }

        const auto& archive = archives_[rec.layer_idx];
        const auto& blob = blobs_[rec.layer_idx];
        if (!archive.mmap_k || !blob.blob_ptr) continue;

        const int h = rec.head_idx;
        const int D = archive.D;
        const size_t elem_bytes = D * sizeof(uint16_t);  // fp16 = 2 bytes per element

        // Read current near_valid[h] atomically
        auto* near_valid_ptr = reinterpret_cast<std::atomic<uint32_t>*>(
            blob.blob_ptr + blob.near_valid_offset);
        const uint32_t cur_valid = near_valid_ptr[h].load(std::memory_order_acquire);

        // Check headroom
        if ((int)cur_valid >= blob.N_near_alloc) {
            stats_.promotions_headroom_full.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // Source: mmap layout is (H_kv, S_arc, D) float16
        const size_t src_offset = ((size_t)h * archive.S_arc + rec.arc_local) * elem_bytes;
        const uint8_t* src_k = (const uint8_t*)archive.mmap_k + src_offset;
        const uint8_t* src_v = (const uint8_t*)archive.mmap_v + src_offset;

        // Destination: blob near layout is (B * H_kv, N_near_alloc, D) float16
        // Batch index is always 0 (single-batch generation).
        const size_t slot_offset = ((size_t)h * blob.N_near_alloc + cur_valid) * elem_bytes;
        std::memcpy(blob.blob_ptr + blob.near_k_offset + slot_offset, src_k, elem_bytes);
        std::memcpy(blob.blob_ptr + blob.near_v_offset + slot_offset, src_v, elem_bytes);

        // Commit: release fence + atomic store makes K,V visible before count
        std::atomic_thread_fence(std::memory_order_release);
        near_valid_ptr[h].store(cur_valid + 1, std::memory_order_release);

        stats_.promotions_completed.fetch_add(1, std::memory_order_relaxed);
    }
}

void PromotionPipeline::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) return;
    queue_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

} // namespace foveated
