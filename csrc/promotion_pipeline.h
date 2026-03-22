#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace foveated {

// ============================================================================
// Promotion pipeline: C++ worker that promotes far-tier tokens into near-tier
// headroom slots by reading fp16 from disk mmap and writing directly into the
// blob's unified memory. The kernel sees promoted tokens on the next dispatch
// via the near_valid[h] count — zero override buffers, zero kernel changes.
//
// Design:
//   - Standalone object, one per generation session (spans all layers)
//   - Each layer registers its blob (BlobWriteInfo) and archive (ArchiveInfo)
//   - drain_spikes() runs on main thread: filters spikes, queues SpikeRecords
//   - Worker thread: reads fp16 from mmap, memcpy into blob, atomic commit
//
// Thread safety:
//   - drain_spikes() must be called from a single thread (main decode thread)
//   - Worker thread only touches blob memory and its own deque
//   - near_valid[h] is the atomic commit point (ARM64 word-atomic)
//   - No locks on the hot path (worker pops from deque under mutex)
// ============================================================================

struct SpikeRecord {
    int layer_idx;
    int head_idx;      // KV head index
    int arc_local;     // archive-local index for mmap read
    int position;      // global token position (for stats/debugging)
};

struct PromotionStats {
    std::atomic<uint64_t> spikes_detected{0};
    std::atomic<uint64_t> spikes_queued{0};
    std::atomic<uint64_t> spikes_deduplicated{0};
    std::atomic<uint64_t> spikes_cooled_down{0};
    std::atomic<uint64_t> promotions_completed{0};
    std::atomic<uint64_t> promotions_headroom_full{0};
};

struct ArchiveInfo {
    void* mmap_k = nullptr;    // POSIX mmap, layout: (H_kv, S_arc, D) float16
    void* mmap_v = nullptr;
    size_t mmap_size_k = 0;
    size_t mmap_size_v = 0;
    int fd_k = -1;
    int fd_v = -1;
    int H_kv = 0, S_arc = 0, D = 0;

    // O(1) position → archive-local index lookup, per head.
    // Built at register_archive time from the flat archive_idx array.
    // Key: token position, Value: archive-local index within that head.
    std::vector<std::unordered_map<int32_t, int>> pos_to_local;  // [H_kv]
};

struct BlobWriteInfo {
    uint8_t* blob_ptr = nullptr;  // raw pointer into blob's unified memory
    size_t near_k_offset = 0;    // byte offset to near_k region in blob
    size_t near_v_offset = 0;    // byte offset to near_v region in blob
    size_t near_valid_offset = 0; // byte offset to near_valid array in blob
    int N_near_alloc = 0;        // allocated near slots per head (actual + headroom)
    int H_kv = 0, D = 0;
};

class PromotionPipeline {
public:
    PromotionPipeline(int n_layers, int cooldown_steps = 5, int max_per_drain = 8);
    ~PromotionPipeline();

    // Non-copyable, non-movable (owns thread + mmap resources)
    PromotionPipeline(const PromotionPipeline&) = delete;
    PromotionPipeline& operator=(const PromotionPipeline&) = delete;

    // Register a disk archive for one layer. Builds O(1) position lookup maps.
    void register_archive(int layer_idx, const std::string& path_k,
                          const std::string& path_v,
                          int H_kv, int S_arc, int D,
                          const int32_t* archive_idx_data,
                          int archive_idx_len);

    // Register the blob write target for one layer (call per layer).
    void register_blob(int layer_idx, BlobWriteInfo info);

    // Process kernel spike outputs. Main thread only, once per step per layer.
    // Reads spike_flags/tokens zero-copy from unified memory pointers.
    void drain_spikes(int layer_idx,
                      const int32_t* spike_flags,   // flat (B * H_q)
                      const int32_t* spike_tokens,  // flat (B * H_q)
                      const int32_t* far_idx,       // flat (B * H_kv * N_far)
                      int B, int H_q, int H_kv, int N_far,
                      int current_step);

    PromotionStats& stats() { return stats_; }
    void stop();

private:
    void worker_loop();

    int n_layers_, cooldown_steps_, max_per_drain_;

    std::vector<ArchiveInfo> archives_;
    std::vector<BlobWriteInfo> blobs_;

    // Dedup set: hash of (layer_idx, head_idx, position).
    // Prevents re-promoting the same token. Cleared implicitly by pipeline
    // lifetime (one pipeline per generation session).
    struct Dedup {
        size_t operator()(uint64_t key) const {
            // Splitmix64 finalizer — excellent distribution for sequential keys
            key ^= key >> 30;
            key *= 0xbf58476d1ce4e5b9ULL;
            key ^= key >> 27;
            key *= 0x94d049bb133111ebULL;
            key ^= key >> 31;
            return key;
        }
    };
    std::unordered_set<uint64_t, Dedup> promoted_;

    // Per-(layer, head) cooldown tracking.
    // Key: (layer_idx << 16) | head_idx. Safe for H_kv < 65536.
    std::unordered_map<uint32_t, int> last_spike_step_;

    std::mutex queue_mutex_;
    std::deque<SpikeRecord> spike_queue_;
    std::condition_variable queue_cv_;

    std::thread worker_;
    std::atomic<bool> running_{true};
    PromotionStats stats_;
};

} // namespace foveated
