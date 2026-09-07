#pragma once

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

#ifndef SPM_MMID_EXTERNAL_ABI
struct mmid_span {
    const void * data;
    void * lease;
};

using mmid_acquire = bool (*)(const ggml_tensor *, int64_t, size_t, mmid_span *, void *);
using mmid_release = void (*)(mmid_span *, void *);
using set_mmid_provider = void (*)(mmid_acquire, mmid_release, void *);
#endif

class mmid_cache {
    struct entry {
        const void * source = nullptr;
        size_t size = 0;
        std::unique_ptr<unsigned char, decltype(&std::free)> bytes{nullptr, &std::free};
        uint64_t observations = 0;
        uint64_t active = 0;
        uint64_t age = 0;
        int layer = -1;
    };

    using key = std::pair<const void *, size_t>;

public:
    explicit mmid_cache(size_t budget, bool layer_aware = false) : budget_(budget), layer_aware_(layer_aware) {}

    bool acquire(const ggml_tensor * tensor, mmid_span * span, size_t size) {
        const auto started = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(mutex_);
        auto & item = entries_[{span->data, size}];
        if (!item) {
            item = std::make_unique<entry>();
            item->source = span->data;
            item->size = size;
            std::sscanf(tensor == nullptr ? "" : tensor->name, "blk.%d.", &item->layer);
        }
        entry & value = *item;
        const bool new_observation = value.active == 0;
        if (new_observation) {
            ++value.observations;
            ++operations_;
            if (value.bytes) {
                ++operation_hits_;
            } else {
                ++operation_misses_;
            }
        }
        ++acquires_;
        if (value.bytes) {
            ++hits_;
        } else {
            ++misses_;
            admit(value);
            if (new_observation && !value.bytes) {
                source_bytes_ += size;
            }
        }
        ++value.active;
        value.age = ++age_;
        span->data = value.bytes ? value.bytes.get() : value.source;
        span->lease = &value;
        add_wait(started);
        return true;
    }

    void release(mmid_span * span) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto * value = static_cast<entry *>(span->lease);
        if (value == nullptr || value->active == 0) {
            std::abort();
        }
        --value->active;
        ++releases_;
        span->lease = nullptr;
    }

    void report() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::printf(
            "mmid_cache operations=%llu operation_hits=%llu operation_misses=%llu acquires=%llu releases=%llu worker_hits=%llu worker_misses=%llu loads=%llu evictions=%llu source_bytes=%llu resident=%zu peak=%zu wait_ms=%.3f balanced=%s\n",
            value(operations_), value(operation_hits_), value(operation_misses_),
            value(acquires_), value(releases_), value(hits_), value(misses_), value(loads_),
            value(evictions_), value(source_bytes_), resident_, peak_, value(wait_ns_) / 1e6,
            acquires_ == releases_ && acquires_ > 0 ? "true" : "false");
    }

    bool balanced() const {
        return acquires_ == releases_ && acquires_ > 0;
    }

private:
    static unsigned long long value(uint64_t number) {
        return static_cast<unsigned long long>(number);
    }

    void add_wait(std::chrono::steady_clock::time_point started) {
        wait_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count();
    }

    bool make_room(size_t size, const entry * incoming) {
        while (resident_ + size > budget_) {
            auto victim = entries_.end();
            for (auto it = entries_.begin(); it != entries_.end(); ++it) {
                const entry & candidate = *it->second;
                const bool same_layer = candidate.layer == incoming->layer;
                const bool victim_same = victim != entries_.end() && victim->second->layer == incoming->layer;
                if (&candidate != incoming && candidate.bytes && candidate.active == 0 &&
                        (victim == entries_.end() ||
                         (!layer_aware_ && candidate.age < victim->second->age) ||
                         (layer_aware_ && ((same_layer && !victim_same) ||
                          (same_layer == victim_same && candidate.age < victim->second->age))))) {
                    victim = it;
                }
            }
            if (victim == entries_.end()) {
                return false;
            }
            resident_ -= victim->second->size;
            victim->second->bytes.reset();
            ++evictions_;
        }
        return true;
    }

    void admit(entry & value) {
        if (value.observations < 2 || value.size > budget_ || !make_room(value.size, &value)) {
            return;
        }
        void * allocation = nullptr;
        if (posix_memalign(&allocation, 64, value.size) != 0) {
            return;
        }
        value.bytes.reset(static_cast<unsigned char *>(allocation));
        std::memcpy(value.bytes.get(), value.source, value.size);
        resident_ += value.size;
        peak_ = std::max(peak_, resident_);
        source_bytes_ += value.size;
        ++loads_;
    }

    size_t budget_;
    bool layer_aware_;
    size_t resident_ = 0;
    size_t peak_ = 0;
    uint64_t age_ = 0;
    uint64_t operations_ = 0;
    uint64_t operation_hits_ = 0;
    uint64_t operation_misses_ = 0;
    uint64_t acquires_ = 0;
    uint64_t releases_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t loads_ = 0;
    uint64_t evictions_ = 0;
    uint64_t source_bytes_ = 0;
    uint64_t wait_ns_ = 0;
    std::mutex mutex_;
    std::map<key, std::unique_ptr<entry>> entries_;
};

inline bool cache_acquire(const ggml_tensor * tensor, int64_t, size_t size, mmid_span * span, void * data) {
    return static_cast<mmid_cache *>(data)->acquire(tensor, span, size);
}

inline void cache_release(mmid_span * span, void * data) {
    static_cast<mmid_cache *>(data)->release(span);
}
