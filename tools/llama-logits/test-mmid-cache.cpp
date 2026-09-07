#include "mmid-cache.hpp"

#include <array>
#include <cassert>
#include <thread>
#include <vector>

int main() {
    std::array<unsigned char, 64> first{};
    std::array<unsigned char, 64> second{};
    first.fill(17);
    second.fill(29);
    mmid_cache cache(64);

    mmid_span first_span{first.data(), nullptr};
    assert(cache.acquire(nullptr, &first_span, first.size()));
    assert(first_span.data == first.data());
    cache.release(&first_span);

    first_span = {first.data(), nullptr};
    assert(cache.acquire(nullptr, &first_span, first.size()));
    assert(first_span.data != first.data());

    mmid_span second_span{second.data(), nullptr};
    assert(cache.acquire(nullptr, &second_span, second.size()));
    cache.release(&second_span);
    second_span = {second.data(), nullptr};
    assert(cache.acquire(nullptr, &second_span, second.size()));
    assert(second_span.data == second.data());
    cache.release(&second_span);
    cache.release(&first_span);

    second_span = {second.data(), nullptr};
    assert(cache.acquire(nullptr, &second_span, second.size()));
    assert(second_span.data != second.data());
    cache.release(&second_span);

    std::vector<std::thread> workers;
    for (int index = 0; index < 8; ++index) {
        workers.emplace_back([&] {
            mmid_span span{second.data(), nullptr};
            assert(cache.acquire(nullptr, &span, second.size()));
            assert(static_cast<const unsigned char *>(span.data)[0] == 29);
            cache.release(&span);
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }
    assert(cache.balanced());
}
