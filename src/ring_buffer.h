#pragma once
#include <array>
#include <atomic>
#include <cstddef>

// Lock-free SPSC (Single Producer, Single Consumer) 링 버퍼
// 반도체 센서 데이터 스트리밍을 위한 무잠금 큐
template<typename T, size_t N = 1024>
class RingBuffer {
    static_assert((N & (N - 1)) == 0, "N must be power of 2");
    static_assert(N > 0, "N must be positive");
    
public:
    bool push(const T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t next_head = (current_head + 1) & (N - 1);
        
        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false;  // Buffer full - DO NOT drop data
        }
        
        buf_[current_head] = item;
        head_.store(next_head, std::memory_order_release);
        return true;
    }
    
    bool pop(T& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        
        if (current_tail == head_.load(std::memory_order_acquire)) {
            return false;  // Buffer empty
        }
        
        item = buf_[current_tail];
        tail_.store((current_tail + 1) & (N - 1), std::memory_order_release);
        return true;
    }
    
    bool is_full() const {
        const size_t next_head = (head_.load(std::memory_order_relaxed) + 1) & (N - 1);
        return next_head == tail_.load(std::memory_order_relaxed);
    }
    
    bool is_empty() const {
        return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_relaxed);
    }
    
    size_t size() const {
        const size_t h = head_.load(std::memory_order_relaxed);
        const size_t t = tail_.load(std::memory_order_relaxed);
        return (h - t) & (N - 1);
    }
    
    size_t capacity() const { return N - 1; }
    
private:
    std::array<T, N> buf_{};
    alignas(64) std::atomic<size_t> head_{0};  // 캐시 라인 분리로 false sharing 방지
    alignas(64) std::atomic<size_t> tail_{0};
};
