#include "engine.h"
#include <iostream>
#include <iomanip>
#include <cstring>

// ─────────────────────────────────────────────────────────────
// Construction / Destruction
// ─────────────────────────────────────────────────────────────

AsyncEngine::AsyncEngine(size_t buffer_capacity)
    : ring_capacity_(buffer_capacity)
{
    ring_buf_.resize(ring_capacity_);
}

AsyncEngine::~AsyncEngine()
{
    // Ensure threads are joined if still running
    if (producer_.joinable()) producer_.join();
    if (consumer_.joinable()) consumer_.join();
}

// ─────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────

void AsyncEngine::set_inference(InferenceFunc fn)
{
    inference_fn_ = std::move(fn);
}

void AsyncEngine::set_threshold(ThresholdFunc fn)
{
    threshold_fn_ = std::move(fn);
}

// ─────────────────────────────────────────────────────────────
// Ring Buffer Helpers (mutex-based, bounded)
// ─────────────────────────────────────────────────────────────

bool AsyncEngine::ring_is_full() const
{
    return ((ring_head_ + 1) % ring_capacity_) == ring_tail_;
}

bool AsyncEngine::ring_is_empty() const
{
    return ring_head_ == ring_tail_;
}

bool AsyncEngine::ring_push(const EnrichedRecord& item)
{
    std::unique_lock<std::mutex> lock(ring_mutex_);

    // Wait until there is space — never drop data
    ring_not_full_.wait(lock, [this]() {
        return !ring_is_full();
    });

    ring_buf_[ring_head_] = item;
    ring_head_ = (ring_head_ + 1) % ring_capacity_;

    lock.unlock();
    ring_not_empty_.notify_one();
    return true;
}

bool AsyncEngine::ring_pop(EnrichedRecord& item)
{
    std::unique_lock<std::mutex> lock(ring_mutex_);

    // Wait until there is data or the producer is done
    ring_not_empty_.wait(lock, [this]() {
        return !ring_is_empty() || producer_done_.load(std::memory_order_acquire);
    });

    if (ring_is_empty()) {
        // Producer finished and buffer drained
        return false;
    }

    item = ring_buf_[ring_tail_];
    ring_tail_ = (ring_tail_ + 1) % ring_capacity_;

    lock.unlock();
    ring_not_full_.notify_one();
    return true;
}

// ─────────────────────────────────────────────────────────────
// Producer Thread
// ─────────────────────────────────────────────────────────────

void AsyncEngine::producer_thread()
{
    // Spin-wait for data to be fed (lightweight; happens once)
    while (!data_ready_.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    if (feed_data_ == nullptr) {
        producer_done_.store(true, std::memory_order_release);
        ring_not_empty_.notify_all();
        return;
    }

    for (const auto& record : *feed_data_) {
        ring_push(record);
        total_produced_.fetch_add(1, std::memory_order_relaxed);
    }

    // Signal that no more items will be produced
    producer_done_.store(true, std::memory_order_release);
    ring_not_empty_.notify_all();
}

// ─────────────────────────────────────────────────────────────
// Consumer Thread
// ─────────────────────────────────────────────────────────────

void AsyncEngine::consumer_thread()
{
    EnrichedRecord record{};

    while (true) {
        if (!ring_pop(record)) {
            // Buffer empty and producer done — exit
            break;
        }

        // ── Timed inference ──────────────────────────────────
        auto t0 = std::chrono::high_resolution_clock::now();
        float probability = inference_fn_(record.features, MAX_FEATURES);
        int8_t predicted   = threshold_fn_(probability);
        auto t1 = std::chrono::high_resolution_clock::now();

        double latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        // ── Build result ─────────────────────────────────────
        InferenceResult result{};
        result.context         = record.context;
        result.probability     = probability;
        result.predicted_label = predicted;
        result.actual_label    = record.label;
        result.latency_us      = static_cast<float>(latency_us);

        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            results_.push_back(result);
            total_latency_us_ += latency_us;
        }

        total_consumed_.fetch_add(1, std::memory_order_relaxed);
    }

    all_done_.store(true, std::memory_order_release);
}

// ─────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────

void AsyncEngine::start()
{
    producer_ = std::thread(&AsyncEngine::producer_thread, this);
    consumer_ = std::thread(&AsyncEngine::consumer_thread, this);
}

void AsyncEngine::feed(const std::vector<EnrichedRecord>& records)
{
    feed_data_ = &records;
    data_ready_.store(true, std::memory_order_release);
}

void AsyncEngine::stop_and_wait()
{
    if (producer_.joinable()) producer_.join();
    if (consumer_.joinable()) consumer_.join();
}

// ─────────────────────────────────────────────────────────────
// Summary
// ─────────────────────────────────────────────────────────────

void AsyncEngine::print_summary()
{
    uint32_t produced = total_produced_.load();
    uint32_t consumed = total_consumed_.load();
    uint32_t drops    = drop_count_.load();

    double avg_latency_ms = 0.0;
    if (consumed > 0) {
        avg_latency_ms = (total_latency_us_ / consumed) / 1000.0;
    }

    double drop_rate = 0.0;
    if (produced > 0) {
        drop_rate = 100.0 * static_cast<double>(drops) / static_cast<double>(produced);
    }

    // Build a completed progress bar
    constexpr int BAR_WIDTH = 20;
    std::string bar(BAR_WIDTH, '\xe2');          // placeholder
    // Use simple ASCII block for maximum terminal compatibility
    std::string progress_bar;
    for (int i = 0; i < BAR_WIDTH; ++i) {
        progress_bar += "█";
    }

    std::cout << "\n";
    std::cout << "[STAGE 5] INT8 Quantization & Async Engine ";
    std::cout << "───────────────────\n";
    std::cout << "  ✓ Ring Buffer: " << ring_capacity_ << " slots initialized\n";
    std::cout << "  ✓ Producer thread: COMPLETED (" << produced << " records)\n";
    std::cout << "  ✓ Consumer thread: COMPLETED (" << consumed << " records)\n";
    std::cout << "  ✓ Processing... [" << progress_bar << "] "
              << consumed << "/" << produced << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  ✓ Avg inference latency: " << avg_latency_ms << " ms\n";
    std::cout << "  ✓ Data drop rate: " << drop_rate << "%\n";
}
