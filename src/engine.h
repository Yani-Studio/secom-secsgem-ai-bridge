#pragma once
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <functional>
#include <vector>
#include <string>
#include <cstdint>
#include "domain_enricher.h"

struct InferenceResult {
    FabContext context;
    float probability;
    int8_t predicted_label;  // +1=Fail, -1=Pass
    int8_t actual_label;
    float latency_us;        // inference time in microseconds
};

// Note: This header needs ring_buffer.h to be included before it
// when used in engine.cpp

class AsyncEngine {
public:
    using InferenceFunc = std::function<float(const float*, uint32_t)>;
    using ThresholdFunc = std::function<int8_t(float)>;
    
    AsyncEngine(size_t buffer_capacity = 1024);
    ~AsyncEngine();
    
    // Set the inference and threshold functions
    void set_inference(InferenceFunc fn);
    void set_threshold(ThresholdFunc fn);
    
    // Start producer and consumer threads
    void start();
    
    // Feed data to producer
    void feed(const std::vector<EnrichedRecord>& records);
    
    // Signal completion and wait for all processing
    void stop_and_wait();
    
    // Get results
    const std::vector<InferenceResult>& results() const { return results_; }
    
    // Print engine summary
    void print_summary();
    
private:
    void producer_thread();
    void consumer_thread();
    
    InferenceFunc inference_fn_;
    ThresholdFunc threshold_fn_;
    
    // Thread management
    std::thread producer_;
    std::thread consumer_;
    std::atomic<bool> producer_done_{false};
    std::atomic<bool> all_done_{false};
    
    // Data feed
    const std::vector<EnrichedRecord>* feed_data_ = nullptr;
    std::atomic<bool> data_ready_{false};
    
    // Ring buffer (simple mutex-based for compatibility)
    std::vector<EnrichedRecord> ring_buf_;
    size_t ring_capacity_;
    size_t ring_head_ = 0;
    size_t ring_tail_ = 0;
    std::mutex ring_mutex_;
    std::condition_variable ring_not_full_;
    std::condition_variable ring_not_empty_;
    
    bool ring_push(const EnrichedRecord& item);
    bool ring_pop(EnrichedRecord& item);
    bool ring_is_full() const;
    bool ring_is_empty() const;
    
    // Results
    std::vector<InferenceResult> results_;
    std::mutex results_mutex_;
    
    // Stats
    std::atomic<uint32_t> total_produced_{0};
    std::atomic<uint32_t> total_consumed_{0};
    std::atomic<uint32_t> drop_count_{0};
    double total_latency_us_ = 0.0;
};
