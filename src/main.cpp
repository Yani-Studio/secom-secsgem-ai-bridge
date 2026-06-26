#include <iostream>
#include <vector>
#include "preprocessor.h"
#include "domain_enricher.h"
#include "models.h"
#include "quantizer.h"
#include "engine.h"

int main() {
    std::cout << "\n════════════════════════════════════════════════════════════════\n";
    std::cout << "  SECOM On-Device AI Collection Engine v1.0\n";
    std::cout << "  Copyright (c) 2026 Kang Gyu Min. All rights reserved.\n";
    std::cout << "════════════════════════════════════════════════════════════════\n\n";

    std::cout << "\n[STAGE 1] Data Preprocessing ---------------------------------\n";
    Preprocessor preprocessor;
    auto data = preprocessor.process("uci-secom.csv", 150);
    preprocessor.save_binary(data, "data/secom_preprocessed.bin");

    std::cout << "\n[STAGE 2] Domain Enrichment ----------------------------------\n";
    DomainEnricher enricher;
    auto enriched_data = enricher.enrich(data, preprocessor.get_timestamps());

    std::cout << "\n[STAGE 3] Loading 5-Model Ensemble ---------------------------\n";
    EnsembleEngine engine;
    if (!engine.load_weights("data/model_weights.bin")) {
        return 1;
    }
    engine.print_summary();

    std::cout << "\n[STAGE 4] INT8 Quantization Evaluation -----------------------\n";
    Quantizer quantizer;
    
    // Count total parameters
    size_t total_params = 0;
    size_t i = 0;
    while(IModel* m = engine.get_model(i++)) {
        total_params += m->param_count();
    }
    
    QuantizationStats stats = quantizer.compute_stats(total_params);
    quantizer.print_summary(stats);
    
    std::cout << "\n[STAGE 5] Async Inference Engine -----------------------------\n";
    AsyncEngine async_engine(1024);
    async_engine.set_inference([&engine](const float* features, uint32_t n) {
        return engine.predict(features, n);
    });
    // Threshold adjusted for higher recall manually from python results
    async_engine.set_threshold([](float prob) {
        return prob >= 0.20f ? 1 : -1;
    });
    
    async_engine.start();
    async_engine.feed(enriched_data);
    async_engine.stop_and_wait();
    async_engine.print_summary();

    return 0;
}
