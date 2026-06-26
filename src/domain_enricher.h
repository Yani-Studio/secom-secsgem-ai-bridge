#pragma once
// ============================================================================
// domain_enricher.h — Stage 2: 도메인 정보 부가기 (Domain Enricher)
// FAB(반도체 공장) 컨텍스트를 전처리 데이터에 매핑
// ============================================================================

#include "preprocessor.h"
#include <cstdint>
#include <array>
#include <string>

constexpr uint32_t MAX_FEATURES = 50;

// FAB 공정 컨텍스트 구조체
struct FabContext {
    char     lot_id[20];       // "LOT-20080719-0001"  (로트 ID)
    char     equipment_id[16]; // "EQP-CVD-01"         (장비 ID)
    uint8_t  chamber_no;       // 0=A, 1=B             (챔버 번호)
    uint8_t  slot_no;          // 1~25                 (슬롯 번호)
    uint64_t timestamp_us;     // epoch microseconds   (타임스탬프)
};

// 도메인 정보가 부가된 레코드
struct EnrichedRecord {
    FabContext context;
    float     features[MAX_FEATURES];  // 전처리된 피처 (고정 크기)
    int8_t    label;                   // -1=Pass(양품), +1=Fail(불량)
    uint8_t   padding[3];              // 메모리 정렬용 패딩
};

class DomainEnricher {
public:
    // 전처리 데이터에 FAB 컨텍스트 부가
    std::vector<EnrichedRecord> enrich(
        const PreprocessedData& data,
        const std::vector<std::string>& timestamps);

    // 부가된 레코드를 바이너리 파일로 저장
    void save_binary(const std::vector<EnrichedRecord>& records,
                     const std::string& path);

    // 바이너리 파일에서 부가된 레코드 로드
    std::vector<EnrichedRecord> load_binary(const std::string& path);
};
