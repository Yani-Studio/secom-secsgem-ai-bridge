// ============================================================================
// domain_enricher.cpp — Stage 2: 도메인 정보 부가기 구현
// FAB 컨텍스트 생성 (로트ID, 장비ID, 챔버, 슬롯, 타임스탬프)
// ============================================================================

#include "domain_enricher.h"

#include <algorithm>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// 장비 ID 목록 — CVD (Chemical Vapor Deposition) 장비 4대 로테이션
// ─────────────────────────────────────────────────────────────────────────────
static const char* EQUIPMENT_IDS[] = {
    "EQP-CVD-01",
    "EQP-CVD-02",
    "EQP-CVD-03",
    "EQP-CVD-04"
};
static constexpr uint32_t NUM_EQUIPMENT = 4;
static constexpr uint32_t WAFERS_PER_LOT = 25;  // 1로트 = 25매 웨이퍼

// ─────────────────────────────────────────────────────────────────────────────
// 타임스탬프 문자열 → epoch microseconds 변환
// 형식: "YYYY-MM-DD HH:MM:SS"
// ─────────────────────────────────────────────────────────────────────────────
static uint64_t parse_timestamp_to_us(const std::string& ts_str) {
    struct tm tm_val = {};
    int year = 0, month = 0, day = 0, hour = 0, minute = 0, second = 0;

    // sscanf로 파싱
    if (std::sscanf(ts_str.c_str(), "%d-%d-%d %d:%d:%d",
                    &year, &month, &day, &hour, &minute, &second) != 6) {
        // 파싱 실패 시 0 반환
        return 0;
    }

    tm_val.tm_year = year - 1900;
    tm_val.tm_mon  = month - 1;
    tm_val.tm_mday = day;
    tm_val.tm_hour = hour;
    tm_val.tm_min  = minute;
    tm_val.tm_sec  = second;
    tm_val.tm_isdst = -1;  // DST 자동 판단

    // mktime은 로컬 타임존 기준 → epoch seconds
    time_t epoch_sec = mktime(&tm_val);
    if (epoch_sec == static_cast<time_t>(-1)) {
        return 0;
    }

    // seconds → microseconds
    return static_cast<uint64_t>(epoch_sec) * 1000000ULL;
}

// ─────────────────────────────────────────────────────────────────────────────
// 타임스탬프 문자열에서 날짜 부분(YYYYMMDD) 추출
// ─────────────────────────────────────────────────────────────────────────────
static std::string extract_date_compact(const std::string& ts_str) {
    // "YYYY-MM-DD HH:MM:SS" → "YYYYMMDD"
    int year = 0, month = 0, day = 0;
    if (std::sscanf(ts_str.c_str(), "%d-%d-%d", &year, &month, &day) != 3) {
        return "00000000";
    }

    char buf[16];
    std::snprintf(buf, sizeof(buf), "%04d%02d%02d", year, month, day);
    return std::string(buf);
}

// =============================================================================
// enrich(): 전처리 데이터에 FAB 컨텍스트 부가
// =============================================================================
std::vector<EnrichedRecord> DomainEnricher::enrich(
        const PreprocessedData& data,
        const std::vector<std::string>& timestamps) {

    std::cout << "\n[STAGE 2] Domain Enrichment "
              << std::string(34, '-') << std::endl;

    const uint32_t num_rows = data.num_rows;
    const uint32_t num_features = data.num_features;

    // 결과 레코드 벡터 할당
    std::vector<EnrichedRecord> records(num_rows);

    // 로트 수 계산
    uint32_t total_lots = (num_rows + WAFERS_PER_LOT - 1) / WAFERS_PER_LOT;

    for (uint32_t r = 0; r < num_rows; ++r) {
        EnrichedRecord& rec = records[r];
        std::memset(&rec, 0, sizeof(EnrichedRecord));

        // ─── 로트 번호 및 슬롯 계산 ───────────────────────────────────
        uint32_t lot_index = r / WAFERS_PER_LOT;         // 0-based 로트 인덱스
        uint32_t slot = (r % WAFERS_PER_LOT) + 1;        // 1~25 슬롯 번호

        // ─── Lot ID 생성: "LOT-YYYYMMDD-XXXX" ────────────────────────
        std::string date_str;
        if (r < timestamps.size() && !timestamps[r].empty()) {
            date_str = extract_date_compact(timestamps[r]);
        } else {
            date_str = "00000000";
        }

        char lot_id_buf[20];
        std::snprintf(lot_id_buf, sizeof(lot_id_buf),
                      "LOT-%s-%04u", date_str.c_str(), lot_index + 1);
        std::strncpy(rec.context.lot_id, lot_id_buf, sizeof(rec.context.lot_id) - 1);
        rec.context.lot_id[sizeof(rec.context.lot_id) - 1] = '\0';

        // ─── Equipment ID: 로트별 라운드 로빈 ────────────────────────
        uint32_t equip_index = lot_index % NUM_EQUIPMENT;
        std::strncpy(rec.context.equipment_id,
                     EQUIPMENT_IDS[equip_index],
                     sizeof(rec.context.equipment_id) - 1);
        rec.context.equipment_id[sizeof(rec.context.equipment_id) - 1] = '\0';

        // ─── Chamber 번호: 로트별 교대 (0=A, 1=B) ────────────────────
        rec.context.chamber_no = static_cast<uint8_t>(lot_index % 2);

        // ─── Slot 번호: 1~25 ─────────────────────────────────────────
        rec.context.slot_no = static_cast<uint8_t>(slot);

        // ─── Timestamp: 문자열 → epoch microseconds ──────────────────
        if (r < timestamps.size() && !timestamps[r].empty()) {
            rec.context.timestamp_us = parse_timestamp_to_us(timestamps[r]);
        } else {
            rec.context.timestamp_us = 0;
        }

        // ─── 피처 복사 (PreprocessedData → 고정 크기 배열) ────────────
        for (uint32_t f = 0; f < num_features && f < MAX_FEATURES; ++f) {
            rec.features[f] = data.data[r * num_features + f];
        }
        // 남은 슬롯 0으로 초기화 (memset으로 이미 처리됨)

        // ─── 라벨 복사 ──────────────────────────────────────────────
        rec.label = data.labels[r];

        // ─── 패딩 초기화 (memset으로 이미 처리됨) ────────────────────
    }

    // ─── 통계 출력 ──────────────────────────────────────────────────────
    std::cout << "  \xe2\x9c\x93 Lot IDs generated: " << total_lots
              << " lots (" << WAFERS_PER_LOT << " wafers/lot)" << std::endl;

    std::cout << "  \xe2\x9c\x93 Equipment rotation: "
              << EQUIPMENT_IDS[0] << " ~ "
              << EQUIPMENT_IDS[NUM_EQUIPMENT - 1] << std::endl;

    std::cout << "  \xe2\x9c\x93 Timestamps converted: string \xe2\x86\x92 uint64_t \xc2\xb5s"
              << std::endl;

    return records;
}

// =============================================================================
// save_binary(): EnrichedRecord 벡터를 바이너리 파일로 저장
// =============================================================================
void DomainEnricher::save_binary(const std::vector<EnrichedRecord>& records,
                                  const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("바이너리 파일을 생성할 수 없습니다: " + path);
    }

    // 레코드 수 기록
    uint32_t count = static_cast<uint32_t>(records.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));

    // EnrichedRecord 크기 기록 (역직렬화 호환성용)
    uint32_t record_size = static_cast<uint32_t>(sizeof(EnrichedRecord));
    out.write(reinterpret_cast<const char*>(&record_size), sizeof(uint32_t));

    // 레코드 데이터 기록
    out.write(reinterpret_cast<const char*>(records.data()),
              static_cast<std::streamsize>(count) * record_size);

    out.close();

    // 파일 크기 확인
    std::ifstream check(path, std::ios::binary | std::ios::ate);
    auto file_size = check.tellg();
    check.close();

    std::cout << "  \xe2\x9c\x93 Enriched binary saved: " << path
              << " (" << (file_size / 1024) << " KB)" << std::endl;
}

// =============================================================================
// load_binary(): 바이너리 파일에서 EnrichedRecord 벡터 로드
// =============================================================================
std::vector<EnrichedRecord> DomainEnricher::load_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("바이너리 파일을 열 수 없습니다: " + path);
    }

    uint32_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));

    uint32_t record_size = 0;
    in.read(reinterpret_cast<char*>(&record_size), sizeof(uint32_t));

    if (record_size != sizeof(EnrichedRecord)) {
        throw std::runtime_error(
            "레코드 크기 불일치: 파일=" + std::to_string(record_size) +
            ", 현재=" + std::to_string(sizeof(EnrichedRecord)));
    }

    std::vector<EnrichedRecord> records(count);
    in.read(reinterpret_cast<char*>(records.data()),
            static_cast<std::streamsize>(count) * record_size);

    in.close();
    return records;
}
