#pragma once
// ============================================================================
// preprocessor.h — Stage 1: 데이터 전처리기 (Data Preprocessor)
// 반도체 공정 센서 데이터 전처리 파이프라인
// ============================================================================

#include <string>
#include <vector>
#include <cstdint>

// 전처리 완료된 데이터 구조체
struct PreprocessedData {
    uint32_t num_rows;
    uint32_t num_features;  // 피처 선택 후 컬럼 수
    std::vector<float> data;      // row-major flat array [num_rows × num_features]
    std::vector<int8_t> labels;   // -1=Pass(양품), +1=Fail(불량)
    std::vector<uint32_t> selected_feature_indices; // 원본 컬럼 인덱스
    std::vector<float> feature_min; // min-max 역정규화용 최솟값
    std::vector<float> feature_max; // min-max 역정규화용 최댓값
};

class Preprocessor {
public:
    // CSV 로드 → NaN 중앙값 대체 → 분산 0 컬럼 제거 →
    // 상위 N개 피처 피어슨 상관계수 선택 → min-max 정규화 [0,1]
    PreprocessedData process(const std::string& csv_path,
                             uint32_t top_n_features = 50);

    // 전처리 데이터를 바이너리 파일로 저장
    void save_binary(const PreprocessedData& data,
                     const std::string& bin_path);

    // 바이너리 파일에서 전처리 데이터 로드
    PreprocessedData load_binary(const std::string& bin_path);

    // Stage 2에서 사용할 타임스탬프 문자열 벡터
    const std::vector<std::string>& get_timestamps() const {
        return timestamps_;
    }

private:
    std::vector<std::string> timestamps_;  // CSV의 Time 컬럼 보관
};
