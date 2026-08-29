#include "llama-kv-stream-config.h"
#include "testing.h"

int main() {
    testing t;

    t.test("streaming is opt-in", [](testing & t) {
        llama_kv_stream_config config;
        const auto result = llama_kv_stream_config_validate(config);
        t.assert_true("disabled config is valid", result.valid);
        t.assert_true("disabled config remains disabled", !result.enabled);
    });

    t.test("supported target configuration is accepted", [](testing & t) {
        llama_kv_stream_config config;
        config.stage_bytes         = 64ULL*1024ULL*1024ULL;
        config.minimum_stage_bytes = 1664ULL*256ULL;
        config.unified_kv_cache    = true;
        config.context_default     = true;
        config.single_sequence     = true;
        config.flash_attention     = true;
        config.kv_offload          = true;
        config.type_pair_supported = true;

        const auto result = llama_kv_stream_config_validate(config);
        t.assert_true("config is valid", result.valid);
        t.assert_true("config is enabled", result.enabled);
    });

    t.test("each unsupported condition fails loudly", [](testing & t) {
        llama_kv_stream_config base;
        base.stage_bytes         = 64ULL*1024ULL*1024ULL;
        base.minimum_stage_bytes = 1664ULL*256ULL;
        base.unified_kv_cache    = true;
        base.context_default     = true;
        base.single_sequence     = true;
        base.flash_attention     = true;
        base.kv_offload          = true;
        base.type_pair_supported = true;

        auto expect_invalid = [&](const char * name, const llama_kv_stream_config & config) {
            const auto result = llama_kv_stream_config_validate(config);
            t.assert_true(name, !result.valid && !result.enabled && !result.error.empty());
        };

        auto config = base;
        config.unified_kv_cache = false;
        expect_invalid("non-unified KV cache architecture", config);
        config = base;
        config.context_default = false;
        expect_invalid("draft/MTP context", config);
        config = base;
        config.single_sequence = false;
        expect_invalid("parallel sequences", config);
        config = base;
        config.flash_attention = false;
        expect_invalid("Flash Attention disabled", config);
        config = base;
        config.kv_offload = false;
        expect_invalid("KV offload disabled", config);
        config = base;
        config.type_pair_supported = false;
        expect_invalid("unsupported cache type pair", config);
        config = base;
        config.stage_bytes = config.minimum_stage_bytes - 1;
        expect_invalid("stage smaller than one page", config);
    });

    return t.summary();
}
