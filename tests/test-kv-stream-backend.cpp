#include "llama-kv-stream-backend.h"
#include "testing.h"

#include "ggml-backend.h"

int main() {
    testing t;

    t.test("a null device reports no streaming support", [](testing & t) {
        const auto caps = llama_kv_stream_backend_caps_query(nullptr, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        t.assert_true("not streamable", !caps.streamable);
        t.assert_true("type pair not supported", !caps.type_pair_supported);
    });

    t.test("a backend that exports no kv-stream symbols reports no support", [](testing & t) {
        // The CPU backend never exports "ggml_backend_kv_stream_supported" -
        // KV streaming targets host<->device staging, which does not apply
        // when the host already is the compute device. This doubles as a
        // regression check that CPU stays a no-op for this feature.
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!t.assert_true("a CPU device is registered", cpu_dev != nullptr)) {
            return;
        }

        const auto caps = llama_kv_stream_backend_caps_query(cpu_dev, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        t.assert_true("CPU is not streamable", !caps.streamable);
        t.assert_true("CPU reports no supported type pair", !caps.type_pair_supported);
    });

    return t.summary();
}
