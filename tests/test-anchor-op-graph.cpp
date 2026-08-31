/*
 * Test: GGML_OP_ANCHOR_DECOMPRESS real graph vs reference anchor_kv_decompress_head.
 *
 * Builds the same compressed representation that anchor_kv_compress produces,
 * uploads it as ggml tensors (matching llama_kv_cache::anchor_kv_upload_layer),
 * builds a real ggml graph with ggml_anchor_decompress, computes it on the
 * CPU backend, and compares the op output against the reference output
 * element by element.
 *
 * This is the only test that exercises the actual ggml op. test-anchor-op-math
 * replicates the op math in plain C++ but never calls the op.
 */

#include "anchor-kv.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>

// wrap C structs in unique_ptr for RAII
struct ggml_backend_ptr {
    ggml_backend_t ptr = nullptr;
    ~ggml_backend_ptr() { if (ptr) ggml_backend_free(ptr); }
    ggml_backend_t get() { return ptr; }
};

struct ggml_backend_buffer_ptr {
    ggml_backend_buffer_t ptr = nullptr;
    ~ggml_backend_buffer_ptr() { if (ptr) ggml_backend_buffer_free(ptr); }
    ggml_backend_buffer_t get() { return ptr; }
};

static float mse(const float * a, const float * b, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double) a[i] - b[i];
        sum += d * d;
    }
    return (float) (sum / (double) n);
}

static float max_abs(const float * a, const float * b, size_t n) {
    float m = 0;
    for (size_t i = 0; i < n; i++) m = std::max(m, fabsf(a[i] - b[i]));
    return m;
}

// print first divergence between op and ref output
static void print_first_divergence(const float * op, const float * ref, size_t n,
                                   const char * label, int n_embd, int D) {
    for (size_t i = 0; i < n; i++) {
        if (fabsf(op[i] - ref[i]) > 1e-3f) {
            int t = (int)(i / n_embd);
            int rem = (int)(i % n_embd);
            int h = rem / D;
            int d = rem % D;
            fprintf(stderr, "  [%s] first divergence at t=%d h=%d d=%d: op=%.6f ref=%.6f diff=%.6f\n",
                    label, t, h, d, op[i], ref[i], op[i] - ref[i]);
            // print a few more
            int count = 0;
            for (size_t j = i + 1; j < n && count < 5; j++) {
                if (fabsf(op[j] - ref[j]) > 1e-3f) {
                    int tj = (int)(j / n_embd);
                    int rj = (int)(j % n_embd);
                    int hj = rj / D;
                    int dj = rj % D;
                    fprintf(stderr, "  [%s]   also at t=%d h=%d d=%d: op=%.6f ref=%.6f\n",
                            label, tj, hj, dj, op[j], ref[j]);
                    count++;
                }
            }
            return;
        }
    }
    fprintf(stderr, "  [%s] no divergence > 1e-3 found\n", label);
}

int main() {
    const int S = 33;
    const int D = 128;
    const int n_heads = 4;
    const float theta = 0.54f;

    // deterministic pseudo-random data
    std::vector<float> keys((size_t) n_heads * S * D);
    std::vector<float> values((size_t) n_heads * S * D);
    uint64_t rng = 12345;
    for (size_t i = 0; i < keys.size(); i++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        keys[i] = ((float) ((rng >> 33) & 0xffff) / 32768.0f - 1.0f) * 2.0f;
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        values[i] = ((float) ((rng >> 33) & 0xffff) / 32768.0f - 1.0f) * 2.0f;
    }

    anchor_kv_params params;
    params.theta = theta;
    params.W = 16;
    params.k_frac = 16;
    params.rho = 0.5f;
    params.kappa = 8;
    params.compress_k = true;
    params.compress_v = true;

    anchor_kv_layer layer = anchor_kv_compress(
        keys.data(), values.data(), S, D, n_heads, params);

    int total_K = 0, total_V = 0;
    for (int h = 0; h < n_heads; h++) {
        total_K += layer.heads[h].n_K;
        total_V += layer.heads[h].n_V;
    }
    if (total_K == 0 || total_V == 0) {
        fprintf(stderr, "FAIL: no residuals allocated (n_K=%d n_V=%d)\n", total_K, total_V);
        return 1;
    }

    printf("layer: %d heads, k=%d anchors, n_K=%d n_V=%d (head 0)\n",
           layer.n_heads, layer.heads[0].k, layer.heads[0].n_K, layer.heads[0].n_V);

    // ---- reference output (head-major [h, S, D]) ----
    std::vector<float> ref_k((size_t) n_heads * S * D);
    std::vector<float> ref_v((size_t) n_heads * S * D);
    for (int h = 0; h < n_heads; h++) {
        anchor_kv_decompress_head(layer.heads[h],
            &ref_k[(size_t) h * S * D], &ref_v[(size_t) h * S * D]);
    }

    // remap reference to cache-major [t, n_embd, d] (matching the op output layout)
    const int n_embd = n_heads * D;
    std::vector<float> ref_k_cm((size_t) n_embd * S);
    std::vector<float> ref_v_cm((size_t) n_embd * S);
    for (int h = 0; h < n_heads; h++) {
        for (int t = 0; t < S; t++) {
            memcpy(&ref_k_cm[(size_t) t * n_embd + h * D],
                   &ref_k  [(size_t) h * S * D + t * D], D * sizeof(float));
            memcpy(&ref_v_cm[(size_t) t * n_embd + h * D],
                   &ref_v  [(size_t) h * S * D + t * D], D * sizeof(float));
        }
    }

    // ---- build ggml tensors matching anchor_kv_upload_layer ----
    int k = layer.heads[0].k;
    int n_K_max = 0, n_V_max = 0;
    for (int h = 0; h < n_heads; h++) {
        n_K_max = std::max(n_K_max, layer.heads[h].n_K);
        n_V_max = std::max(n_V_max, layer.heads[h].n_V);
    }
    const int cpr = D / 4;

    // pack data (same layout as anchor_kv_upload_layer)
    std::vector<ggml_bf16_t> anchors_data((size_t) n_heads * 2 * k * D, ggml_fp32_to_bf16(0.0f));
    std::vector<int32_t> anchor_of_data(2 * n_heads * S, 0);
    std::vector<ggml_fp16_t> gamma_data(2 * n_heads * S, ggml_fp32_to_fp16(0.0f));
    std::vector<int32_t> slot_of_data(2 * n_heads * S, -1);
    std::vector<uint8_t> k_codes((size_t) n_heads * n_K_max * cpr, 0);
    std::vector<float> k_scales((size_t) n_heads * n_K_max, 0.0f);
    std::vector<uint8_t> v_codes((size_t) n_heads * n_V_max * cpr, 0);
    std::vector<float> v_scales((size_t) n_heads * n_V_max, 0.0f);

    for (int h = 0; h < n_heads; h++) {
        const anchor_kv_head & head = layer.heads[h];
        for (int a = 0; a < head.k; a++) {
            ggml_fp32_to_bf16_row(&head.anchor_keys[a * D],
                    &anchors_data[((h * 2 + 0) * k + a) * D], D);
            ggml_fp32_to_bf16_row(&head.anchor_values[a * D],
                    &anchors_data[((h * 2 + 1) * k + a) * D], D);
        }
        for (int t = 0; t < S; t++) {
            anchor_of_data[(0 * n_heads + h) * S + t] = head.k_anchor_of[t];
            anchor_of_data[(1 * n_heads + h) * S + t] = head.v_anchor_of[t];
            gamma_data[(0 * n_heads + h) * S + t]     = ggml_fp32_to_fp16(head.k_gamma[t]);
            gamma_data[(1 * n_heads + h) * S + t]     = ggml_fp32_to_fp16(head.v_gamma[t]);
            slot_of_data[(0 * n_heads + h) * S + t]   = head.k_slot_of[t];
            slot_of_data[(1 * n_heads + h) * S + t]   = head.v_slot_of[t];
        }
        for (int i = 0; i < head.n_K; i++) {
            memcpy(&k_codes[(size_t) (h * n_K_max + i) * cpr], &head.k_res_codes[i * cpr], cpr);
            k_scales[h * n_K_max + i] = head.k_res_scales[i];
        }
        for (int i = 0; i < head.n_V; i++) {
            memcpy(&v_codes[(size_t) (h * n_V_max + i) * cpr], &head.v_res_codes[i * cpr], cpr);
            v_scales[h * n_V_max + i] = head.v_res_scales[i];
        }
    }

    // ---- create ggml context and tensors ----
    // tensors: 8 srcs + 2 dst (scratch_k, scratch_v) + 2 graph nodes + overhead
    ggml_init_params iparams = {
        /*.mem_size   =*/ 512 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(iparams);
    if (!ctx) {
        fprintf(stderr, "FAIL: ggml_init failed\n");
        return 1;
    }

    // tensor dims match the fixed layout (after the dimension fix):
    // anchors:      [D, k, 2, n_heads] bf16
    // anchor_of:    [S, n_heads, 2] i32
    // gamma:        [S, n_heads, 2] f16
    // slot_of:      [S, n_heads, 2] i32
    // k_res_codes:  [cpr, n_K_max, n_heads] i8
    // k_res_scales: [n_K_max, n_heads] f32
    // v_res_codes:  [cpr, n_V_max, n_heads] i8
    // v_res_scales: [n_V_max, n_heads] f32
    // scratch:      [n_embd, kv_size] f16 (kv_size = S for this test)
    ggml_tensor * t_anchors      = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, D, k, 2, n_heads);
    ggml_tensor * t_anchor_of    = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, S, n_heads, 2);
    ggml_tensor * t_gamma        = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, S, n_heads, 2);
    ggml_tensor * t_slot_of      = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, S, n_heads, 2);
    ggml_tensor * t_k_res_codes  = ggml_new_tensor_3d(ctx, GGML_TYPE_I8,  cpr, n_K_max, n_heads);
    ggml_tensor * t_k_res_scales = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_K_max, n_heads);
    ggml_tensor * t_v_res_codes  = ggml_new_tensor_3d(ctx, GGML_TYPE_I8,  cpr, n_V_max, n_heads);
    ggml_tensor * t_v_res_scales = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_V_max, n_heads);

    // scratch (dst) tensors - one for K, one for V
    ggml_tensor * t_scratch_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, S);
    ggml_tensor * t_scratch_v = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, S);

    // allocate on CPU
    ggml_backend_ptr backend;
    backend.ptr = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend.get()) {
        fprintf(stderr, "FAIL: failed to init CPU backend\n");
        return 1;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend.get());

    ggml_backend_buffer_ptr buf;
    buf.ptr = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf.get()) {
        fprintf(stderr, "FAIL: failed to allocate tensors\n");
        return 1;
    }

    // upload data
    ggml_backend_tensor_set(t_anchors,      anchors_data.data(),   0, anchors_data.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(t_anchor_of,    anchor_of_data.data(), 0, anchor_of_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_gamma,        gamma_data.data(),     0, gamma_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(t_slot_of,      slot_of_data.data(),   0, slot_of_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_k_res_codes,  k_codes.data(),        0, k_codes.size());
    ggml_backend_tensor_set(t_k_res_scales, k_scales.data(),       0, k_scales.size() * sizeof(float));
    ggml_backend_tensor_set(t_v_res_codes,  v_codes.data(),        0, v_codes.size());
    ggml_backend_tensor_set(t_v_res_scales, v_scales.data(),       0, v_scales.size() * sizeof(float));

    // clear scratch
    ggml_backend_tensor_set(t_scratch_k, std::vector<uint8_t>(n_embd * S * 2, 0).data(), 0, n_embd * S * 2);
    ggml_backend_tensor_set(t_scratch_v, std::vector<uint8_t>(n_embd * S * 2, 0).data(), 0, n_embd * S * 2);

    // ---- build graph: K decompress ----
    // n_rot=0 disables RoPE so we compare pure reconstruction math
    ggml_cgraph * gf_k = ggml_new_graph(ctx);
    ggml_tensor * op_k = ggml_anchor_decompress(
        ctx,
        /* dst = */ t_scratch_k,
        t_anchors, t_anchor_of, t_gamma, t_slot_of,
        t_k_res_codes, t_k_res_scales, t_v_res_codes, t_v_res_scales,
        /* is_k = */ true,
        /* prev_dep = */ nullptr,
        /* n_rot = */ 0, /* freq_base = */ 10000.0f, /* freq_scale = */ 1.0f);
    ggml_build_forward_expand(gf_k, op_k);

    if (ggml_backend_graph_compute(backend.get(), gf_k) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: K graph compute failed\n");
        return 1;
    }

    // read K output
    std::vector<ggml_fp16_t> op_k_raw((size_t) n_embd * S);
    ggml_backend_tensor_get(t_scratch_k, op_k_raw.data(), 0, op_k_raw.size() * sizeof(ggml_fp16_t));
    std::vector<float> op_k_f((size_t) n_embd * S);
    for (size_t i = 0; i < op_k_raw.size(); i++) {
        op_k_f[i] = ggml_fp16_to_fp32(op_k_raw[i]);
    }

    // ---- build graph: V decompress ----
    ggml_cgraph * gf_v = ggml_new_graph(ctx);
    ggml_tensor * op_v = ggml_anchor_decompress(
        ctx,
        /* dst = */ t_scratch_v,
        t_anchors, t_anchor_of, t_gamma, t_slot_of,
        t_k_res_codes, t_k_res_scales, t_v_res_codes, t_v_res_scales,
        /* is_k = */ false,
        /* prev_dep = */ nullptr,
        /* n_rot = */ 0, /* freq_base = */ 10000.0f, /* freq_scale = */ 1.0f);
    ggml_build_forward_expand(gf_v, op_v);

    // need to re-clear scratch_v since K compute may have left it dirty
    ggml_backend_tensor_set(t_scratch_v, std::vector<uint8_t>(n_embd * S * 2, 0).data(), 0, n_embd * S * 2);

    if (ggml_backend_graph_compute(backend.get(), gf_v) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: V graph compute failed\n");
        return 1;
    }

    std::vector<ggml_fp16_t> op_v_raw((size_t) n_embd * S);
    ggml_backend_tensor_get(t_scratch_v, op_v_raw.data(), 0, op_v_raw.size() * sizeof(ggml_fp16_t));
    std::vector<float> op_v_f((size_t) n_embd * S);
    for (size_t i = 0; i < op_v_raw.size(); i++) {
        op_v_f[i] = ggml_fp16_to_fp32(op_v_raw[i]);
    }

    // ---- compare ----
    const size_t total = (size_t) n_embd * S;

    float mse_k = mse(op_k_f.data(), ref_k_cm.data(), total);
    float mse_v = mse(op_v_f.data(), ref_v_cm.data(), total);
    float max_k = max_abs(op_k_f.data(), ref_k_cm.data(), total);
    float max_v = max_abs(op_v_f.data(), ref_v_cm.data(), total);

    printf("K: mse=%.3e max_abs=%.3e\n", mse_k, max_k);
    printf("V: mse=%.3e max_abs=%.3e\n", mse_v, max_v);

    if (mse_k > 1e-3f || max_k > 1e-1f) {
        print_first_divergence(op_k_f.data(), ref_k_cm.data(), total, "K", n_embd, D);
    }
    if (mse_v > 1e-3f || max_v > 1e-1f) {
        print_first_divergence(op_v_f.data(), ref_v_cm.data(), total, "V", n_embd, D);
    }

    // bf16/f16 storage rounding introduces small errors; tolerance is generous
    const bool pass = mse_k < 1e-1f && mse_v < 1e-1f && max_k < 1.0f && max_v < 1.0f;
    printf(pass ? "PASS: op matches reference\n" : "FAIL: op diverges from reference\n");

    ggml_free(ctx);
    return pass ? 0 : 1;
}
