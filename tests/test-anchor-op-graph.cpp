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

// --------- shared-scratch ordering test (multi-layer overwrite) ---------
//
// In the real decode graph all 36 layers share one scratch buffer per side.
// Each layer's GGML_OP_ANCHOR_DECOMPRESS writes positions [0, S) of that
// scratch, overwriting the previous layer's data. The chain/prev_dep mechanism
// must force the previous layer's READ (cpy into a separate accumulation
// buffer) to happen before the next layer's WRITE. This test models that: two
// independent compressed "layers" write the SAME scratch, with a materialized
// read (ggml_cpy) of layer A's output chained as opB's prev_dep. If readA keeps
// layer A's data and readB gets layer B's data, the ordering is correct.

struct anchor_packed {
    std::vector<ggml_bf16_t> anchors;
    std::vector<int32_t> anchor_of;
    std::vector<ggml_fp16_t> gamma;
    std::vector<int32_t> slot_of;
    std::vector<uint8_t> k_codes;
    std::vector<float> k_scales;
    std::vector<uint8_t> v_codes;
    std::vector<float> v_scales;
    int n_K_max = 0;
    int n_V_max = 0;
};

static anchor_packed anchor_pack_layer(const anchor_kv_layer & layer, int n_heads, int S, int D) {
    const int k = layer.heads[0].k;
    const int cpr = D / 4;
    anchor_packed p;
    for (int h = 0; h < n_heads; h++) {
        p.n_K_max = std::max(p.n_K_max, layer.heads[h].n_K);
        p.n_V_max = std::max(p.n_V_max, layer.heads[h].n_V);
    }
    p.anchors.assign((size_t) n_heads * 2 * k * D, ggml_fp32_to_bf16(0.0f));
    p.anchor_of.assign(2 * n_heads * S, 0);
    p.gamma.assign(2 * n_heads * S, ggml_fp32_to_fp16(0.0f));
    p.slot_of.assign(2 * n_heads * S, -1);
    p.k_codes.assign((size_t) n_heads * p.n_K_max * cpr, 0);
    p.k_scales.assign((size_t) n_heads * p.n_K_max, 0.0f);
    p.v_codes.assign((size_t) n_heads * p.n_V_max * cpr, 0);
    p.v_scales.assign((size_t) n_heads * p.n_V_max, 0.0f);

    for (int h = 0; h < n_heads; h++) {
        const anchor_kv_head & head = layer.heads[h];
        for (int a = 0; a < head.k; a++) {
            ggml_fp32_to_bf16_row(&head.anchor_keys[a * D],   &p.anchors[((h * 2 + 0) * k + a) * D], D);
            ggml_fp32_to_bf16_row(&head.anchor_values[a * D], &p.anchors[((h * 2 + 1) * k + a) * D], D);
        }
        for (int t = 0; t < S; t++) {
            p.anchor_of[(0 * n_heads + h) * S + t] = head.k_anchor_of[t];
            p.anchor_of[(1 * n_heads + h) * S + t] = head.v_anchor_of[t];
            p.gamma[(0 * n_heads + h) * S + t]     = ggml_fp32_to_fp16(head.k_gamma[t]);
            p.gamma[(1 * n_heads + h) * S + t]     = ggml_fp32_to_fp16(head.v_gamma[t]);
            p.slot_of[(0 * n_heads + h) * S + t]   = head.k_slot_of[t];
            p.slot_of[(1 * n_heads + h) * S + t]   = head.v_slot_of[t];
        }
        for (int i = 0; i < head.n_K; i++) {
            memcpy(&p.k_codes[(size_t) (h * p.n_K_max + i) * cpr], &head.k_res_codes[i * cpr], cpr);
            p.k_scales[h * p.n_K_max + i] = head.k_res_scales[i];
        }
        for (int i = 0; i < head.n_V; i++) {
            memcpy(&p.v_codes[(size_t) (h * p.n_V_max + i) * cpr], &head.v_res_codes[i * cpr], cpr);
            p.v_scales[h * p.n_V_max + i] = head.v_res_scales[i];
        }
    }
    return p;
}

// a compressed layer + its packed tensors + cache-major reference output
struct layer_data {
    anchor_packed packed;
    std::vector<float> ref_k_cm;
    std::vector<float> ref_v_cm;
};

static layer_data make_layer_data(uint64_t seed, int S, int D, int n_heads, const anchor_kv_params & params) {
    layer_data ld;
    std::vector<float> keys((size_t) n_heads * S * D);
    std::vector<float> values((size_t) n_heads * S * D);
    uint64_t rng = seed;
    for (size_t i = 0; i < keys.size(); i++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        keys[i] = ((float) ((rng >> 33) & 0xffff) / 32768.0f - 1.0f) * 2.0f;
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        values[i] = ((float) ((rng >> 33) & 0xffff) / 32768.0f - 1.0f) * 2.0f;
    }
    anchor_kv_layer layer = anchor_kv_compress(keys.data(), values.data(), S, D, n_heads, params);
    ld.packed = anchor_pack_layer(layer, n_heads, S, D);

    std::vector<float> ref_k((size_t) n_heads * S * D);
    std::vector<float> ref_v((size_t) n_heads * S * D);
    for (int h = 0; h < n_heads; h++) {
        anchor_kv_decompress_head(layer.heads[h], &ref_k[(size_t) h * S * D], &ref_v[(size_t) h * S * D]);
    }
    const int n_embd = n_heads * D;
    ld.ref_k_cm.assign((size_t) n_embd * S, 0.0f);
    ld.ref_v_cm.assign((size_t) n_embd * S, 0.0f);
    for (int h = 0; h < n_heads; h++) {
        for (int t = 0; t < S; t++) {
            memcpy(&ld.ref_k_cm[(size_t) t * n_embd + h * D], &ref_k[(size_t) h * S * D + t * D], D * sizeof(float));
            memcpy(&ld.ref_v_cm[(size_t) t * n_embd + h * D], &ref_v[(size_t) h * S * D + t * D], D * sizeof(float));
        }
    }
    return ld;
}

struct anchor_tensors {
    ggml_tensor * anchors;
    ggml_tensor * anchor_of;
    ggml_tensor * gamma;
    ggml_tensor * slot_of;
    ggml_tensor * k_codes;
    ggml_tensor * k_scales;
    ggml_tensor * v_codes;
    ggml_tensor * v_scales;
};

static anchor_tensors anchor_make_tensors(ggml_context * ctx, const anchor_packed & p, int S, int D, int k, int n_heads) {
    const int cpr = D / 4;
    anchor_tensors t;
    t.anchors      = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, D, k, 2, n_heads);
    t.anchor_of    = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, S, n_heads, 2);
    t.gamma        = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, S, n_heads, 2);
    t.slot_of      = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, S, n_heads, 2);
    t.k_codes      = ggml_new_tensor_3d(ctx, GGML_TYPE_I8,  cpr, p.n_K_max, n_heads);
    t.k_scales     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.n_K_max, n_heads);
    t.v_codes      = ggml_new_tensor_3d(ctx, GGML_TYPE_I8,  cpr, p.n_V_max, n_heads);
    t.v_scales     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.n_V_max, n_heads);
    return t;
}

static void anchor_set_tensors(ggml_backend_t backend, const anchor_tensors & t, const anchor_packed & p) {
    ggml_backend_tensor_set(t.anchors,      p.anchors.data(),   0, p.anchors.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(t.anchor_of,    p.anchor_of.data(), 0, p.anchor_of.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t.gamma,        p.gamma.data(),     0, p.gamma.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(t.slot_of,      p.slot_of.data(),   0, p.slot_of.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t.k_codes,      p.k_codes.data(),   0, p.k_codes.size());
    ggml_backend_tensor_set(t.k_scales,     p.k_scales.data(),  0, p.k_scales.size() * sizeof(float));
    ggml_backend_tensor_set(t.v_codes,      p.v_codes.data(),   0, p.v_codes.size());
    ggml_backend_tensor_set(t.v_scales,     p.v_scales.data(),  0, p.v_scales.size() * sizeof(float));
}

static int test_shared_scratch_order() {
    const int S = 33;
    const int D = 128;
    const int n_heads = 4;
    const int n_embd = n_heads * D; // 512

    anchor_kv_params params;
    params.theta   = 0.54f;
    params.W       = 16;
    params.k_frac  = 16;
    params.rho     = 0.5f;
    params.kappa   = 8;
    params.compress_k = true;
    params.compress_v = true;

    layer_data A = make_layer_data(12345, S, D, n_heads, params); // "layer" A (prefill)
    layer_data B = make_layer_data(98765, S, D, n_heads, params); // "layer" B (next layer)
    const int k = 16; // same for both (params.W dominates these short S)

    ggml_init_params iparams = {
        /*.mem_size   =*/ 512 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(iparams);
    if (!ctx) {
        fprintf(stderr, "FAIL: shared-scratch ggml_init failed\n");
        return 1;
    }

    anchor_tensors tA = anchor_make_tensors(ctx, A.packed, S, D, k, n_heads);
    anchor_tensors tB = anchor_make_tensors(ctx, B.packed, S, D, k, n_heads);
    ggml_tensor * scratch = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, S);
    ggml_tensor * readA   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, S);
    ggml_tensor * readB   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, S);

    ggml_backend_ptr backend;
    backend.ptr = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend.get()) {
        fprintf(stderr, "FAIL: shared-scratch CPU backend init failed\n");
        return 1;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend.get());

    ggml_backend_buffer_ptr buf;
    buf.ptr = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf.get()) {
        fprintf(stderr, "FAIL: shared-scratch tensor alloc failed\n");
        return 1;
    }

    anchor_set_tensors(backend.get(), tA, A.packed);
    anchor_set_tensors(backend.get(), tB, B.packed);
    std::vector<uint8_t> zeros((size_t) n_embd * S * 2, 0);
    ggml_backend_tensor_set(scratch, zeros.data(), 0, zeros.size());
    ggml_backend_tensor_set(readA,   zeros.data(), 0, zeros.size());
    ggml_backend_tensor_set(readB,   zeros.data(), 0, zeros.size());

    // build one graph containing both decompress ops writing the SAME scratch:
    //   opA   -> writes scratch (layer A data)
    //   cpyA  -> copies scratch into readA (materializes A before B overwrites)
    //   opB   -> writes scratch again, prev_dep = cpyA (B must wait for A's read)
    //   cpyB  -> copies scratch into readB (layer B data)
    // is_k=false (V side) so no RoPE -- pure reconstruction ordering check
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * opA = ggml_anchor_decompress(
        ctx, scratch,
        tA.anchors, tA.anchor_of, tA.gamma, tA.slot_of,
        tA.k_codes, tA.k_scales, tA.v_codes, tA.v_scales,
        /* is_k = */ false, /* prev_dep = */ nullptr,
        /* n_rot = */ 0, /* freq_base = */ 10000.0f, /* freq_scale = */ 1.0f);
    ggml_tensor * cpyA = ggml_cpy(ctx, opA, readA);
    ggml_tensor * opB = ggml_anchor_decompress(
        ctx, scratch,
        tB.anchors, tB.anchor_of, tB.gamma, tB.slot_of,
        tB.k_codes, tB.k_scales, tB.v_codes, tB.v_scales,
        /* is_k = */ false, /* prev_dep = */ cpyA,
        /* n_rot = */ 0, /* freq_base = */ 10000.0f, /* freq_scale = */ 1.0f);
    ggml_tensor * cpyB = ggml_cpy(ctx, opB, readB);
    ggml_build_forward_expand(gf, cpyB);

    if (ggml_backend_graph_compute(backend.get(), gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: shared-scratch graph compute failed\n");
        return 1;
    }

    // read back the two materialized buffers
    std::vector<ggml_fp16_t> rawA((size_t) n_embd * S);
    std::vector<ggml_fp16_t> rawB((size_t) n_embd * S);
    ggml_backend_tensor_get(readA, rawA.data(), 0, rawA.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(readB, rawB.data(), 0, rawB.size() * sizeof(ggml_fp16_t));
    std::vector<float> fA((size_t) n_embd * S), fB((size_t) n_embd * S);
    for (size_t i = 0; i < fA.size(); i++) {
        fA[i] = ggml_fp16_to_fp32(rawA[i]);
        fB[i] = ggml_fp16_to_fp32(rawB[i]);
    }

    const size_t total = (size_t) n_embd * S;
    float mse_A = mse(fA.data(), A.ref_v_cm.data(), total);
    float mse_B = mse(fB.data(), B.ref_v_cm.data(), total);
    float mse_A_vs_B = mse(fA.data(), B.ref_v_cm.data(), total); // if A was overwritten by B
    float max_A = max_abs(fA.data(), A.ref_v_cm.data(), total);
    float max_B = max_abs(fB.data(), B.ref_v_cm.data(), total);

    printf("shared-scratch: readA vs A mse=%.3e max=%.3e | readA vs B mse=%.3e (overwrite check)\n",
           mse_A, max_A, mse_A_vs_B);
    printf("shared-scratch: readB vs B mse=%.3e max=%.3e\n", mse_B, max_B);

    int fail = 0;
    if (mse_A > 1e-2f || max_A > 1e-1f) {
        fprintf(stderr, "FAIL: shared-scratch readA does not match layer A (mse=%.3e, max=%.3e)\n", mse_A, max_A);
        if (mse_A_vs_B < mse_A) {
            fprintf(stderr, "  readA looks like layer B's data -- the chain ordering let layer B overwrite layer A before its read.\n");
        }
        print_first_divergence(fA.data(), A.ref_v_cm.data(), total, "scA", n_embd, D);
        fail = 1;
    }
    if (mse_B > 1e-2f || max_B > 1e-1f) {
        fprintf(stderr, "FAIL: shared-scratch readB does not match layer B (mse=%.3e, max=%.3e)\n", mse_B, max_B);
        print_first_divergence(fB.data(), B.ref_v_cm.data(), total, "scB", n_embd, D);
        fail = 1;
    }
    if (!fail) {
        printf("PASS: shared-scratch chain ordering correct (A read before B overwrite)\n");
    }

    ggml_free(ctx);
    return fail;
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

    // shared-scratch multi-layer ordering check
    const int shared_fail = test_shared_scratch_order();

    return (pass ? 0 : 1) | shared_fail;
}
