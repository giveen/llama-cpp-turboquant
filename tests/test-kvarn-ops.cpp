// Exercises GGML_OP_KVARN_SEAL / GGML_OP_KVARN_MATERIALIZE through the real
// graph executor (ggml_graph_compute_with_ctx), not just as eager C++ calls -
// this is the gap test-kvarn-quant.c and test-kvarn-store.cpp both left open,
// since cpy_k/get_k in llama_kv_cache build graph nodes that run later on a
// backend device rather than executing immediately.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static void make_token_vec(float * v, int head_dim, unsigned token_idx, unsigned salt) {
    for (int i = 0; i < head_dim; i++) {
        v[i] = sinf((token_idx + salt) * 0.07f + i * 0.031f) * (1.0f + (float) (token_idx % 7));
    }
}

// Plain (self-inverse) 128-point WHT, same math as kvarn_hadamard_128 in
// ggml-kvarn-quant.c, reimplemented here so the test doesn't need to link
// against it directly (this test builds fixtures via ordinary graph ops).
static void hadamard_128(float * x) {
    for (int stride = 1; stride < 128; stride *= 2) {
        for (int base = 0; base < 128; base += 2 * stride) {
            for (int i = 0; i < stride; i++) {
                float a = x[base + i], b = x[base + stride + i];
                x[base + i] = a + b;
                x[base + stride + i] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt_128;
}

static double cosine(const float * a, const float * b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double) a[i] * (double) b[i];
        na  += (double) a[i] * (double) a[i];
        nb  += (double) b[i] * (double) b[i];
    }
    return (na > 0.0 && nb > 0.0) ? dot / (sqrt(na) * sqrt(nb)) : 0.0;
}

int main() {
    const int head_dim   = 128;
    const int n_sealed_g = 1;
    const int tail_count = 44;
    const int n_total    = n_sealed_g * 128 + tail_count;

    printf("=== KVarN Custom Op Graph Test ===\n\n");
    int failures = 0;

    for (int bits : {2, 3, 4, 5, 6}) {
        struct ggml_init_params params = {
            /* .mem_size   = */ (size_t) 64 << 20,
            /* .mem_base   = */ NULL,
            /* .no_alloc   = */ false,
        };
        struct ggml_context * ctx = ggml_init(params);

        struct ggml_tensor * k_tail0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, 128);
        struct ggml_tensor * v_tail0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, 128);
        struct ggml_tensor * k_tail  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, 128);
        struct ggml_tensor * v_tail  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, 128);

        std::vector<float> orig_k((size_t) n_total * head_dim);
        std::vector<float> orig_v((size_t) n_total * head_dim);

        // First 128 tokens go into the group-0 tail buffer that gets sealed.
        for (int t = 0; t < 128; t++) {
            float k[128], v[128];
            make_token_vec(k, head_dim, (unsigned) t, 0);
            make_token_vec(v, head_dim, (unsigned) t, 1000);
            memcpy(orig_k.data() + (size_t) t * head_dim, k, head_dim * sizeof(float));
            memcpy(orig_v.data() + (size_t) t * head_dim, v, head_dim * sizeof(float));
            hadamard_128(k);
            hadamard_128(v);
            memcpy((float *) k_tail0->data + (size_t) t * head_dim, k, head_dim * sizeof(float));
            memcpy((float *) v_tail0->data + (size_t) t * head_dim, v, head_dim * sizeof(float));
        }

        // Remaining tokens (tail_count of them) go into the live tail tensor.
        memset(k_tail->data, 0, ggml_nbytes(k_tail));
        memset(v_tail->data, 0, ggml_nbytes(v_tail));
        for (int t = 0; t < tail_count; t++) {
            const int global_t = 128 + t;
            float k[128], v[128];
            make_token_vec(k, head_dim, (unsigned) global_t, 0);
            make_token_vec(v, head_dim, (unsigned) global_t, 1000);
            memcpy(orig_k.data() + (size_t) global_t * head_dim, k, head_dim * sizeof(float));
            memcpy(orig_v.data() + (size_t) global_t * head_dim, v, head_dim * sizeof(float));
            hadamard_128(k);
            hadamard_128(v);
            memcpy((float *) k_tail->data + (size_t) t * head_dim, k, head_dim * sizeof(float));
            memcpy((float *) v_tail->data + (size_t) t * head_dim, v, head_dim * sizeof(float));
        }

        // idxs: a stand-in for k_idxs/v_idxs - element [n_tokens-1]+1 must
        // equal n_total (the real content length), which materialize now
        // derives on-device instead of taking as an int param.
        struct ggml_tensor * idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
        ((int64_t *) idxs->data)[0] = n_total - 1;

        struct ggml_tensor * sealed = ggml_kvarn_seal(ctx, k_tail0, v_tail0, bits, bits, 16);
        struct ggml_tensor * mat_k  = ggml_kvarn_materialize(ctx, sealed, k_tail, idxs, bits, bits, /*is_v=*/0, n_total);
        struct ggml_tensor * mat_v  = ggml_kvarn_materialize(ctx, sealed, v_tail, idxs, bits, bits, /*is_v=*/1, n_total);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, sealed);
        ggml_build_forward_expand(gf, mat_k);
        ggml_build_forward_expand(gf, mat_v);
        ggml_graph_compute_with_ctx(ctx, gf, 4);

        std::vector<float> out_k((size_t) n_total * head_dim);
        std::vector<float> out_v((size_t) n_total * head_dim);
        memcpy(out_k.data(), mat_k->data, out_k.size() * sizeof(float));
        memcpy(out_v.data(), mat_v->data, out_v.size() * sizeof(float));

        for (int t = 0; t < n_total; t++) {
            hadamard_128(out_k.data() + (size_t) t * head_dim);
            hadamard_128(out_v.data() + (size_t) t * head_dim);
        }

        const double cos_k = cosine(orig_k.data(), out_k.data(), orig_k.size());
        const double cos_v = cosine(orig_v.data(), out_v.data(), orig_v.size());
        printf("kvarn%d graph op round-trip (n=%d, 1 sealed group + %d tail): K cosine=%.6f V cosine=%.6f\n",
                bits, n_total, tail_count, cos_k, cos_v);

        const double min_cosine =
            bits >= 6 ? 0.99 :
            bits >= 5 ? 0.97 :
            bits >= 4 ? 0.9  :
            bits >= 3 ? 0.95 : 0.85;
        if (cos_k < min_cosine || cos_v < min_cosine) {
            printf("kvarn%d: FAIL cosine below floor %.2f\n", bits, min_cosine);
            failures++;
        }

        ggml_free(ctx);
    }

    printf("\n");
    if (failures) {
        printf("=== FAILED: %d check(s) ===\n", failures);
        return 1;
    }
    printf("=== Done ===\n");
    return 0;
}
