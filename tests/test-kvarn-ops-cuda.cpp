// Direct numeric check that GGML_OP_KVARN_SEAL/MATERIALIZE's CUDA (or any
// other available non-CPU backend) kernels agree with the CPU reference
// already verified by test-kvarn-ops.cpp - closes the gap that test only
// left open (it always runs through ggml_graph_compute_with_ctx, which is
// CPU-only). Skips (exit 0) if no non-CPU backend is available.

#include "ggml.h"
#include "ggml-backend.h"
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

static ggml_backend_t find_non_cpu_backend() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend) {
                printf("using backend: %s (%s)\n", ggml_backend_name(backend), ggml_backend_dev_name(dev));
                return backend;
            }
        }
    }
    return nullptr;
}

int main() {
    printf("=== KVarN CUDA/GPU-Backend Op Numeric Check ===\n\n");

    ggml_backend_t backend = find_non_cpu_backend();
    if (!backend) {
        printf("no non-CPU backend available - skipping\n");
        return 0;
    }

    const int head_dim   = 128;
    const int n_sealed_g = 1;
    const int tail_count = 44;
    const int n_total    = n_sealed_g * 128 + tail_count;

    int failures = 0;

    for (int bits : {2, 3, 4, 5, 6}) {
        // Build the graph in a no_alloc context, backed by the real backend.
        struct ggml_init_params params = {
            /* .mem_size   = */ (size_t) 16 * 1024 * 1024,
            /* .mem_base   = */ NULL,
            /* .no_alloc   = */ true,
        };
        struct ggml_context * ctx = ggml_init(params);

        struct ggml_tensor * k_tail0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, 128);
        struct ggml_tensor * v_tail0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, 128);
        struct ggml_tensor * k_tail  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, 128);
        struct ggml_tensor * v_tail  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, 128);
        struct ggml_tensor * idxs    = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1); // element [0]+1 must equal n_total

        struct ggml_tensor * sealed = ggml_kvarn_seal(ctx, k_tail0, v_tail0, bits, bits, 16);
        struct ggml_tensor * mat_k  = ggml_kvarn_materialize(ctx, sealed, k_tail, idxs, bits, bits, /*is_v=*/0, n_total);
        struct ggml_tensor * mat_v  = ggml_kvarn_materialize(ctx, sealed, v_tail, idxs, bits, bits, /*is_v=*/1, n_total);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            printf("kvarn%d: failed to allocate backend buffer\n", bits);
            failures++;
            ggml_free(ctx);
            continue;
        }

        std::vector<float> orig_k((size_t) n_total * head_dim);
        std::vector<float> orig_v((size_t) n_total * head_dim);
        std::vector<float> k_tail0_data((size_t) 128 * head_dim);
        std::vector<float> v_tail0_data((size_t) 128 * head_dim);
        std::vector<float> k_tail_data((size_t) 128 * head_dim, 0.0f);
        std::vector<float> v_tail_data((size_t) 128 * head_dim, 0.0f);

        for (int t = 0; t < 128; t++) {
            float k[128], v[128];
            make_token_vec(k, head_dim, (unsigned) t, 0);
            make_token_vec(v, head_dim, (unsigned) t, 1000);
            memcpy(orig_k.data() + (size_t) t * head_dim, k, head_dim * sizeof(float));
            memcpy(orig_v.data() + (size_t) t * head_dim, v, head_dim * sizeof(float));
            hadamard_128(k);
            hadamard_128(v);
            memcpy(k_tail0_data.data() + (size_t) t * head_dim, k, head_dim * sizeof(float));
            memcpy(v_tail0_data.data() + (size_t) t * head_dim, v, head_dim * sizeof(float));
        }
        for (int t = 0; t < tail_count; t++) {
            const int global_t = 128 + t;
            float k[128], v[128];
            make_token_vec(k, head_dim, (unsigned) global_t, 0);
            make_token_vec(v, head_dim, (unsigned) global_t, 1000);
            memcpy(orig_k.data() + (size_t) global_t * head_dim, k, head_dim * sizeof(float));
            memcpy(orig_v.data() + (size_t) global_t * head_dim, v, head_dim * sizeof(float));
            hadamard_128(k);
            hadamard_128(v);
            memcpy(k_tail_data.data() + (size_t) t * head_dim, k, head_dim * sizeof(float));
            memcpy(v_tail_data.data() + (size_t) t * head_dim, v, head_dim * sizeof(float));
        }

        ggml_backend_tensor_set(k_tail0, k_tail0_data.data(), 0, ggml_nbytes(k_tail0));
        ggml_backend_tensor_set(v_tail0, v_tail0_data.data(), 0, ggml_nbytes(v_tail0));
        ggml_backend_tensor_set(k_tail,  k_tail_data.data(),  0, ggml_nbytes(k_tail));
        ggml_backend_tensor_set(v_tail,  v_tail_data.data(),  0, ggml_nbytes(v_tail));
        const int64_t idxs_val = n_total - 1;
        ggml_backend_tensor_set(idxs, &idxs_val, 0, sizeof(idxs_val));

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, sealed);
        ggml_build_forward_expand(gf, mat_k);
        ggml_build_forward_expand(gf, mat_v);

        ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            printf("kvarn%d: backend graph compute failed (status=%d)\n", bits, (int) status);
            failures++;
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            continue;
        }

        std::vector<float> out_k((size_t) n_total * head_dim);
        std::vector<float> out_v((size_t) n_total * head_dim);
        ggml_backend_tensor_get(mat_k, out_k.data(), 0, out_k.size() * sizeof(float));
        ggml_backend_tensor_get(mat_v, out_v.data(), 0, out_v.size() * sizeof(float));

        for (int t = 0; t < n_total; t++) {
            hadamard_128(out_k.data() + (size_t) t * head_dim);
            hadamard_128(out_v.data() + (size_t) t * head_dim);
        }

        const double cos_k = cosine(orig_k.data(), out_k.data(), orig_k.size());
        const double cos_v = cosine(orig_v.data(), out_v.data(), orig_v.size());
        printf("kvarn%d %s round-trip (n=%d, 1 sealed group + %d tail): K cosine=%.6f V cosine=%.6f\n",
                bits, ggml_backend_name(backend), n_total, tail_count, cos_k, cos_v);

        const double min_cosine =
            bits >= 6 ? 0.99 :
            bits >= 5 ? 0.97 :
            bits >= 4 ? 0.9  :
            bits >= 3 ? 0.95 : 0.85;
        if (cos_k < min_cosine || cos_v < min_cosine) {
            printf("kvarn%d: FAIL cosine below floor %.2f\n", bits, min_cosine);
            failures++;
        }

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    ggml_backend_free(backend);

    printf("\n");
    if (failures) {
        printf("=== FAILED: %d check(s) ===\n", failures);
        return 1;
    }
    printf("=== Done ===\n");
    return 0;
}
