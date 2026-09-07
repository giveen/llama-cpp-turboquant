// Exercises GGML_OP_KVARN_STORE through the real graph executor
// (ggml_graph_compute_with_ctx), the fixed-topology write op that replaced
// the old per-group host loop of GGML_OP_KVARN_SEAL/ggml_cpy nodes (see the
// KVarN graph-reuse plan doc). Unlike test-kvarn-ops.cpp (which builds
// already-sealed fixtures directly), this test starts from a NONZERO
// starting tail (tail_count0 > 0) and a single call spanning multiple
// 128-token groups, specifically to cover:
//   - reading the still-live old tail before it gets overwritten (only
//     relevant for the first group a call completes)
//   - sealing more than one group from a single op call (the whole point
//     of the fixed-topology redesign: one node, N groups)
//   - deriving position from `idxs`'s tensor VALUE rather than any
//     op_params baked at graph-build time
//   - more than one KV head, to catch head-stride bugs
// Read-back uses the already-verified GGML_OP_KVARN_MATERIALIZE as the
// oracle, so this test isolates STORE's own correctness.

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
    const int head_dim    = 128;
    const int n_head_kv   = 2; // exercise more than one head, not just head 0
    const int tail_count0 = 50;
    const int n_tokens    = 250;
    const int n_total     = tail_count0 + n_tokens; // 300: 2 groups sealed + 44 tail
    const int n_groups_max = 4;
    const int max_groups_per_call = n_tokens / 128 + 2;
    const int tail_count_end = n_total % 128;

    printf("=== KVarN GGML_OP_KVARN_STORE Graph Test ===\n\n");
    int failures = 0;

    for (int bits : {2, 3, 4, 5, 6}) {
        struct ggml_init_params params = {
            /* .mem_size   = */ (size_t) 64 << 20,
            /* .mem_base   = */ NULL,
            /* .no_alloc   = */ false,
        };

        // kvarn_make_layout/kvarn_tile_layout live in the internal
        // ggml-kvarn-quant.h this test doesn't link against directly - probe
        // the per-bit-width record size via a throwaway ggml_kvarn_seal call
        // instead (its output tensor's ne[0] is exactly tile_bytes).
        int64_t tile_bytes;
        {
            struct ggml_context * probe_ctx = ggml_init(params);
            struct ggml_tensor * probe_k = ggml_new_tensor_2d(probe_ctx, GGML_TYPE_F32, head_dim, 128);
            struct ggml_tensor * probe_v = ggml_new_tensor_2d(probe_ctx, GGML_TYPE_F32, head_dim, 128);
            struct ggml_tensor * probe_record = ggml_kvarn_seal(probe_ctx, probe_k, probe_v, bits, bits, 16);
            tile_bytes = probe_record->ne[0];
            ggml_free(probe_ctx);
        }

        struct ggml_context * ctx = ggml_init(params);

        // Reference (un-rotated) token vectors, per head/side, for the full
        // [0, n_total) position range.
        std::vector<std::vector<float>> orig_k(n_head_kv), orig_v(n_head_kv);
        for (int h = 0; h < n_head_kv; h++) {
            orig_k[h].resize((size_t) n_total * head_dim);
            orig_v[h].resize((size_t) n_total * head_dim);
            for (int t = 0; t < n_total; t++) {
                make_token_vec(orig_k[h].data() + (size_t) t * head_dim, head_dim, (unsigned) t, (unsigned) (h * 2000));
                make_token_vec(orig_v[h].data() + (size_t) t * head_dim, head_dim, (unsigned) t, (unsigned) (h * 2000 + 1000));
            }
        }

        // k_tail/v_tail: persistent [128, 128, n_head_kv] tensors, pre-populated
        // with the first tail_count0 positions' ROTATED data (simulating the
        // state left behind by an earlier call).
        struct ggml_tensor * k_tail = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 128, n_head_kv);
        struct ggml_tensor * v_tail = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 128, n_head_kv);
        memset(k_tail->data, 0, ggml_nbytes(k_tail));
        memset(v_tail->data, 0, ggml_nbytes(v_tail));
        for (int h = 0; h < n_head_kv; h++) {
            for (int t = 0; t < tail_count0; t++) {
                float k[128], v[128];
                memcpy(k, orig_k[h].data() + (size_t) t * head_dim, sizeof(k));
                memcpy(v, orig_v[h].data() + (size_t) t * head_dim, sizeof(v));
                hadamard_128(k);
                hadamard_128(v);
                memcpy((float *) k_tail->data + ((size_t) h * 128 + t) * head_dim, k, sizeof(k));
                memcpy((float *) v_tail->data + ((size_t) h * 128 + t) * head_dim, v, sizeof(v));
            }
        }

        // cur: [128, n_head_kv, n_tokens] - this call's new tokens (positions
        // tail_count0 .. n_total-1), ROTATED (matches what cpy_kvarn feeds
        // ggml_kvarn_store: already post-ggml_turbo_wht).
        struct ggml_tensor * cur_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_head_kv, n_tokens);
        struct ggml_tensor * cur_v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_head_kv, n_tokens);
        for (int h = 0; h < n_head_kv; h++) {
            for (int tok = 0; tok < n_tokens; tok++) {
                const int t = tail_count0 + tok;
                float k[128], v[128];
                memcpy(k, orig_k[h].data() + (size_t) t * head_dim, sizeof(k));
                memcpy(v, orig_v[h].data() + (size_t) t * head_dim, sizeof(v));
                hadamard_128(k);
                hadamard_128(v);
                memcpy((float *) cur_k->data + ((size_t) tok * n_head_kv + h) * head_dim, k, sizeof(k));
                memcpy((float *) cur_v->data + ((size_t) tok * n_head_kv + h) * head_dim, v, sizeof(v));
            }
        }

        // idxs: absolute position per token, tail_count0 .. n_total-1 - the
        // value ggml_kvarn_store's kernel reads at execution time (element 0
        // gives pos0), NOT anything baked into op_params.
        struct ggml_tensor * idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
        for (int tok = 0; tok < n_tokens; tok++) {
            ((int64_t *) idxs->data)[tok] = tail_count0 + tok;
        }

        struct ggml_tensor * sealed = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, tile_bytes, n_groups_max, n_head_kv);
        memset(sealed->data, 0, ggml_nbytes(sealed));

        struct ggml_tensor * store_k = ggml_kvarn_store(ctx, cur_k, idxs, k_tail, sealed,
                bits, bits, 16, /*is_v=*/0, max_groups_per_call);
        struct ggml_tensor * store_v = ggml_kvarn_store(ctx, cur_v, idxs, v_tail, sealed,
                bits, bits, 16, /*is_v=*/1, max_groups_per_call);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, store_k);
        ggml_build_forward_expand(gf, store_v);
        ggml_graph_compute_with_ctx(ctx, gf, 4);

        // Read back via the already-verified materialize op, per head.
        int local_failures = 0;
        for (int h = 0; h < n_head_kv; h++) {
            struct ggml_tensor * sealed_h = ggml_view_2d(ctx, sealed, tile_bytes, n_groups_max,
                    sealed->nb[1], (size_t) h * sealed->nb[2]);
            struct ggml_tensor * k_tail_h = ggml_view_2d(ctx, k_tail, head_dim, 128, k_tail->nb[1], (size_t) h * k_tail->nb[2]);
            struct ggml_tensor * v_tail_h = ggml_view_2d(ctx, v_tail, head_dim, 128, v_tail->nb[1], (size_t) h * v_tail->nb[2]);

            struct ggml_tensor * mat_k = ggml_kvarn_materialize(ctx, sealed_h, k_tail_h, idxs, bits, bits, 0, n_total);
            struct ggml_tensor * mat_v = ggml_kvarn_materialize(ctx, sealed_h, v_tail_h, idxs, bits, bits, 1, n_total);

            struct ggml_cgraph * gf2 = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf2, mat_k);
            ggml_build_forward_expand(gf2, mat_v);
            ggml_graph_compute_with_ctx(ctx, gf2, 4);

            std::vector<float> out_k((size_t) n_total * head_dim);
            std::vector<float> out_v((size_t) n_total * head_dim);
            memcpy(out_k.data(), mat_k->data, out_k.size() * sizeof(float));
            memcpy(out_v.data(), mat_v->data, out_v.size() * sizeof(float));
            for (int t = 0; t < n_total; t++) {
                hadamard_128(out_k.data() + (size_t) t * head_dim);
                hadamard_128(out_v.data() + (size_t) t * head_dim);
            }

            const double cos_k = cosine(orig_k[h].data(), out_k.data(), orig_k[h].size());
            const double cos_v = cosine(orig_v[h].data(), out_v.data(), orig_v[h].size());
            printf("kvarn%d head=%d STORE-op round-trip (n=%d, 2 groups sealed in ONE call + %d tail): K cosine=%.6f V cosine=%.6f\n",
                    bits, h, n_total, tail_count_end, cos_k, cos_v);

            const double min_cosine =
                bits >= 6 ? 0.99 :
                bits >= 5 ? 0.97 :
                bits >= 4 ? 0.9  :
                bits >= 3 ? 0.95 : 0.85;
            if (cos_k < min_cosine || cos_v < min_cosine) {
                printf("kvarn%d head=%d: FAIL cosine below floor %.2f\n", bits, h, min_cosine);
                local_failures++;
            }
        }
        failures += local_failures;

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
