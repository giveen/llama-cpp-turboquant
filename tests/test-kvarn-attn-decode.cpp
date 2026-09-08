// Correctness check for GGML_OP_KVARN_ATTN_DECODE: builds a synthetic
// sealed+tail cache (multiple sealed groups + partial tail, 2 KV heads, 4x GQA
// broadcast), runs the fused decode-attention op on both CPU and any
// available GPU backend, and compares against a plain dense-attention
// reference computed directly from the known ground-truth (pre-rotation)
// K/V vectors - the same standard the rest of the kvarn test suite holds.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

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

// Plain dense reference attention (no kvarn involved) from ground truth.
static void reference_attention(
        const float * q, const std::vector<float> & K, const std::vector<float> & V,
        int n_total, float kq_scale, float * out) {
    double running_max = -1e30, running_sum = 0.0;
    std::vector<double> acc(128, 0.0);
    for (int t = 0; t < n_total; t++) {
        double dot = 0.0;
        for (int c = 0; c < 128; c++) dot += (double) q[c] * K[(size_t) t * 128 + c];
        const double score = dot * kq_scale;
        const double new_max = std::max(running_max, score);
        const double rescale = exp(running_max - new_max);
        const double w = exp(score - new_max);
        for (int c = 0; c < 128; c++) acc[c] = acc[c] * rescale + w * V[(size_t) t * 128 + c];
        running_sum = running_sum * rescale + w;
        running_max = new_max;
    }
    for (int c = 0; c < 128; c++) out[c] = (float) (acc[c] / std::max(running_sum, 1e-20));
}

static bool run_on_backend(ggml_backend_t backend, int bits, int n_group_bc, int n_q, double * cos_out) {
    const int head_dim   = 128;
    const int n_head_kv  = 2;
    const int n_head_q   = n_head_kv * n_group_bc;
    const int n_sealed_g = 9;
    const int tail_count = 44;
    const int n_total    = n_sealed_g * 128 + tail_count;

    struct ggml_init_params params = { (size_t) 16 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * k_tail  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 128, n_head_kv);
    struct ggml_tensor * v_tail  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 128, n_head_kv);
    struct ggml_tensor * q       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_head_q, n_q);

    std::vector<struct ggml_tensor *> sealed_k_inputs;
    std::vector<struct ggml_tensor *> sealed_v_inputs;
    struct ggml_tensor * combined_sealed = nullptr;
    for (int h = 0; h < n_head_kv; h++) {
        struct ggml_tensor * head_sealed = nullptr;
        for (int g = 0; g < n_sealed_g; g++) {
            struct ggml_tensor * k_group = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, 128);
            struct ggml_tensor * v_group = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, 128);
            sealed_k_inputs.push_back(k_group);
            sealed_v_inputs.push_back(v_group);
            struct ggml_tensor * sealed_h = ggml_kvarn_seal(ctx, k_group, v_group, bits, bits, 16);
            struct ggml_tensor * sealed_h_3d = ggml_reshape_3d(ctx, sealed_h, sealed_h->ne[0], 1, 1);
            head_sealed = g == 0 ? sealed_h_3d : ggml_concat(ctx, head_sealed, sealed_h_3d, 1);
        }
        head_sealed = ggml_reshape_3d(ctx, head_sealed, head_sealed->ne[0], n_sealed_g, 1);
        combined_sealed = h == 0 ? head_sealed : ggml_concat(ctx, combined_sealed, head_sealed, 2);
    }
    combined_sealed = ggml_reshape_3d(ctx, combined_sealed, combined_sealed->ne[0], n_sealed_g, n_head_kv);

    const float kq_scale = 1.0f / sqrtf((float) head_dim);
    struct ggml_tensor * attn_out = ggml_kvarn_attn_decode(
            ctx, q, combined_sealed, k_tail, v_tail, bits, bits, n_head_kv, n_total, tail_count, kq_scale);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "alloc failed\n"); ggml_free(ctx); return false; }

    std::vector<float> q_rot((size_t) head_dim * n_head_q);
    std::vector<std::vector<float>> Kh(n_head_kv, std::vector<float>((size_t) n_total * head_dim));
    std::vector<std::vector<float>> Vh(n_head_kv, std::vector<float>((size_t) n_total * head_dim));
    std::vector<float> sealed_k_data((size_t) n_sealed_g * n_head_kv * 128 * head_dim);
    std::vector<float> sealed_v_data((size_t) n_sealed_g * n_head_kv * 128 * head_dim);
    std::vector<float> k_tail_data((size_t) n_head_kv * 128 * head_dim, 0.0f);
    std::vector<float> v_tail_data((size_t) n_head_kv * 128 * head_dim, 0.0f);

    for (int h = 0; h < n_head_kv; h++) {
        for (int g = 0; g < n_sealed_g; g++) {
            for (int t = 0; t < 128; t++) {
                const int gt = g * 128 + t;
                float k[128], v[128];
                make_token_vec(k, head_dim, (unsigned) gt, (unsigned) (h * 10000));
                make_token_vec(v, head_dim, (unsigned) gt, (unsigned) (h * 10000 + 1000));
                memcpy(Kh[h].data() + (size_t) gt * head_dim, k, head_dim * sizeof(float));
                memcpy(Vh[h].data() + (size_t) gt * head_dim, v, head_dim * sizeof(float));
                hadamard_128(k); hadamard_128(v);
                memcpy(sealed_k_data.data() + ((size_t) g * n_head_kv + h) * 128 * head_dim + (size_t) t * head_dim,
                       k, head_dim * sizeof(float));
                memcpy(sealed_v_data.data() + ((size_t) g * n_head_kv + h) * 128 * head_dim + (size_t) t * head_dim,
                       v, head_dim * sizeof(float));
            }
        }
        for (int t = 0; t < tail_count; t++) {
            const int gt = n_sealed_g * 128 + t;
            float k[128], v[128];
            make_token_vec(k, head_dim, (unsigned) gt, (unsigned) (h * 10000));
            make_token_vec(v, head_dim, (unsigned) gt, (unsigned) (h * 10000 + 1000));
            memcpy(Kh[h].data() + (size_t) gt * head_dim, k, head_dim * sizeof(float));
            memcpy(Vh[h].data() + (size_t) gt * head_dim, v, head_dim * sizeof(float));
            hadamard_128(k); hadamard_128(v);
            memcpy(k_tail_data.data() + ((size_t) h * 128 + t) * head_dim, k, head_dim * sizeof(float));
            memcpy(v_tail_data.data() + ((size_t) h * 128 + t) * head_dim, v, head_dim * sizeof(float));
        }
    }

    for (size_t i = 0; i < sealed_k_inputs.size(); i++) {
        const size_t head = i / n_sealed_g;
        const size_t group = i % n_sealed_g;
        const size_t group_head = head * n_sealed_g + group;
        ggml_backend_tensor_set(sealed_k_inputs[i],
                sealed_k_data.data() + group_head * 128 * head_dim, 0, ggml_nbytes(sealed_k_inputs[i]));
        ggml_backend_tensor_set(sealed_v_inputs[i],
                sealed_v_data.data() + group_head * 128 * head_dim, 0, ggml_nbytes(sealed_v_inputs[i]));
    }

    q_rot.resize((size_t) head_dim * n_head_q * n_q);
    for (int qi = 0; qi < n_q; qi++) {
      for (int qh = 0; qh < n_head_q; qh++) {
        float q_raw[128];
        make_token_vec(q_raw, head_dim, 999u + (unsigned) qi, (unsigned) (qh * 777));
        float q_r[128];
        memcpy(q_r, q_raw, sizeof(q_r));
        hadamard_128(q_r);
        memcpy(q_rot.data() + ((size_t) qi * n_head_q + qh) * head_dim, q_r, head_dim * sizeof(float));
      }
    }

    ggml_backend_tensor_set(k_tail,  k_tail_data.data(),  0, ggml_nbytes(k_tail));
    ggml_backend_tensor_set(v_tail,  v_tail_data.data(),  0, ggml_nbytes(v_tail));
    ggml_backend_tensor_set(q,       q_rot.data(),        0, ggml_nbytes(q));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, attn_out);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> out((size_t) head_dim * n_head_q * n_q);
    ggml_backend_tensor_get(attn_out, out.data(), 0, out.size() * sizeof(float));

    // Un-rotate output, compare against dense reference per Q head and row.
    // Row qi sees keys 0..pos_begin+qi with pos_begin = n_total - n_q.
    double min_cos = 1.0;
    for (int qi = 0; qi < n_q; qi++) {
      const int key_lim = n_total - n_q + 1 + qi;
      for (int qh = 0; qh < n_head_q; qh++) {
        const int kv_h = qh / n_group_bc;
        float * out_h = out.data() + ((size_t) qi * n_head_q + qh) * head_dim;
        hadamard_128(out_h);

        float q_raw[128];
        make_token_vec(q_raw, head_dim, 999u + (unsigned) qi, (unsigned) (qh * 777));
        float ref[128];
        reference_attention(q_raw, Kh[kv_h], Vh[kv_h], key_lim, kq_scale, ref);

        const double c = cosine(ref, out_h, head_dim);
        min_cos = std::min(min_cos, c);
      }
    }

    *cos_out = min_cos;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

int main() {
    printf("=== KVarN Fused Decode-Attention Op Correctness Test ===\n\n");
    int failures = 0;

    // CPU
    {
        ggml_backend_t cpu = ggml_backend_cpu_init();
      for (int n_group_bc : {4, 6}) {
      for (int n_q : {1, 5}) {
        for (int bits : {2, 3, 4, 5, 6}) {
            double cos_val = 0.0;
            if (!run_on_backend(cpu, bits, n_group_bc, n_q, &cos_val)) { failures++; continue; }
            printf("CPU  kvarn%d gqa%d nq%d: min cosine across heads/rows = %.6f\n", bits, n_group_bc, n_q, cos_val);
            // bits=2's floor is much lower than the plain tile round-trip
            // tests use: softmax amplifies the extra quantization noise in
            // the QK scores, so attention-level fidelity degrades faster
            // than per-tile MSE/cosine would suggest (measured ~0.78 here
            // vs. ~0.94 for a bare tile round-trip at the same bit width).
            const double floor =
                bits >= 6 ? 0.99 :
                bits >= 5 ? 0.97 :
                bits >= 4 ? 0.9  :
                bits >= 3 ? 0.95 : 0.7;
            if (cos_val < floor) { printf("  FAIL below floor %.2f\n", floor); failures++; }
        }
      }
      }
        ggml_backend_free(cpu);
    }

    // Any available GPU backend
    ggml_backend_load_all();
    ggml_backend_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = ggml_backend_dev_init(dev, nullptr);
            if (gpu) { printf("\nusing GPU backend: %s\n", ggml_backend_name(gpu)); break; }
        }
    }
    if (gpu) {
      for (int n_group_bc : {4, 6}) {
      for (int n_q : {1, 5}) {
        for (int bits : {2, 3, 4, 5, 6}) {
            double cos_val = 0.0;
            if (!run_on_backend(gpu, bits, n_group_bc, n_q, &cos_val)) { failures++; continue; }
            printf("GPU  kvarn%d gqa%d nq%d: min cosine across heads/rows = %.6f\n", bits, n_group_bc, n_q, cos_val);
            // bits=2's floor is much lower than the plain tile round-trip
            // tests use: softmax amplifies the extra quantization noise in
            // the QK scores, so attention-level fidelity degrades faster
            // than per-tile MSE/cosine would suggest (measured ~0.78 here
            // vs. ~0.94 for a bare tile round-trip at the same bit width).
            const double floor =
                bits >= 6 ? 0.99 :
                bits >= 5 ? 0.97 :
                bits >= 4 ? 0.9  :
                bits >= 3 ? 0.95 : 0.7;
            if (cos_val < floor) { printf("  FAIL below floor %.2f\n", floor); failures++; }
        }
      }
      }
        ggml_backend_free(gpu);
    } else {
        printf("\nno GPU backend available - skipping GPU check\n");
    }

    printf("\n");
    if (failures) { printf("=== FAILED: %d check(s) ===\n", failures); return 1; }
    printf("=== Done ===\n");
    return 0;
}
