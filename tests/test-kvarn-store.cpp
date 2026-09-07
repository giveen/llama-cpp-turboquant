// Exercises the KVarN group-staging state machine (llama_kvarn_layer_store)
// across the tail/seal boundary: append tokens one at a time (as they would
// arrive during decode), let groups seal at 128, then read back a range that
// spans multiple sealed groups plus a partial tail and check it reconstructs
// the original (pre-rotation) vectors.

#include "llama-kvarn-store.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static void make_token_vec(float * v, int head_dim, unsigned token_idx) {
    for (int i = 0; i < head_dim; i++) {
        v[i] = sinf(token_idx * 0.07f + i * 0.031f) * (1.0f + (float) (token_idx % 7));
    }
}

static void metrics(const float * a, const float * b, size_t n, double * mse, double * cosv) {
    double se = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double) a[i] - (double) b[i];
        se  += d * d;
        dot += (double) a[i] * (double) b[i];
        na  += (double) a[i] * (double) a[i];
        nb  += (double) b[i] * (double) b[i];
    }
    *mse  = se / n;
    *cosv = (na > 0.0 && nb > 0.0) ? dot / (sqrt(na) * sqrt(nb)) : 0.0;
}

int main() {
    const int head_dim = llama_kvarn_layer_store::HEAD_DIM;
    const int n_tokens  = 300; // 2 full sealed groups (256) + a 44-token partial tail

    printf("=== KVarN Layer Store Round-Trip Test ===\n\n");

    int failures = 0;

    for (int bits : {2, 3, 4, 5, 6}) {
        llama_kvarn_layer_store store(bits, bits);

        std::vector<float> orig_k((size_t) n_tokens * head_dim);
        std::vector<float> orig_v((size_t) n_tokens * head_dim);

        for (int t = 0; t < n_tokens; t++) {
            float k[128], v[128];
            make_token_vec(k, head_dim, (unsigned) t);
            make_token_vec(v, head_dim, (unsigned) t + 1000);
            memcpy(orig_k.data() + (size_t) t * head_dim, k, head_dim * sizeof(float));
            memcpy(orig_v.data() + (size_t) t * head_dim, v, head_dim * sizeof(float));

            kvarn_hadamard_128(k);
            kvarn_hadamard_128(v);
            const uint32_t idx = store.append(k, v);
            if (idx != (uint32_t) t) {
                printf("kvarn%d: append index mismatch: got %u expected %d\n", bits, idx, t);
                failures++;
            }
        }

        if (store.size() != (uint32_t) n_tokens) {
            printf("kvarn%d: size mismatch: got %u expected %d\n", bits, store.size(), n_tokens);
            failures++;
        }

        std::vector<float> out_k((size_t) n_tokens * head_dim);
        std::vector<float> out_v((size_t) n_tokens * head_dim);
        store.read((uint32_t) n_tokens, out_k.data(), out_v.data());

        // Undo the per-token rotation (self-inverse) before comparing.
        for (int t = 0; t < n_tokens; t++) {
            kvarn_hadamard_128(out_k.data() + (size_t) t * head_dim);
            kvarn_hadamard_128(out_v.data() + (size_t) t * head_dim);
        }

        double mse_k, cos_k, mse_v, cos_v;
        metrics(orig_k.data(), out_k.data(), orig_k.size(), &mse_k, &cos_k);
        metrics(orig_v.data(), out_v.data(), orig_v.size(), &mse_v, &cos_v);
        printf("kvarn%d store round-trip (n=%d, %u sealed groups): K cosine=%.6f V cosine=%.6f\n",
                bits, n_tokens, store.size() / llama_kvarn_layer_store::GROUP, cos_k, cos_v);

        const double min_cosine =
            bits >= 6 ? 0.99 :
            bits >= 5 ? 0.97 :
            bits >= 4 ? 0.9  :
            bits >= 3 ? 0.95 : 0.85;
        if (cos_k < min_cosine || cos_v < min_cosine) {
            printf("kvarn%d: FAIL cosine below floor %.2f\n", bits, min_cosine);
            failures++;
        }

        // A partial-tail-only read (before any group has sealed) must be exact.
        llama_kvarn_layer_store store2(bits, bits);
        float k0[128], v0[128], k0_check[128], v0_check[128];
        make_token_vec(k0, head_dim, 42);
        make_token_vec(v0, head_dim, 4242);
        memcpy(k0_check, k0, sizeof(k0));
        memcpy(v0_check, v0, sizeof(v0));
        kvarn_hadamard_128(k0);
        kvarn_hadamard_128(v0);
        store2.append(k0, v0);

        float out_k0[128], out_v0[128];
        store2.read(1, out_k0, out_v0);
        kvarn_hadamard_128(out_k0);
        kvarn_hadamard_128(out_v0);

        double mse0, cos0;
        metrics(k0_check, out_k0, head_dim, &mse0, &cos0);
        if (mse0 > 1e-6) {
            printf("kvarn%d: FAIL partial-tail-only read not exact (MSE=%.3g)\n", bits, mse0);
            failures++;
        }
    }

    printf("\n");
    if (failures) {
        printf("=== FAILED: %d check(s) ===\n", failures);
        return 1;
    }
    printf("=== Done ===\n");
    return 0;
}
