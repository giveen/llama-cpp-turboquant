#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// Include ggml headers for block types
extern "C" {
#include "ggml-common.h"
#include "ggml-quants.h"
}

static float cosine_sim(const float * a, const float * b, int n) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return dot / (sqrtf(na)*sqrtf(nb) + 1e-10f);
}

int main(int argc, char **argv) {
    int S = 8192;
    int D = 128;

    FILE * fk = fopen("/tmp/anchor-kv-k.bin", "rb");
    FILE * fv = fopen("/tmp/anchor-kv-v.bin", "rb");
    if (!fk || !fv) { fprintf(stderr, "No KV files\n"); return 1; }
    std::vector<float> keys(S*D), values(S*D);
    size_t nk = fread(keys.data(), sizeof(float), S*D, fk);
    size_t nv = fread(values.data(), sizeof(float), S*D, fv);
    fclose(fk); fclose(fv);

    printf("=== Turbo4 Real Codec Test ===\n");
    printf("S=%d D=%d\n\n", S, D);

    // Turbo4 quantize + dequantize K
    int64_t total = S * D;
    int n_blocks = total / QK_TURBO4;
    printf("QK_TURBO4 = %d, n_blocks = %d\n", QK_TURBO4, n_blocks);
    printf("sizeof(block_turbo4_0) = %zu bytes\n", sizeof(block_turbo4_0));

    std::vector<block_turbo4_0> qk(n_blocks);
    std::vector<float> t4_k(S*D);

    printf("Quantizing K with turbo4...\n");
    quantize_row_turbo4_0_ref(keys.data(), qk.data(), total);
    printf("Dequantizing K...\n");
    dequantize_row_turbo4_0(qk.data(), t4_k.data(), total);

    // Turbo4 quantize + dequantize V
    std::vector<block_turbo4_0> qv(n_blocks);
    std::vector<float> t4_v(S*D);
    printf("Quantizing V with turbo4...\n");
    quantize_row_turbo4_0_ref(values.data(), qv.data(), total);
    printf("Dequantizing V...\n");
    dequantize_row_turbo4_0(qv.data(), t4_v.data(), total);

    // Memory
    size_t t4_size = sizeof(block_turbo4_0) * n_blocks * 2;
    size_t f16_size = S * D * 2 * 2;
    size_t q8_size = S * D * 1 * 2;

    printf("\nMemory (single head, K+V):\n");
    printf("  f16:     %8zu bytes  (1.0x)\n", f16_size);
    printf("  q8_0:    %8zu bytes  (2.0x)\n", q8_size);
    printf("  turbo4:  %8zu bytes  (%.1fx)\n", t4_size, (float)f16_size/t4_size);

    // KV quality
    float cos_k = cosine_sim(keys.data(), t4_k.data(), S*D);
    float cos_v = cosine_sim(values.data(), t4_v.data(), S*D);
    printf("\nKV reconstruction quality:\n");
    printf("  K cosine: %.6f\n", cos_k);
    printf("  V cosine: %.6f\n", cos_v);

    // Attention test
    printf("\nAttention quality (5 queries):\n");
    std::vector<float> queries(5*D);
    { uint64_t rng = 42;
      for (int i=0;i<5*D;i++) { rng=rng*6364136223846793005ULL+1442695040888963407ULL; queries[i]=(float)((int32_t)(rng>>32))/(float)(1ULL<<31); }
      for (int q=0;q<5;q++) { float n=0; for(int d=0;d<D;d++) n+=queries[q*D+d]*queries[q*D+d]; n=sqrtf(n); for(int d=0;d<D;d++) queries[q*D+d]/=n; }
    }

    auto run_attn = [&](const float * Q, const float * K, const float * V) -> std::vector<float> {
        std::vector<float> out(D, 0);
        float scale = 1.0f/sqrtf((float)D);
        std::vector<float> scores(S);
        for (int t=0;t<S;t++) { float dot=0; for(int d=0;d<D;d++) dot+=Q[d]*K[t*D+d]; scores[t]=dot*scale; }
        float mx=scores[0]; for(int t=1;t<S;t++) if(scores[t]>mx) mx=scores[t];
        float se=0; for(int t=0;t<S;t++) { scores[t]=expf(scores[t]-mx); se+=scores[t]; }
        float inv=1.0f/se; for(int t=0;t<S;t++) scores[t]*=inv;
        for(int t=0;t<S;t++) for(int d=0;d<D;d++) out[d]+=scores[t]*V[t*D+d];
        return out;
    };

    std::vector<std::vector<float>> dense_out(5);
    for (int q=0;q<5;q++) dense_out[q] = run_attn(&queries[q*D], keys.data(), values.data());

    printf("  turbo4: ");
    for (int q=0;q<5;q++) {
        auto out = run_attn(&queries[q*D], t4_k.data(), t4_v.data());
        printf("%.6f ", cosine_sim(dense_out[q].data(), out.data(), D));
    }
    printf("\n");

    return 0;
}
