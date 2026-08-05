// OSCAR rotation calibration tool.
//
// Loads a GGUF model, runs forward passes on calibration text, and uses the
// eval callback to intercept Q, K, V, and attention-output tensors at runtime.
// Accumulates per-layer Q-covariance (Q^T Q) and score-weighted V-covariance
// (V^T diag(w) V) online, then dumps them as raw float32 binaries for the
// Python orchestrator (calibrate_rotation.py) to eigendecompose.
//
// Usage:
//   llama-oscar-calib -m model.gguf -f calibration.txt -o covariances/
//
// Output: covariances/layer_N_qcov.bin and layer_N_vcov.bin (raw f32, row-major).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <direct.h>
#else
#  include <sys/stat.h>
#endif

// Accumulator for one layer's Q^T Q and V^T diag(w) V matrices.
// Sized for head_dim up to 512 (the max oscar2 supports).
struct calib_accumulator {
    int n_embd_head;
    int n_head;
    int n_head_kv;
    int gqa_ratio;

    // Q-covariance: n_embd_head x n_embd_head, accumulated across all KV heads.
    // For GQA, query heads are grouped by KV head, so we accumulate per-KV-head.
    std::vector<std::vector<float>> q_cov;  // [n_head_kv][n_embd_head * n_embd_head]

    // V-covariance: n_embd_head x n_embd_head per KV head.
    std::vector<std::vector<float>> v_cov;  // [n_head_kv][n_embd_head * n_embd_head]

    // Token count for normalization.
    int64_t n_tokens;

    calib_accumulator() : n_embd_head(0), n_head(0), n_head_kv(0), gqa_ratio(1), n_tokens(0) {}

    void init(int hd, int nh, int nhkv) {
        n_embd_head = hd;
        n_head = nh;
        n_head_kv = nhkv;
        gqa_ratio = nh / nhkv;
        n_tokens = 0;
        // assign() replaces every element: resize() would keep existing inner
        // vectors at their old size when hd changes (head dim flips on
        // mixed-head models), leaving q_cov/v_cov undersized -> OOB writes.
        q_cov.assign(nhkv, std::vector<float>((size_t) hd * hd, 0.0f));
        v_cov.assign(nhkv, std::vector<float>((size_t) hd * hd, 0.0f));
    }

    // Accumulate Q^T Q for one KV head. Q is [gqa_ratio, n_embd_head] in row-major.
    void acc_q_cov(int kv_head, const float * Q_row, int stride_q) {
        // Q_row points to the start of the gqa_ratio query heads for this KV head.
        // stride_q is the stride between consecutive query heads (in floats).
        const int g = gqa_ratio;
        const int d = n_embd_head;
        auto & cov = q_cov[kv_head];
        for (int gi = 0; gi < g; ++gi) {
            const float * q = Q_row + gi * stride_q;
            // rank-1 update: cov += q * q^T
            for (int r = 0; r < d; ++r) {
                for (int c = 0; c < d; ++c) {
                    cov[r * d + c] += q[r] * q[c];
                }
            }
        }
    }

    // Accumulate V^T diag(w) V for one KV head. V is [n_embd_head], w is scalar.
    void acc_v_cov(int kv_head, const float * V_row, float weight) {
        const int d = n_embd_head;
        auto & cov = v_cov[kv_head];
        for (int r = 0; r < d; ++r) {
            for (int c = 0; c < d; ++c) {
                cov[r * d + c] += weight * V_row[r] * V_row[c];
            }
        }
    }
};

// Per-layer accumulators (one per transformer layer).
static std::vector<calib_accumulator> g_accumulators;
static int g_n_layers = 0;
static int g_n_embd_head = 0;
static std::string g_output_dir;

// The eval callback. Called for each node during graph evaluation.
// We check tensor names to identify Q and V nodes.
static bool calib_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);
    if (ask) {
        // Return true if we want to read this tensor's data after evaluation.
        const char * name = ggml_get_name(t);
        if (!name || name[0] == '\0') return false;
        // Fork graph naming (llama-graph.cpp cb()/graph_get_cb): tensors are
        // named "{name}-{layer}", e.g. "Qcur-0". We need:
        //   Qcur: post-RoPE Q (before any calibrated rotation) - real graph node
        //   Vcur: pre-rotation V - a real node only when Q/K/V projections are
        //         separate; with a fused attn_qkv the Q/K/V are views of the
        //         "wqkv" node, so V is extracted from wqkv instead.
        // There is no separately named softmax tensor (fused FA), so V
        // accumulates unweighted.
        std::string s(name);
        const size_t dash = s.rfind('-');
        if (dash == std::string::npos) return false;
        const std::string prefix = s.substr(0, dash);
        if (prefix == "Qcur" || prefix == "Vcur" || prefix == "wqkv") {
            return true;
        }
        return false;
    }

    // Process: tensor data is available now. NOTE: the scheduler stops
    // evaluating the graph if this callback returns false, so every path here
    // must return true.
    const char * name = ggml_get_name(t);
    if (!name || name[0] == '\0') return true;
    std::string s(name);

    // Parse layer index and tensor name from "{prefix}-{layer}" (fork naming).
    const size_t dash = s.rfind('-');
    if (dash == std::string::npos) return true;
    const int layer = std::atoi(s.c_str() + dash + 1);
    const std::string suffix = s.substr(0, dash);
    if (layer < 0 || layer >= g_n_layers) return true;

    auto & acc = g_accumulators[layer];

    // We only process tensors of the right type and shape.
    const enum ggml_type type = t->type;
    if (type != GGML_TYPE_F32) return true;

    // The tensor may live in device memory (CUDA compute buffer): reading
    // t->data directly from host segfaults. Copy through the backend instead.
    const size_t nbytes = ggml_nbytes(t);
    if (nbytes == 0) return true;
    std::vector<float> host(ggml_nelements(t));
    ggml_backend_tensor_get(t, host.data(), 0, nbytes);
    const float * data = host.data();

    const int ne0 = ggml_nelements(t) > 0 ? t->ne[0] : 0;
    const int ne1 = t->ne[1];
    if (ne1 == 0) return true;


    const int nh   = acc.n_head;
    const int nhkv = acc.n_head_kv;
    const int g    = acc.gqa_ratio;

    // Auto-detect n_embd_head from the tensor shape on first encounter.
    // n_embd/n_head is unreliable for models with shared KV layers (e.g. Gemma4).
    // Qcur can be 2D [d*nh, n_tokens] (projection output) or 3D [d, nh, n_tokens]
    // (post-RoPE). Fused qkv is [d*(nh + 2*nhkv), n_tokens].
    int d = acc.n_embd_head;
    int d_candidate = -1;
    if (suffix == "Qcur") {
        if (ne1 == nh && ne0 >= 32) {
            d_candidate = ne0;      // 3D [d, nh, n_tokens]
        } else if (ne0 % nh == 0 && ne0 / nh >= 32) {
            d_candidate = ne0 / nh; // 2D [d*nh, n_tokens]
        }
    } else if (suffix == "wqkv" && ne0 % (nh + 2*nhkv) == 0) {
        d_candidate = ne0 / (nh + 2*nhkv);
    }
    if (d_candidate > 0 && d_candidate != d) {
        d = d_candidate;
        acc.n_embd_head = d;
        g_n_embd_head = d;
        acc.init(d, nh, nhkv);
        LOG_INF("layer %d: auto-detected n_embd_head=%d (from %s tensor)\n",
                layer, d, suffix.c_str());
    }
    if (d == 0) return true;

    // accumulate Q^T Q over the g query heads per KV head
    // data_row points at one token's Q block of ne0_q = d*nh floats (row-major
    // [d*nh, n_tokens]); Q heads are grouped: KV head kvh owns heads [kvh*g, (kvh+1)*g).
    const auto acc_q_tok = [&](const float * data_row, int ne0_q) {
        for (int kvh = 0; kvh < nhkv; ++kvh) {
            acc.acc_q_cov(kvh, data_row + kvh * g * d, d);
        }
        (void) ne0_q;
    };

    // accumulate V^T V over the KV heads; data_row points at one token's V block
    // of ne0_v = d*nhkv floats, each KV head's V at stride d.
    const auto acc_v_tok = [&](const float * data_row, int ne0_v) {
        const int n_heads = ne0_v / d;
        for (int kvh = 0; kvh < nhkv && kvh < n_heads; ++kvh) {
            acc.acc_v_cov(kvh, data_row + kvh * d, 1.0f);
        }
    };

    if (suffix == "Qcur") {
        // [d*nh, n_tokens] - each column is one token's flattened Q.
        // (also accept the [d, nh, n_tokens] 3D layout)
        int n_tok;
        if (ne0 == d * nh) {
            n_tok = ne1;
            for (int tok = 0; tok < n_tok; ++tok) {
                acc_q_tok(data + (size_t) tok * d * nh, d * nh);
            }
        } else if (ne0 == d && ne1 == nh) {
            n_tok = (int) (ggml_nbytes(t) / sizeof(float)) / (d * nh);
            for (int tok = 0; tok < n_tok; ++tok) {
                acc_q_tok(data + (size_t) tok * nh * d, d * nh);
            }
        } else {
            return true;
        }
        acc.n_tokens += n_tok;
    } else if (suffix == "Vcur") {
        // Separate-projection V: [d*nhkv, n_tokens] (or [d, nhkv, n_tokens]).
        int n_tok;
        if (ne0 == d * nhkv) {
            n_tok = ne1;
            for (int tok = 0; tok < n_tok; ++tok) {
                acc_v_tok(data + (size_t) tok * d * nhkv, d * nhkv);
            }
        } else if (ne0 == d && ne1 == nhkv) {
            n_tok = (int) (ggml_nbytes(t) / sizeof(float)) / (d * nhkv);
            for (int tok = 0; tok < n_tok; ++tok) {
                acc_v_tok(data + (size_t) tok * nhkv * d, d * nhkv);
            }
        } else {
            return true;
        }
    } else if (suffix == "wqkv") {
        // Fused projection: [d*(nh + 2*nhkv), n_tokens]; Q at rows [0, d*nh),
        // V at rows [d*(nh+nhkv), d*(nh+2*nhkv)). No token-count bump for V
        // (n_tokens already counted from Q).
        if (ne0 != d * (nh + 2*nhkv)) return true;
        const int n_tok = ne1;
        const int q_rows = d * nh;
        const int v_off  = d * (nh + nhkv);
        for (int tok = 0; tok < n_tok; ++tok) {
            acc_q_tok(data + (size_t) tok * ne0, q_rows);
            acc_v_tok(data + (size_t) tok * ne0 + v_off, d * nhkv);
        }
        acc.n_tokens += n_tok;
    }

    return true;
}

static void dump_covariances(const std::string & output_dir) {
    for (int layer = 0; layer < g_n_layers; ++layer) {
        const auto & acc = g_accumulators[layer];
        const int d = acc.n_embd_head;
        const int nhkv = acc.n_head_kv;

        // Average across KV heads and normalize by token count.
        // For simplicity, average the covariance matrices across KV heads.
        std::vector<float> avg_q_cov(d * d, 0.0f);
        std::vector<float> avg_v_cov(d * d, 0.0f);

        for (int kvh = 0; kvh < nhkv; ++kvh) {
            for (int i = 0; i < d * d; ++i) {
                avg_q_cov[i] += acc.q_cov[kvh][i];
                avg_v_cov[i] += acc.v_cov[kvh][i];
            }
        }

        const float norm = 1.0f / (float)(acc.n_tokens > 0 ? acc.n_tokens : 1);
        for (int i = 0; i < d * d; ++i) {
            avg_q_cov[i] *= norm / nhkv;
            avg_v_cov[i] *= norm / nhkv;
        }

        // Write raw f32 binaries.
        char path[512];
        snprintf(path, sizeof(path), "%s/layer_%02d_qcov.bin", output_dir.c_str(), layer);
        FILE * f = fopen(path, "wb");
        if (f) { fwrite(avg_q_cov.data(), sizeof(float), d * d, f); fclose(f); }

        snprintf(path, sizeof(path), "%s/layer_%02d_vcov.bin", output_dir.c_str(), layer);
        f = fopen(path, "wb");
        if (f) { fwrite(avg_v_cov.data(), sizeof(float), d * d, f); fclose(f); }
    }
    LOG_INF("dumped covariances for %d layers to %s\n", g_n_layers, output_dir.c_str());
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    // Add custom args.
    params.cb_eval = calib_cb_eval;
    params.cb_eval_user_data = nullptr;
    params.warmup = false;

    // Parse -o <dir> before common_params_parse, then strip it so the
    // upstream parser does not reject it as an unknown argument.
    g_output_dir = "covariances";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-o" && i + 1 < argc) {
            g_output_dir = argv[i + 1];
            // shift remaining args over the -o pair
            for (int j = i; j < argc - 2; ++j) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            break;
        }
    }

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // Ensure output dir exists.
#ifdef _WIN32
    _mkdir(g_output_dir.c_str());
#else
    mkdir(g_output_dir.c_str(), 0755);
#endif

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model || !ctx) {
        LOG_ERR("failed to init model\n");
        return 1;
    }

    // Query model dimensions. n_embd_head is not exposed directly in the public
    // llm API, so derive it from n_embd / n_head.
    const int n_embd      = llama_model_n_embd(model);
    const int n_head      = llama_model_n_head(model);
    const int n_embd_head = n_embd / n_head;
    const int n_head_kv   = llama_model_n_head_kv(model);
    const int n_layers    = llama_model_n_layer(model);

    g_n_layers = n_layers;
    g_n_embd_head = n_embd_head;

    LOG_INF("model: head_dim=%d, n_head=%d, n_head_kv=%d, n_layers=%d\n",
            n_embd_head, n_head, n_head_kv, n_layers);

    // Initialize accumulators.
    g_accumulators.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        g_accumulators[i].init(n_embd_head, n_head, n_head_kv);
    }

    // Tokenize and run.
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("no input tokens\n");
        return 1;
    }

    LOG_INF("processing %zu tokens...\n", tokens.size());

    // Process in chunks to avoid OOM. Each chunk is an independent window:
    // clear the cache and restart positions at 0. The slot finder never evicts
    // cells of the same sequence, and M-RoPE (this model) requires strictly
    // increasing positions across decodes, so a cleared cache (X = -1 < Y = 0)
    // is the only pattern that works for corpora longer than n_ctx.
    const int n_ctx = llama_n_ctx(ctx);
    for (size_t i = 0; i < tokens.size(); i += n_ctx) {
        size_t n = std::min((size_t)n_ctx, tokens.size() - i);
        llama_memory_clear(llama_get_memory(ctx), false);
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (size_t t = 0; t < n; ++t) {
            batch.token[t]     = tokens[i + t];
            batch.pos[t]       = (llama_pos)t;
            batch.n_seq_id[t]  = 1;
            batch.seq_id[t][0] = 0;
            batch.logits[t]    = 1;
        }
        batch.n_tokens = n;
        if (llama_decode(ctx, batch)) {
            LOG_ERR("failed to decode at offset %zu\n", i);
            llama_batch_free(batch);
            return 1;
        }
        llama_batch_free(batch);
    }

    // Dump results.
    dump_covariances(g_output_dir);

    LOG_INF("done. %lld tokens processed.\n", (long long)g_accumulators[0].n_tokens);

    llama_backend_free();
    return 0;
}
