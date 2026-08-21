#!/bin/bash
# Trimmed AnchorKV benchmark suite - corrected version.
# Changes vs ~/Desktop/anchor-kv-benchmark.sh:
#   - contexts 2048 + 32768 only (256K f16 baseline cannot fit in 32GB VRAM)
#   - all perplexity runs bounded with --chunks (unbounded logits files killed the previous run)
#   - KL divergence only at 2048 (--chunks 1 keeps the logits file ~1.2GB)
set -u
M=/mnt/storage/models/qwen3/Qwen3-8B-Q8_0.gguf
WK=/mnt/storage/blackbeard/wikitext-2-raw/wiki.test.raw
B=/mnt/storage/Projects/llama-cpp-turboquant/build-anchor-kv/bin
OUT=/home/jabbatheduck/Desktop/anchor-kv-report-trimmed-$(date +%Y%m%d-%H%M%S).md
T=4

names=(f16 anchor1 anchor2 anchor3 mix)
labels=("f16 (baseline)" "anchor1 (5x)" "anchor2 (10x)" "anchor3 (20x)" "anchor2 K + anchor3 V")

bench_args() {
    case $1 in
        f16)     echo "" ;;
        mix)     echo "-ctk anchor2 -ctv anchor3" ;;
        *)       echo "-ctk $1" ;;
    esac
}

echo "# AnchorKV Benchmark Report (trimmed)" > "$OUT"
echo "**Date:** $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT"
echo "**Model:** $(basename "$M") | **Build:** $(git -C /mnt/storage/Projects/llama-cpp-turboquant log -1 --format=%h)" >> "$OUT"

for CTX in 2048 32768; do
    echo "" >> "$OUT"; echo "## Context: $CTX" >> "$OUT"; echo "" >> "$OUT"

    echo "--- llama-bench ---"
    echo "### llama-bench" >> "$OUT"; echo "" >> "$OUT"
    echo "| Config | pp (t/s) | tg (t/s) |" >> "$OUT"
    for i in "${!names[@]}"; do
        n=${names[$i]}
        echo "  [bench $n] ctx=$CTX"
        RES=$("$B/llama-bench" -m "$M" -p "$CTX" -n 128 -r 2 -t $T $(bench_args "$n") 2>&1 | grep -E "^\| qwen3")
        PP=$(echo "$RES" | grep "pp" | grep -oP '\|\s*\K[\d.]+(?=\s+±)' | head -1)
        TG=$(echo "$RES" | grep "tg" | grep -oP '\|\s*\K[\d.]+(?=\s+±)' | head -1)
        echo "| ${labels[$i]} | ${PP:-FAIL} | ${TG:-FAIL} |" >> "$OUT"
        echo "$RES" >> "$OUT.raw.txt"
    done

    echo "--- llama-perplexity (--chunks 4) ---"
    echo "" >> "$OUT"; echo "### llama-perplexity (chunks=4)" >> "$OUT"; echo "" >> "$OUT"
    echo "| Config | PPL |" >> "$OUT"
    for i in "${!names[@]}"; do
        n=${names[$i]}
        echo "  [ppl $n] ctx=$CTX"
        RES=$("$B/llama-perplexity" -m "$M" -c "$CTX" -ngl -1 -t $T --chunks 4 \
              $(bench_args "$n") -f "$WK" 2>&1)
        PPL=$(echo "$RES" | grep -i "Final estimate" | grep -oP '[\d.]+' | head -1)
        echo "| ${labels[$i]} | ${PPL:-FAIL} |" >> "$OUT"
        echo "$RES" | tail -3 >> "$OUT.raw.txt"
    done

    if [ "$CTX" = "2048" ]; then
        echo "--- KL divergence (chunks=1) ---"
        BASE=/tmp/anchor-kl-base-$CTX.bin
        echo "  [kl baseline]"
        "$B/llama-perplexity" -m "$M" -c "$CTX" -ngl -1 -t $T --chunks 1 \
            -f "$WK" --save-all-logits "$BASE" > /dev/null 2>&1
        echo "" >> "$OUT"; echo "### KL divergence (ctx=2048, chunks=1)" >> "$OUT"; echo "" >> "$OUT"
        echo "| Config | PPL | KL |" >> "$OUT"
        for i in "${!names[@]}"; do
            n=${names[$i]}
            [ "$n" = "f16" ] && { echo "| f16 (baseline) | - | 0 |" >> "$OUT"; continue; }
            echo "  [kl $n]"
            RES=$("$B/llama-perplexity" -m "$M" -c "$CTX" -ngl -1 -t $T --chunks 1 \
                  $(bench_args "$n") -f "$WK" \
                  --kl-divergence --kl-divergence-base "$BASE" 2>&1)
            PPL=$(echo "$RES" | grep -i "Final estimate" | grep -oP '[\d.]+' | head -1)
            KLD=$(echo "$RES" | grep -i "KL divergence" | tail -1 | grep -oP '[\d.]+$' | head -1)
            echo "| ${labels[$i]} | ${PPL:-FAIL} | ${KLD:-FAIL} |" >> "$OUT"
        done
        rm -f "$BASE"
    fi
done

echo "" >> "$OUT"
echo "=== Suite complete: $OUT ==="
