#!/bin/bash
# 32K bench matrix on anchor branch
cd /mnt/storage/Projects/llama-cpp-turboquant
B=./build-anchor-kv/bin
M=/mnt/storage/models/qwen3/Qwen3-8B-Q8_0.gguf
for ct in f16 anchor1 anchor2 anchor3; do
    extra=""
    [ "$ct" != "f16" ] && extra="-ctk $ct"
    $B/llama-bench -m $M -p 32768 -n 64 -r 1 -t 4 --no-warmup -fa 1 $extra > /tmp/b32-$ct.txt 2>&1
done
echo DONE > /tmp/b32-DONE.txt
