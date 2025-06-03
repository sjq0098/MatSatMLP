#!/bin/bash

# 代码文件名和可执行文件名
SRC="matmul_no_mpi.cpp"
EXE="matmul_no_mpi"

# 输出文件
OUTFILE="results.txt"

# 先删除旧的可执行文件（如果存在）
if [ -f "$EXE" ]; then
    rm "$EXE"
fi

# 编译 matmul_no_mpi.cpp
echo "正在编译 $SRC ..."
g++ -fopenmp -O3 -march=native -o "$EXE" "$SRC"
if [ $? -ne 0 ]; then
    echo "编译失败，请检查源码或编译选项。" 
    exit 1
fi
echo "编译成功，生成可执行文件：$EXE"

# 清空或新建 results.txt
> "$OUTFILE"

# 要测试的所有模式
MODES=("baseline" "openmp" "block" "other" "simd" "simd_openmp" "simd_block")

# 依次运行每个模式并将输出追加到 results.txt
for mode in "${MODES[@]}"; do
    echo "=== 模式: $mode ===" >> "$OUTFILE"
    ./"$EXE" "$mode" >> "$OUTFILE"
    echo "" >> "$OUTFILE"
done

echo "所有实验结果已写入 $OUTFILE"
